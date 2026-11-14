"""Distributed contraction: one logical tensor partitioned across P = 2^p GPUs (after arXiv:2606.01852).

Slicing fixes a label for the whole tree and repeats the work for both values. Distribution fixes a label
only while it helps: GPU g holds the shard T[x = bit_i(g)] of a large intermediate T for the label x of
"slot" i, contracts locally, and when another label y becomes the better one to split on, the GPUs
exchange half of their shards (redistribution) and continue with T[y = bit_i(g)].

Model used here (all labels have dimension 2, P = 2^p GPUs, "slots" i = 0..p-1 = the bits of the GPU index g):
  * For every tensor of the tree and every slot, the tensor is in one of three states:
      None   - replicated: the GPUs of a pair (g, g + 2^i) hold the same tensor;
      x      - split along its label x: GPU g holds the shard T[x = bit_i(g)] (x is fixed in the leaves below);
      summed - x was summed below: each GPU holds a partial sum, and the OR over the pair is the tensor.
    Two operands can be contracted locally if, per slot, one of them is replicated or both are split along
    the same label. A slot that is never exchanged is exactly a sliced label, summed at the root.
  * An exchange on a tensor replaces the label x of slot i by another label y of that tensor: every GPU
    moves y to the top bit of its shard, then each pair swaps the half [y=1] of g with the half [y=0] of
    g + 2^i in place. Afterwards y is fixed and the top bit is x again (x live), or, if x was dead, the
    two halves are OR-ed (the pair's partial sums are merged).
  * Which labels to split on and where to exchange is chosen by dynamic programming over the tree with
    the fitted H100 kernel model plus a link model (bandwidth, latency) for the exchanges.

plan_distribution() -> Schedule (planner; CPU only)
emulate()            numpy reference execution of a Schedule (tests)
DistPlan             compiled execution on the GPUs (ctypes; one library copy and one thread per device)
"""
import ctypes
import os
import shutil
import tempfile
import time

import numpy as np

from .tree_utils import h100_time, H100_MODEL

# link between two GPUs: bytes/s in each direction and fixed seconds per exchange
LINK_MODEL = dict(bw=170e9, latency=3e-4)      # measured between two H800 (8 NVLink lanes): 170 GB/s each way


def _out(t, ixs):
    return ixs[t["tensor_index"]] if "tensor_index" in t else t["eins"]["iy"]


def _bmnk(a, b, c):
    return len(a & b & c), len((a - b) & c), len((b - a) & c), len((a & b) - c)


def _ctime(a, b, c, model):
    B, M, N, K = _bmnk(a, b, c)
    return h100_time(B, M, N, K, model, small=(B + M + N) <= model["small_rank"])


def exchange_time(rank, link=LINK_MODEL, model=H100_MODEL):
    """Modeled seconds to exchange half of a local shard of 2^rank bits with the pair GPU: move the new
    label to the top (one permute), swap half the shard over the link, and one more permute because the
    consumer cannot take the exchanged layout directly."""
    bits = 2.0 ** rank
    perm = 2 * bits / model["perm_bw_bits"]
    return 2 * perm + (bits / 16) / link["bw"] + link["latency"]


# ------------------------------------------------------------------------------------------------
# planner
# ------------------------------------------------------------------------------------------------
class Schedule:
    """Result of plan_distribution.

    ltree      the contraction every GPU runs: {"tensor_index": i}, {"args": [l, r], "eins": {"iy": local
               labels}} and exchange nodes {"xchg": child, "slot", "top", "newtop", "rank", "eins"}
    leaf_slots leaf_slots[i] = {label: slot}: labels fixed in leaf i to the slot's bit of the GPU index
    exchanges  [dict(slot, top, newtop, rank)] in execution order (rank = log2 bits of the local shard)
    model_s    modeled seconds per GPU for one contraction (kernels + exchanges)
    """

    def __init__(self, ltree, leaf_slots, p, cap, model_s, exchanges, ixs):
        self.ltree, self.leaf_slots, self.p, self.cap = ltree, leaf_slots, p, cap
        self.model_s, self.exchanges, self.ixs = model_s, exchanges, ixs

    def exchanged_bytes(self):
        """Bytes one GPU sends over all exchanges of one contraction."""
        return sum(2 ** x["rank"] // 16 for x in self.exchanges)

    def exchange_model_s(self, link=LINK_MODEL, model=H100_MODEL):
        return sum(exchange_time(x["rank"], link, model) for x in self.exchanges)

    def max_local_rank(self):
        best = 0
        stack = [self.ltree]
        while stack:
            t = stack.pop()
            if "tensor_index" in t:
                continue
            best = max(best, len(t["eins"]["iy"]))
            stack.extend([t["xchg"]] if "xchg" in t else t["args"])
        return best


def plan_distribution(tree, ixs, p, cap, link=LINK_MODEL, model=H100_MODEL, ncand=None, beam=None,
                      exchanges=True, min_xrank=8, nsrc=None):
    """Choose what every tensor is split along and where to exchange, for P = 2^p GPUs.

    tree: omeco dict with outputs recomputed for the network `ixs` (global slices removed);
    cap: largest local tensor, log2 bits. exchanges=False plans without redistribution: the best p labels
    to slice on this tree (slice-parallel execution of the same tree).
    min_xrank: smallest local tensor (log2 bits) that may be exchanged (the GPU swaps whole words).
    Returns a Schedule, or None if no assignment keeps every local tensor within `cap`.

    Dynamic program over the tree, bottom-up. A state of a tensor is (set of labels it is split along,
    number of slots holding partial sums); the remaining slots are replicated. Slots are told apart only
    when the plan is written out. For p = 1 the search is exact; for p > 1 large tables keep the `beam`
    cheapest states per group (see groups_of), exchanges are tried from the cheapest states to the
    `ncand` longest-lived labels, and the plan without exchanges is returned when the pruned search did
    not beat it.
    """
    if p <= 1 or ncand is not None or beam is not None or nsrc is not None:
        best = _plan(tree, ixs, p, cap, link, model, ncand, beam, exchanges, min_xrank, nsrc)
        if p > 1 and exchanges:
            alt = _plan(tree, ixs, p, cap, link, model, ncand, beam, False, min_xrank, nsrc)
            if alt is not None and (best is None or alt.model_s <= best.model_s):
                best = alt
        return best
    # p > 1: the search is pruned. Seed it with the greedy slicer's labels when they fit in the p slots (the
    # search then never does worse than that slicing), and widen it until the plan stops improving.
    from .slicer import greedy_slice
    seed = None
    try:
        g = greedy_slice(tree, ixs, cap, kind="h100")[0]
        if len(g) <= p:
            seed = frozenset(g)
    except (TypeError, ValueError):      # no label left to slice
        pass
    best = None
    for b, c, n_ in ((12, 6, 8), (40, 12, 24), (120, 20, 60)):
        prev = best
        for ex in ((False, True) if exchanges else (False,)):
            sch = _plan(tree, ixs, p, cap, link, model, c, b, ex, min_xrank, n_, seed)
            if sch is not None and (best is None or sch.model_s < best.model_s):
                best = sch
        if prev is not None and best.model_s > 0.98 * prev.model_s:
            break
    return best


def _plan(tree, ixs, p, cap, link, model, ncand, beam, exchanges, min_xrank, nsrc, seed=None):
    import sys
    if ncand is None:
        ncand = 1 << 30 if p == 1 else 6
    if beam is None:
        beam = 1 << 30 if p == 1 else 12
    if nsrc is None:
        nsrc = 1 << 30 if p == 1 else 8
    keep_all = 600                     # tables up to this size are never pruned
    # post-order, depth and parent of every node, depth at which each label is summed (for lifetimes)
    order, depth, parent, stack = [], {id(tree): 0}, {}, [(tree, False)]
    while stack:
        t, done = stack.pop()
        if done or "tensor_index" in t:
            order.append(t)
            continue
        stack.append((t, True))
        for c in t["args"]:
            depth[id(c)] = depth[id(t)] + 1
            parent[id(c)] = t
            stack.append((c, False))
    sum_depth = {}
    for t in order:
        if "tensor_index" not in t:
            a, b = t["args"]
            for l in (set(_out(a, ixs)) | set(_out(b, ixs))) - set(t["eins"]["iy"]):
                sum_depth[l] = depth[id(t)]

    def groups_of(T, nxt):
        """States grouped by (#split labels, #partial-sum slots, #split labels that survive the next
        contraction); each group sorted cheapest first, among equals the longest-lived labels first."""
        groups = {}
        for st, e in T.items():
            L = st[0]
            groups.setdefault((len(L), st[1], len(L & nxt) if L else 0), []).append((st, e))
        for g in groups.values():
            g.sort(key=lambda se: (se[1][0], sum(sum_depth.get(l, -1) for l in se[0][0])))
        return groups

    def prune(T, nxt, lc=None):
        if len(T) <= keep_all:
            return T
        out = {}
        for g in groups_of(T, nxt).values():
            out.update(g[:beam])
        if seed is not None and lc is not None:      # the seed slicing's state of this tensor
            want = seed & lc
            for st, e in T.items():
                if st[0] == want:
                    out[st] = e
        return out

    tab = {}
    for t in order:
        if "tensor_index" in t:
            i = t["tensor_index"]
            labs = list(dict.fromkeys(ixs[i]))
            subs = [frozenset()]
            for l in labs:
                subs += [q | {l} for q in subs if len(q) < p]
            tab[id(t)] = {(s, 0): (0.0, "leaf", i, s) for s in subs}
            continue
        a, b = t["args"]
        Ta, Tb = tab.pop(id(a)), tab.pop(id(b))
        la, lb, lc = set(_out(a, ixs)), set(_out(b, ixs)), set(t["eins"]["iy"])
        nxt = set(parent[id(t)]["eins"]["iy"]) if id(t) in parent else set()
        tcache = {}
        T = {}
        # compatible pairs: per slot one side is replicated or both are split along the same label, and a
        # label held by one side only is not a label of the other operand  <=>  La & lb == Lb & la
        idx = {}
        for (Lb, db), eb in Tb.items():
            idx.setdefault(Lb & la if Lb else Lb, []).append((Lb, db, eb))
        for (La, da), ea in Ta.items():
            for Lb, db, eb in idx.get(La & lb if La else La, ()):
                F = La | Lb
                nF = len(F)
                if nF + da + db > p:
                    continue
                tc = tcache.get(F)
                if tc is None:
                    a2, b2, c2 = la - F, lb - F, lc - F
                    tc = tcache[F] = (_ctime(a2, b2, c2, model), max(len(a2), len(b2), len(c2)), F & lc)
                if tc[1] > cap:
                    continue
                live = tc[2]
                st = (live, da + db + nF - len(live))
                c = ea[0] + eb[0] + tc[0]
                o = T.get(st)
                if o is None or c < o[0]:
                    T[st] = (c, "join", t, ea, eb, F, da, db)
        if not T:
            return None
        T = prune(T, nxt, lc)
        if exchanges and t is not tree and len(lc) >= min_xrank:
            cand = sorted(lc, key=lambda l: (sum_depth.get(l, -1), str(l)))[:ncand]   # longest-lived first
            iy_full = t["eins"]["iy"]
            for _ in range(p):
                if len(T) <= nsrc:
                    src = list(T.items())
                else:                        # the cheapest states overall and of every group
                    src = dict(sorted(T.items(), key=lambda kv: kv[1][0])[:nsrc])
                    for g in groups_of(T, nxt).values():
                        src.update(g[:2])
                    src = list(src.items())
                for (L, nd), e in src:
                    if not L and not nd:
                        continue
                    r = len(lc) - len(L)
                    if r < min_xrank:
                        continue
                    c = e[0] + exchange_time(r, link, model)
                    for y in cand:
                        if y in L:
                            continue
                        if nd > 0:           # merge the partial sums of a slot and split along y
                            st = (L | {y}, nd - 1)
                            o = T.get(st)
                            if o is None or c < o[0]:
                                T[st] = (c, "xchg", e, y, None, r, iy_full)
                        for x in L:          # replace the split label x by y
                            st = ((L - {x}) | {y}, nd)
                            o = T.get(st)
                            if o is None or c < o[0]:
                                T[st] = (c, "xchg", e, y, x, r, iy_full)
            T = prune(T, nxt, lc)
        tab[id(t)] = T
    best = min(tab[id(tree)].values(), key=lambda e: e[0])
    # write the plan out: local tree, fixed leaf labels with their slots, exchanges in execution order
    leaf_slots, xs = {}, []
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * len(order) + 1000))

    def build(e, slot_of, dead):
        """e: table entry; slot_of: {split label: slot} of its state; dead: slots holding partial sums."""
        if e[1] == "leaf":
            leaf_slots[e[2]] = dict(slot_of)
            return {"tensor_index": e[2]}
        if e[1] == "join":
            _, _, node, ea, eb, F, da, db = e
            dead = list(dead)
            pre = dict(slot_of)
            for l in F:
                if l not in pre:                 # summed at this node: one of the partial-sum slots was its slot
                    pre[l] = dead.pop()
            sa, sb = state_of(ea), state_of(eb)
            ta = build(ea, {l: pre[l] for l in sa[0]}, [dead.pop() for _ in range(sa[1])])
            tb = build(eb, {l: pre[l] for l in sb[0]}, [dead.pop() for _ in range(sb[1])])
            return {"args": [ta, tb], "eins": {"iy": [l for l in node["eins"]["iy"] if l not in F]}}
        _, _, prev, y, x, r, iy_full = e
        slot = slot_of[y]
        prev_slots = {l: s for l, s in slot_of.items() if l != y}
        prev_dead = list(dead)
        if x is None:
            prev_dead.append(slot)
        else:
            prev_slots[x] = slot
        child = build(prev, prev_slots, prev_dead)
        liy = [l for l in iy_full if l not in prev_slots and l != y] + ([] if x is None else [x])
        rec = dict(slot=slot, top=y, newtop=x, rank=r)
        xs.append(rec)
        return dict(rec, xchg=child, eins={"iy": liy})

    def state_of(e):
        """(split labels, number of partial-sum slots) of the tensor an entry describes."""
        if e[1] == "leaf":
            return e[3], 0
        if e[1] == "join":
            F, lc = e[5], set(e[2]["eins"]["iy"])
            live = F & lc
            return live, e[6] + e[7] + len(F) - len(live)
        L, nd = state_of(e[2])
        y, x = e[3], e[4]
        return ((L | {y}, nd - 1) if x is None else ((L - {x}) | {y}, nd))

    L, nd = state_of(best)
    assert not L
    ltree = build(best, {}, list(range(nd)))
    return Schedule(ltree, leaf_slots, p, cap, best[0], xs, ixs)


# ------------------------------------------------------------------------------------------------
# local network: what every GPU contracts
# ------------------------------------------------------------------------------------------------
def local_segments(sch):
    """The contraction each GPU runs, cut at the exchanges.

    Returns the segments in execution order. segments[j] = dict(tree, xchg): `tree` is a binary tree in
    omeco dict form with local output labels whose leaves are network tensors {"tensor_index": i} or
    {"carry": j'}, the tensor left by segment j' < j after its exchange (a tree may be just such a leaf).
    xchg = dict(slot, top, newtop, rank) or None (last segment): after the tree, move label `top` to the
    top bit and swap half-shards within the GPU pairs of `slot`; the top bit then is the label `newtop`
    (the old slot label, still live), or with newtop None the two halves are OR-ed (partial sums)."""
    import sys
    segments = []
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

    def cut(t):
        if "tensor_index" in t:
            return t
        if "xchg" in t:
            sub = cut(t["xchg"])
            segments.append(dict(tree=sub, xchg={k: t[k] for k in ("slot", "top", "newtop", "rank")}))
            return {"carry": len(segments) - 1}
        return {"args": [cut(t["args"][0]), cut(t["args"][1])], "eins": t["eins"]}

    root = cut(sch.ltree)
    segments.append(dict(tree=root, xchg=None))
    return segments


def renamed_clauses(clauses, n, leaf_slots, p):
    """Clauses in which every slot-fixed variable is renamed to the virtual variable n + slot (0-based), so
    that a slicer over [global sliced vars] + [n .. n+p-1] yields each GPU's leaves from the slice index
    s | g << len(sliced)."""
    out = []
    for i, c in enumerate(clauses):
        m = leaf_slots.get(i, {})
        out.append([(1 if l > 0 else -1) * (n + m[abs(l) - 1] + 1) if (abs(l) - 1) in m else l for l in c])
    return out


# ------------------------------------------------------------------------------------------------
# numpy reference execution (small networks; tests)
# ------------------------------------------------------------------------------------------------
def emulate(sch, clauses, n, sliced=(), slice_index=0):
    """Execute the schedule with numpy bool arrays for all 2^p GPUs; returns the OR of the GPU results."""
    segments, leaf_slots = local_segments(sch), sch.leaf_slots
    P = 1 << sch.p
    gfix = {v: (slice_index >> i) & 1 for i, v in enumerate(sliced)}

    def leaf(i, g):
        fixed = dict(gfix)
        for l, s in leaf_slots.get(i, {}).items():
            fixed[l] = (g >> s) & 1
        c = clauses[i]
        labels = [abs(l) - 1 for l in c if (abs(l) - 1) not in fixed]
        arr = np.ones((2,) * len(labels), dtype=bool)
        if not any((abs(l) - 1) in fixed and fixed[abs(l) - 1] == (1 if l > 0 else 0) for l in c):
            arr[tuple(1 if l < 0 else 0 for l in c if (abs(l) - 1) not in fixed)] = False
        return arr, labels

    def contract(A, B, iy):
        (a, la), (b, lb) = A, B
        ids = {l: i for i, l in enumerate(dict.fromkeys(la + lb))}
        r = np.einsum(a.astype(np.int64), [ids[l] for l in la], b.astype(np.int64), [ids[l] for l in lb],
                      [ids[l] for l in iy])
        return r > 0, list(iy)

    carries = {}

    def rec(t):
        if "carry" in t:
            return carries.pop(t["carry"])
        if "tensor_index" in t:
            return [leaf(t["tensor_index"], g) for g in range(P)]
        A, B = rec(t["args"][0]), rec(t["args"][1])
        return [contract(A[g], B[g], t["eins"]["iy"]) for g in range(P)]

    for j, seg in enumerate(segments):
        vals = rec(seg["tree"])
        x = seg["xchg"]
        if x is None:
            return any(bool(a.reshape(-1)[0]) for a, _ in vals)
        i, y = x["slot"], x["top"]
        moved = []
        for (a, la) in vals:
            ax = la.index(y)
            moved.append((np.moveaxis(a, ax, -1), la[:ax] + la[ax + 1:]))
        out = [None] * P
        for g in range(P):
            if (g >> i) & 1:
                continue
            h = g | (1 << i)
            (a0, l0), (a1, l1) = moved[g], moved[h]
            assert l0 == l1
            new_g = np.stack([a0[..., 0], a1[..., 0]], axis=-1)     # g keeps y = 0 of both
            new_h = np.stack([a0[..., 1], a1[..., 1]], axis=-1)     # h keeps y = 1 of both
            if x["newtop"] is None:
                out[g], out[h] = (new_g.any(axis=-1), l0), (new_h.any(axis=-1), l0)
            else:
                out[g], out[h] = (new_g, l0 + [x["newtop"]]), (new_h, l0 + [x["newtop"]])
        carries[j] = out


# ------------------------------------------------------------------------------------------------
# GPU execution
# ------------------------------------------------------------------------------------------------
class Program:
    """Device-independent compilation of the local contraction (the same on every GPU): arena layout,
    the kernel launches of every segment, and where the exchanged buffer lies.

    This is engine.CompiledTree applied segment by segment over one shared arena: the stem tensor a
    segment leaves behind (after the exchange) is the `carry` leaf of the next one."""

    def __init__(self, segments, leaf_labels, small_rc=14):
        from . import engine as E
        arena = E.Arena()
        assert all(len(l) <= 16 for l in leaf_labels), "leaf tensors must have rank <= 16"
        leaf_words = [E.nwords(len(l)) for l in leaf_labels]
        self.leaf_word_off = np.concatenate([[0], np.cumsum(leaf_words)[:-1]]).astype(np.int64)
        self.leaf_total_words = int(sum(leaf_words))
        self.leaf_off, _ = arena.alloc(4 * self.leaf_total_words)
        self.flag_off, _ = arena.alloc(4)
        self.arena = arena
        self.small_rc = small_rc
        self.tensors = [E.PlanTensor(l, self.leaf_off + 4 * int(self.leaf_word_off[i]), None)
                        for i, l in enumerate(leaf_labels)]
        self.segs = []           # dict(levels, ops, records, xchg=None | dict(off, half_bytes, slot, fold))
        carries = {}             # segment index -> PlanTensor it left behind
        for j, seg in enumerate(segments):
            root, levels, ops, records = self._compile(seg["tree"], carries)
            x = seg["xchg"]
            xs = None
            if x is not None:
                y = x["top"]
                want = [l for l in root.labels if l != y] + [y]
                if want != root.labels:
                    pos = {l: i for i, l in enumerate(root.labels)}
                    off, tid = arena.alloc(4 * E.nwords(root.r))
                    ops.append(("perm", root.off, root.r, off, root.r, [pos[l] for l in want], []))
                    if root.tid is not None:
                        arena.free(root.tid)
                    root = E.PlanTensor(want, off, tid)
                assert root.r >= 6, "exchanged tensors need at least 2 words"
                half = 4 * E.nwords(root.r) // 2
                xs = dict(off=root.off, half_bytes=half, slot=x["slot"], fold=x["newtop"] is None, rank=root.r)
                labels = root.labels[:-1] + ([] if x["newtop"] is None else [x["newtop"]])
                carries[j] = E.PlanTensor(labels, root.off, root.tid)
            else:
                self.root_off = root.off
            self.segs.append(dict(levels=levels, ops=ops, records=records, xchg=xs))
        self.arena_bytes = arena.top

    def _compile(self, tree, carries):
        """Plan one segment tree. Returns (root PlanTensor (not freed), small levels, ops, records)."""
        from . import engine as E
        arena, small_rc = self.arena, self.small_rc
        leaves = list(self.tensors)

        def subst(t):
            if "carry" in t:
                leaves.append(carries.pop(t["carry"]))
                return {"tensor_index": len(leaves) - 1}
            if "tensor_index" in t:
                return t
            return {"args": [subst(t["args"][0]), subst(t["args"][1])], "eins": t["eins"]}

        tree = subst(tree)
        ops, records = [], []

        def talloc(labels):
            off, tid = arena.alloc(4 * E.nwords(len(labels)))
            return E.PlanTensor(labels, off, tid)

        def tfree(t):
            if t.tid is not None:
                arena.free(t.tid)

        def permute_spec(t, new_labels, red=()):
            if list(new_labels) == t.labels and not red:
                return t
            out = talloc(new_labels)
            pos = {l: i for i, l in enumerate(t.labels)}
            ops.append(("perm", t.off, t.r, out.off, out.r, [pos[l] for l in new_labels], [pos[l] for l in red]))
            return out

        def contract_planned(d, a, b):
            I, O = info[id(d)], orders[id(d)]
            A, B = (b, a) if I["swap"] else (a, b)
            kind = I["kind"]
            Kt, Mt, Nt, Bt = O["K"], O["M"], O["N"], O["B"]
            nops0 = len(ops)
            Ao = Kt + Mt + Bt
            Bo = Nt + Kt + Bt if I["nfirst"] else Kt + Nt + Bt
            Co = Nt + Mt + Bt
            A2 = permute_spec(A, Ao, I["Ra"])
            B2 = permute_spec(B, Bo, I["Rb"])
            if kind == "tcp":
                kk = len(Kt)
                dummy = [("pad", i) for i in range(7 - kk)]
                A3 = talloc(Kt + dummy + Mt + Bt); B3 = talloc(Kt + dummy + Nt + Bt)
                ops.append(("pad", A2.off, A3.off, len(Mt) + len(Bt), kk))
                ops.append(("pad", B2.off, B3.off, len(Nt) + len(Bt), kk))
                for t in (A2, B2):
                    if t is not A and t is not B:
                        tfree(t)
                A2, B2 = A3, B3
            C = talloc(Co)
            if kind in ("tc", "tcp"):
                ops.append(("tc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt) if kind == "tc" else 7))
            else:
                ops.append(("cc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            for t in (A2, B2):
                if t is not A and t is not B:
                    tfree(t)
            records.append((kind, len(Bt), len(Mt), len(Nt), len(Kt), nops0, len(ops)))
            return C

        small, levels = {}, {}

        def is_small_shape(la, lb, lc):
            sa, sb, sc = set(la), set(lb), set(lc)
            nk = len((sa | sb) - sc)
            return len(lc) <= small_rc and nk <= 8 and len(la) <= 24 and len(lb) <= 24 and len(lc) + nk <= small_rc + 2

        def mark(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]].labels, 0
            ra, rb = mark(d["args"][0]), mark(d["args"][1])
            if ra is None or rb is None:
                return None
            lc = d["eins"]["iy"]
            if not is_small_shape(ra[0], rb[0], lc):
                return None
            lev = 1 + max(ra[1], rb[1])
            small[id(d)] = (lc, lev)
            return lc, lev

        if small_rc > 0:
            mark(tree)
        info, orders, smallwant = E.plan_layouts(
            tree,
            lambda d: leaves[d["tensor_index"]].labels if "tensor_index" in d else
            (small[id(d)][0] if id(d) in small else d["eins"]["iy"]),
            lambda d: id(d) in small,
            lambda Bt, Kt, Mt, Nt: (lambda sk: (sk[0], sk[1], sk[1] == "cc"))(E.choose_kind(Bt, Kt, Mt, Nt)))

        def tensor_of(d):
            return leaves[d["tensor_index"]] if "tensor_index" in d else small[id(d)]

        def build_small(d):
            if "tensor_index" in d or not isinstance(small.get(id(d)), tuple):
                return
            build_small(d["args"][0]); build_small(d["args"][1])
            lc, lev = small[id(d)]
            lc = smallwant.get(id(d), lc)
            A, B_ = tensor_of(d["args"][0]), tensor_of(d["args"][1])
            off, _ = arena.alloc(4 * E.nwords(len(lc)))
            C = E.PlanTensor(lc, off, None)
            pa = {l: i for i, l in enumerate(A.labels)}
            pb = {l: i for i, l in enumerate(B_.labels)}
            summed = [l for l in dict.fromkeys(A.labels + B_.labels) if l not in set(lc)]
            levels.setdefault(lev, []).append((A.off, B_.off, C.off, len(lc), len(summed),
                                               [pa.get(l, -1) for l in lc], [pb.get(l, -1) for l in lc],
                                               [pa.get(l, -1) for l in summed], [pb.get(l, -1) for l in summed]))
            small[id(d)] = C

        def walk_small(d):
            if "tensor_index" in d:
                return
            if id(d) in small:
                build_small(d)
                return
            walk_small(d["args"][0]); walk_small(d["args"][1])

        walk_small(tree)

        def rec(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]]
            if id(d) in small:
                return small[id(d)]
            a, b = rec(d["args"][0]), rec(d["args"][1])
            c = contract_planned(d, a, b)
            tfree(a); tfree(b)
            return c

        return rec(tree), levels, ops, records


def _setup_api(L):
    """ctypes signatures of a libbooltn_mg copy (engine.lib()'s plus the multi-GPU functions)."""
    from . import engine as E
    c = ctypes
    L.bt_malloc.restype = c.c_void_p
    L.bt_malloc.argtypes = [c.c_size_t]
    L.bt_free.argtypes = [c.c_void_p]
    L.bt_h2d.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
    L.bt_d2h.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
    L.bt_memset.argtypes = [c.c_void_p, c.c_int, c.c_size_t]
    L.bt_event_create.restype = c.c_void_p
    L.bt_event_record.argtypes = [c.c_void_p]
    L.bt_event_elapsed.restype = c.c_float
    L.bt_event_elapsed.argtypes = [c.c_void_p, c.c_void_p]
    L.bt_fill_random.argtypes = [c.c_void_p, c.c_longlong, c.c_uint, c.c_int]
    E._setup_plan_api(L)
    L.bt_mg_init.argtypes = [c.c_int]
    L.bt_mg_enable_peer.argtypes = [c.c_int, c.c_int]
    L.bt_mg_swap.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_int, c.c_size_t, c.c_size_t]
    L.bt_mg_or_halves.argtypes = [c.c_void_p, c.c_longlong]
    L.bt_mg_mem_info.argtypes = [c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)]


class _Device:
    """One GPU: a private copy of the kernel library and the only thread that calls into it.

    libbooltn keeps one stream per loaded copy, and the CUDA driver keeps the current context per thread,
    so every device gets its own copy of libbooltn_mg.so and its own thread; all calls into the copy go
    through call() / submit()."""

    def __init__(self, dev):
        from concurrent.futures import ThreadPoolExecutor
        self.dev = int(dev)
        self.ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"booltn-gpu{dev}")
        self.L = self.ex.submit(self._load).result()

    def _load(self):
        from . import engine as E
        src = os.path.join(E.LIB_DIR, "libbooltn_mg.so")
        if not os.path.exists(src):
            raise OSError(f"{src} not found: build it with `make -C kernels mg`")
        tmp = tempfile.mkdtemp(prefix="booltn_mg_")
        dst = os.path.join(tmp, f"libbooltn_mg.{self.dev}.so")
        shutil.copy(src, dst)
        L = ctypes.CDLL(dst)
        shutil.rmtree(tmp, ignore_errors=True)       # the mapping stays valid
        _setup_api(L)
        if L.bt_mg_init(self.dev) != 0:
            raise RuntimeError(f"bt_mg_init({self.dev}) failed")
        return L

    def submit(self, fn, *args):
        return self.ex.submit(fn, *args)

    def call(self, fn, *args):
        return self.ex.submit(fn, *args).result()


_DEVICES = {}


def device(dev):
    """The _Device of GPU `dev` (created on first use)."""
    if dev not in _DEVICES:
        _DEVICES[dev] = _Device(dev)
    return _DEVICES[dev]


def device_count():
    d = device(0)
    return d.call(d.L.bt_mg_device_count)


class _Rank:
    """One GPU's instance of a Program: arena on its device and one compiled plan per segment."""

    def __init__(self, prog, dev, use_graph=True):
        self.prog, self.dev, self.use_graph = prog, dev, use_graph
        self.device = device(dev)
        self.L = self.device.L
        self.device.call(self._build)

    def _build(self):
        """(device thread) allocate the arena and record the plans."""
        from . import engine as E
        prog, dev, L = self.prog, self.dev, self.L
        self.base = L.bt_malloc(prog.arena_bytes)
        if not self.base:
            raise MemoryError(f"arena of {prog.arena_bytes / 2**30:.2f} GiB failed on device {dev}")
        B = self.base
        self.plans = []
        for si, seg in enumerate(prog.segs):
            P = L.bt_plan_new()
            self.plans.append(P)
            for lev in sorted(seg["levels"]):
                lst = seg["levels"][lev]
                n = len(lst)
                ptrs = (ctypes.c_void_p * (3 * n))()
                rc = (ctypes.c_int * n)(); nk = (ctypes.c_int * n)()
                apos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n)); bpos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n))
                ka = (ctypes.c_int * (8 * n))(*([-1] * 8 * n)); kb = (ctypes.c_int * (8 * n))(*([-1] * 8 * n))
                for i, (ao, bo, co, rci, nki, ap, bp, kai, kbi) in enumerate(lst):
                    ptrs[3 * i], ptrs[3 * i + 1], ptrs[3 * i + 2] = B + ao, B + bo, B + co
                    rc[i], nk[i] = rci, nki
                    for q, v in enumerate(ap): apos[24 * i + q] = v
                    for q, v in enumerate(bp): bpos[24 * i + q] = v
                    for q, v in enumerate(kai): ka[8 * i + q] = v
                    for q, v in enumerate(kbi): kb[8 * i + q] = v
                if L.bt_plan_small_batch(P, n, ptrs, rc, nk, apos, bpos, ka, kb):
                    raise RuntimeError("small batch failed")
            for op in seg["ops"]:
                if op[0] == "perm":
                    _, ioff, rin, ooff, rout, src, red = op
                    r = L.bt_plan_permute(P, B + ioff, rin, B + ooff, rout, E.IntArr(src), E.IntArr(red), len(red))
                elif op[0] == "tc":
                    _, a, b, c, bb, mm, nn, kk = op
                    r = L.bt_plan_tc(P, B + a, B + b, B + c, bb, mm, nn, kk, -1)
                elif op[0] == "pad":
                    _, ioff, ooff, lrows, kk = op
                    r = L.bt_plan_pad_rows(P, B + ioff, B + ooff, lrows, kk)
                else:
                    _, a, b, c, bb, mm, nn, kk = op
                    r = L.bt_plan_cc(P, B + a, B + b, B + c, bb, mm, nn, kk)
                if r != 0:
                    raise RuntimeError(f"plan op {op[0]} failed rc={r}")
            if si == len(prog.segs) - 1:
                L.bt_plan_or_accum(P, B + prog.flag_off, B + prog.root_off)
        L.bt_memset(B + prog.flag_off, 0, 4)

    def set_leaves(self, words):
        w = np.ascontiguousarray(words, dtype=np.uint32)
        assert w.size == self.prog.leaf_total_words
        self._leafbuf = w
        self.L.bt_h2d(self.base + self.prog.leaf_off, w.ctypes.data, w.nbytes)

    def run(self, si, sync=False):
        """Launch segment si (asynchronous unless sync: returns once the launches are queued on the device)."""
        if self.L.bt_plan_size(self.plans[si]) > 0 and self.L.bt_plan_run(self.plans[si], 1 if self.use_graph else 0):
            raise RuntimeError("plan run failed")
        if sync:
            self.sync()

    def fold(self, off, nwords):
        if self.L.bt_mg_or_halves(self.base + off, nwords):
            raise RuntimeError("bt_mg_or_halves failed")

    def swap_with(self, other, off, half, chunk):
        """Swap this rank's upper half of the buffer at `off` with the lower half of `other`'s."""
        if self.L.bt_mg_swap(self.base + off + half, self.dev, other.base + off, other.dev, half, chunk):
            raise RuntimeError("bt_mg_swap failed")

    def sync(self):
        if self.L.bt_sync():
            raise RuntimeError("booltn: CUDA error (see stderr)")

    def flag(self):
        buf = np.zeros(1, dtype=np.uint32)
        if self.L.bt_d2h(buf.ctypes.data, self.base + self.prog.flag_off, 4):
            raise RuntimeError("booltn: CUDA error in the plan (see stderr)")
        return bool(buf[0])

    def reset_flag(self):
        self.L.bt_memset(self.base + self.prog.flag_off, 0, 4)

    def free(self):
        for P in self.plans:
            self.L.bt_plan_free(P)
        self.L.bt_free(self.base)


class DistPlan:
    """A Schedule compiled for execution on the GPUs.

    devices: one device id per rank g = 0..2^p-1 (default: rank g on device g mod #devices; ranks may
    share a device, which only makes sense for testing). All slices replay the same plans.

        sch = plan_distribution(tree, ixs_sliced, p=1, cap=36)
        dp = DistPlan(clauses, n, sliced, sch)
        sat, done, seconds = dp.decide()
    """

    def __init__(self, clauses, n, sliced, sch, devices=None, small_rc=14, chunk=64 << 20, use_graph=True):
        from . import engine as E
        self.p, self.P = sch.p, 1 << sch.p
        self.sliced = list(sliced)
        self.sch = sch
        t = time.time()
        segments, leaf_slots = local_segments(sch), sch.leaf_slots
        self.slicer = E.SatSlicer(renamed_clauses(clauses, n, leaf_slots, sch.p),
                                  self.sliced + [n + i for i in range(sch.p)])
        self.prog = Program(segments, self.slicer.labels, small_rc)
        if devices is None:
            nd = device_count()
            devices = [g % nd for g in range(self.P)]
        self.devices = list(devices)
        assert len(self.devices) == self.P
        self.ranks = [_Rank(self.prog, d, use_graph) for d in self.devices]
        self.direct = {}
        d0 = self.ranks[0].device
        for a in sorted(set(self.devices)):
            for b in sorted(set(self.devices)):
                if a < b:
                    self.direct[(a, b)] = d0.call(d0.L.bt_mg_enable_peer, a, b)
        self.chunk = chunk
        self.compile_s = time.time() - t
        self.t_xchg = 0.0
        self.nsegs = len(self.prog.segs)

    def _all(self, fn_of_rank):
        """Run fn_of_rank(rank)() on every rank's device thread and wait."""
        futs = [r.device.submit(fn_of_rank(r)) for r in self.ranks]
        for f in futs:
            f.result()

    def run_slice(self, s):
        """Contract global slice s on all GPUs (results are OR-accumulated in the device flags)."""
        S, P = len(self.sliced), self.P
        futs = [r.device.submit(r.set_leaves, self.slicer.words(s | (g << S))) for g, r in enumerate(self.ranks)]
        for f in futs:
            f.result()
        for si, seg in enumerate(self.prog.segs):
            x = seg["xchg"]
            futs = [r.device.submit(r.run, si, x is not None) for r in self.ranks]
            for f in futs:
                f.result()
            if x is None:
                continue
            t = time.time()
            i, off, half = x["slot"], x["off"], x["half_bytes"]
            # g keeps [top = 0] of both shards, its pair [top = 1] of both
            futs = [self.ranks[g].device.submit(self.ranks[g].swap_with, self.ranks[g | (1 << i)], off, half, self.chunk)
                    for g in range(P) if not (g >> i) & 1]
            for f in futs:
                f.result()
            self.t_xchg += time.time() - t
            if x["fold"]:
                futs = [r.device.submit(r.fold, off, half // 4) for r in self.ranks]
                for f in futs:
                    f.result()

    def sync(self):
        for f in [r.device.submit(r.sync) for r in self.ranks]:
            f.result()

    def flags(self):
        return [f.result() for f in [r.device.submit(r.flag) for r in self.ranks]]

    def reset_flags(self):
        for f in [r.device.submit(r.reset_flag) for r in self.ranks]:
            f.result()

    def decide(self, max_slices=None, early_exit=False):
        """OR over the global slices. Returns (sat, slices_done, seconds)."""
        nsl = 1 << len(self.sliced)
        todo = nsl if max_slices is None else min(nsl, max_slices)
        self.reset_flags()
        self.sync()
        self.t_xchg = 0.0
        t = time.time()
        done = 0
        for s in range(todo):
            self.run_slice(s)
            done += 1
            if early_exit and s % 16 == 15 and any(self.flags()):
                break
        sat = any(self.flags())
        self.sync()
        return sat, done, time.time() - t

    def free(self):
        for f in [r.device.submit(r.free) for r in self.ranks]:
            f.result()
