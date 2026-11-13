"""Exact model counting (#SAT) by tensor-network contraction on the GPU (libcount8.so).

Each pass computes the count modulo a small modulus p; the exact count is rebuilt by CRT over enough moduli.
Width 1: p <= 256, one byte per tensor entry (int8 tensor cores / dp4a); the 54 pairwise coprime moduli
<= 256 reach about 10^109. Width 2: primes 256 < p < 2^16, 16-bit residues in two byte planes (four int8
GEMMs per contraction), for counts beyond the 8-bit range. The compiled plan (CUDA graph) is shared by all
moduli of one width: p lives in device memory and every kernel reads it.

Layout conventions match engine.py: a tensor with labels [l0..l_{r-1}] has 2^r entries, l0 fastest.
"""
import ctypes, os, math, time
import numpy as np
from .engine import Arena, PlanTensor, plan as classify, LIB_DIR, plan_layouts

_LIB = None
IntArr = lambda xs: (ctypes.c_int * max(1, len(xs)))(*xs)
VP = ctypes.c_void_p


def lib():
    global _LIB
    if _LIB is None:
        path = os.path.join(LIB_DIR, "libcount8.so")
        if not os.path.exists(path):
            raise OSError(f"{path} not found: build it with `make -C kernels` in the booltn repository (CUDA 12.8, sm_90a)")
        L = ctypes.CDLL(path)
        L.c8_malloc.restype = VP
        L.c8_malloc.argtypes = [ctypes.c_size_t]
        L.c8_free.argtypes = [VP]
        L.c8_h2d.argtypes = [VP, VP, ctypes.c_size_t, ctypes.c_int]
        L.c8_d2h.argtypes = [VP, VP, ctypes.c_size_t, ctypes.c_int]
        L.c8_memset.argtypes = [VP, ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        L.c8_event_create.restype = VP
        L.c8_event_record.argtypes = [VP, ctypes.c_int]
        L.c8_event_elapsed.restype = ctypes.c_float
        L.c8_event_elapsed.argtypes = [VP, VP]
        L.c8_plan_new.restype = VP
        L.c8_plan_new_w.restype = VP
        L.c8_plan_new_w.argtypes = [ctypes.c_int]
        L.c8_plan_free.argtypes = [VP]
        L.c8_plan_permute.argtypes = [VP, VP, VP, ctypes.c_int, VP, VP, ctypes.c_int, VP]
        L.c8_plan_cc.argtypes = [VP, VP, VP, VP, VP] + [ctypes.c_int] * 4 + [VP]
        L.c8_plan_tc.argtypes = [VP, VP, VP, VP] + [ctypes.c_int] * 4 + [VP]
        L.c8_plan_small_batch.argtypes = [VP, ctypes.c_int, VP] + [VP] * 6 + [VP]
        L.c8_plan_small_batch2.argtypes = [VP, ctypes.c_int, VP] + [VP] * 8 + [VP]
        L.c8_plan_add_accum.argtypes = [VP, VP, VP, VP]
        L.c8_plan_cco.argtypes = [VP, VP, VP, VP] + [ctypes.c_int] * 4 + [VP]
        L.c8_plan_run.argtypes = [VP, ctypes.c_int, ctypes.c_int]
        L.c8_plan_run_range.argtypes = [VP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.c8_make_streams.argtypes = [ctypes.c_int]
        assert L.c8_init() == 0
        _LIB = L
    return _LIB


def primes_desc(limit=251):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [p for p in range(limit, 1, -1) if sieve[p]]


def moduli_desc(limit=256):
    """Pairwise coprime CRT moduli for byte tensors: the largest power <= limit of every prime < limit
    (256, 251, 243, ...). Their product is about 10^109.2, vs 10^100.8 for the primes alone; the kernels'
    Barrett reduction and int32 accumulation bounds hold for any modulus <= 256."""
    out = []
    for p in primes_desc(limit - 1):
        q = p
        while q * p <= limit:
            q *= p
        out.append(q)
    return sorted(out, reverse=True)


def primes16_desc(lo=257, hi=65535):
    """Primes in [lo, hi], largest first: 16-bit moduli (coprime to every 8-bit modulus)."""
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return [int(p) for p in np.nonzero(sieve)[0][::-1] if p >= lo]


def width_of(p):
    return 1 if p <= 256 else 2


def crt(residues, primes):
    x, M = 0, 1
    for r, p in zip(residues, primes):
        # solve x + M*t = r (mod p)
        t = ((r - x) * pow(M, -1, p)) % p
        x += M * t
        M *= p
    return x, M


class CountTree:
    """Compiled counting plan for one contraction tree (omeco dict) with leaf label lists (rank <= 5)."""

    def __init__(self, tree, leaf_labels, small_rc=12, use_graph=True, tc_min_k=5, layout="planned", width=1):
        """layout: "planned" (producers emit their consumer's operand layout where possible, see
        engine.plan_layouts) or "greedy" (per-contraction choice). width: 1 (moduli <= 256) or 2 (moduli
        < 2^16; twice the memory, every tensor as a low-byte and a high-byte plane)."""
        assert layout in ("planned", "greedy") and width in (1, 2)
        self.width = width
        L = lib()
        self.L = L
        self.use_graph = use_graph
        self.stream = -1
        self.records = []
        arena = Arena()
        nleaf = len(leaf_labels)
        assert all(len(l) <= 5 for l in leaf_labels)
        p_off, _ = arena.alloc(4)
        acc_off, _ = arena.alloc(4)
        leaf_off, _ = arena.alloc(32 * width * nleaf)
        self.p_off, self.acc_off, self.leaf_off, self.nleaf = p_off, acc_off, leaf_off, nleaf
        self.leaf_ranks = [len(l) for l in leaf_labels]
        leaves = [PlanTensor(l, leaf_off + 32 * width * i, None) for i, l in enumerate(leaf_labels)]
        ops = []

        def nbytes(r):
            return width << r

        def talloc(labels):
            off, tid = arena.alloc(nbytes(len(labels)))
            return PlanTensor(labels, off, tid)

        def tfree(t):
            if t.tid is not None:
                arena.free(t.tid)
                t.tid = None

        def consumed(src, dst):
            """src was permuted into dst: ops run in order on one stream, so src's bytes may be reused by any
            later op (a tensor of the tree is read by exactly one contraction). Peak memory: 3 large tensors
            (both operands' permuted copies and the output) instead of 5."""
            if dst is not src:
                tfree(src)

        def permute_spec(t, new_labels, red=()):
            if list(new_labels) == t.labels and not red:
                return t
            out = talloc(new_labels)
            pos = {l: i for i, l in enumerate(t.labels)}
            ops.append(("perm", t.off, out.off, out.r, [pos[l] for l in new_labels], [pos[l] for l in red]))
            return out

        # ---- pass 1: small subtrees, batched per level ----
        small, levels = {}, {}

        def is_small(la, lb, lc):
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
            if not is_small(ra[0], rb[0], lc):
                return None
            small[id(d)] = (lc, 1 + max(ra[1], rb[1]))
            return small[id(d)]

        if small_rc > 0:
            mark(tree)

        def choose(Bt, Kt, Mt, Nt):
            swap = len(Mt) > len(Nt)
            if swap:
                Mt, Nt = Nt, Mt
            if len(Kt) >= tc_min_k and len(Nt) >= 5 and len(Bt) <= 30:
                kind = "tc"
            elif len(Kt) <= 4 and len(Nt) >= 4:
                kind = "cco"
            else:
                kind = "cc"
            return swap, kind, kind == "cco"

        info, orders, smallwant = {}, {}, {}
        if layout == "planned":
            info, orders, smallwant = plan_layouts(
                tree,
                lambda d: leaves[d["tensor_index"]].labels if "tensor_index" in d else
                (small[id(d)][0] if id(d) in small else d["eins"]["iy"]),
                lambda d: id(d) in small, choose)

        def tensor_of(d):
            return leaves[d["tensor_index"]] if "tensor_index" in d else small[id(d)]

        def build_small(d):
            if "tensor_index" in d or not isinstance(small.get(id(d)), tuple):
                return
            build_small(d["args"][0]); build_small(d["args"][1])
            lc, lev = small[id(d)]
            lc = smallwant.get(id(d), lc)       # written directly in the large consumer's operand layout
            A, B = tensor_of(d["args"][0]), tensor_of(d["args"][1])
            off, _ = arena.alloc(nbytes(len(lc)))
            pa = {l: i for i, l in enumerate(A.labels)}
            pb = {l: i for i, l in enumerate(B.labels)}
            summed = [l for l in dict.fromkeys(A.labels + B.labels) if l not in set(lc)]
            levels.setdefault(lev, []).append((A.off, B.off, off, len(lc), len(summed), len(A.labels), len(B.labels),
                                               [pa.get(l, -1) for l in lc], [pb.get(l, -1) for l in lc],
                                               [pa.get(l, -1) for l in summed], [pb.get(l, -1) for l in summed]))
            small[id(d)] = PlanTensor(lc, off, None)

        def walk_small(d):
            if "tensor_index" in d:
                return
            if id(d) in small:
                build_small(d); return
            walk_small(d["args"][0]); walk_small(d["args"][1])

        walk_small(tree)
        self.small_levels = levels

        # ---- pass 2: large contractions ----
        def contract_spec(A, B, lc):
            Bt, Kt, Mt, Nt, Ra, Rb = classify(A.labels, B.labels, lc)
            n0 = len(ops)
            tmp = []
            if Ra:
                A0, A = A, permute_spec(A, [l for l in A.labels if l not in Ra], Ra); tmp.append(A); consumed(A0, A)
            if Rb:
                B0, B = B, permute_spec(B, [l for l in B.labels if l not in Rb], Rb); tmp.append(B); consumed(B0, B)
            if len(Mt) > len(Nt):
                A, B, Mt, Nt = B, A, Nt, Mt
            # shared groups ordered by the larger operand (cheaper permutation of the big one)
            big = A.labels if len(A.labels) >= len(B.labels) else B.labels
            pos = {l: i for i, l in enumerate(big)}
            Bt, Kt = sorted(Bt, key=pos.get), sorted(Kt, key=pos.get)
            if len(Kt) >= tc_min_k and len(Nt) >= 5 and len(Bt) <= 30:
                kind = "tc"
            elif len(Kt) <= 4 and len(Nt) >= 4:
                kind = "cco"   # small K: outer-product CUDA-core kernel, B stored n-fastest
            else:
                kind = "cc"
            Ao, Bo, Co = Kt + Mt + Bt, (Nt + Kt + Bt) if kind == "cco" else (Kt + Nt + Bt), Nt + Mt + Bt
            A2 = permute_spec(A, Ao); consumed(A, A2)
            B2 = permute_spec(B, Bo); consumed(B, B2)
            C = talloc(Co)
            if kind == "tc":
                ops.append(("tc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            elif kind == "cco":
                ops.append(("cco", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            else:
                nout = 1 << (len(Bt) + len(Mt) + len(Nt))
                scratch = None
                if nout < (1 << 17) and len(Kt) > 8:
                    scratch = talloc(["scratch"] * (len(Bt) + len(Mt) + len(Nt) + 2))  # 4 bytes per element
                ops.append(("cc", A2.off, B2.off, C.off, scratch.off if scratch else None,
                            len(Bt), len(Mt), len(Nt), len(Kt)))
                if scratch:
                    tfree(scratch)
            for t in (A2, B2):
                if t is not A and t is not B:
                    tfree(t)
            for t in tmp:
                if t is not A2 and t is not B2:
                    tfree(t)
            self.records.append((kind, len(Bt), len(Mt), len(Nt), len(Kt), n0, len(ops)))
            return C

        def contract_planned(d, a, b):
            I, O = info[id(d)], orders[id(d)]
            A, B = (b, a) if I["swap"] else (a, b)
            kind = I["kind"]
            Kt, Mt, Nt, Bt = O["K"], O["M"], O["N"], O["B"]
            n0 = len(ops)
            Ao, Bo, Co = Kt + Mt + Bt, (Nt + Kt + Bt) if kind == "cco" else (Kt + Nt + Bt), Nt + Mt + Bt
            A2 = permute_spec(A, Ao, I["Ra"]); consumed(A, A2)
            B2 = permute_spec(B, Bo, I["Rb"]); consumed(B, B2)
            C = talloc(Co)
            if kind in ("tc", "cco"):
                ops.append((kind, A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            else:
                nout = 1 << (len(Bt) + len(Mt) + len(Nt))
                scratch = None
                if nout < (1 << 17) and len(Kt) > 8:
                    scratch = talloc(["scratch"] * (len(Bt) + len(Mt) + len(Nt) + 2))  # 4 bytes per element
                ops.append(("cc", A2.off, B2.off, C.off, scratch.off if scratch else None,
                            len(Bt), len(Mt), len(Nt), len(Kt)))
                if scratch:
                    tfree(scratch)
            for t in (A2, B2):
                if t is not A and t is not B:
                    tfree(t)
            self.records.append((kind, len(Bt), len(Mt), len(Nt), len(Kt), n0, len(ops)))
            return C

        def rec(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]]
            if id(d) in small:
                return small[id(d)]
            a, b = rec(d["args"][0]), rec(d["args"][1])
            c = contract_planned(d, a, b) if id(d) in orders else contract_spec(a, b, d["eins"]["iy"])
            tfree(a); tfree(b)
            return c

        root = rec(tree)
        if root.r > 0:      # a lone leaf (a one-clause network): sum out its labels
            root = permute_spec(root, [], root.labels)
        assert root.r == 0, f"root must be a scalar, got rank {root.r}"
        self.arena_bytes = arena.top
        self.base = L.c8_malloc(self.arena_bytes)
        if not self.base:
            raise MemoryError(f"arena {self.arena_bytes/2**30:.2f} GiB")
        B = self.base
        pp = B + p_off
        self.plan = L.c8_plan_new_w(width)
        nb = 0
        for lev in sorted(levels):
            lst = levels[lev]; n = len(lst)
            ptrs = (VP * (3 * n))()
            rc = (ctypes.c_int * n)(); nk = (ctypes.c_int * n)(); ra = (ctypes.c_int * n)(); rb = (ctypes.c_int * n)()
            apos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n)); bpos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n))
            ka = (ctypes.c_int * (8 * n))(*([-1] * 8 * n)); kb = (ctypes.c_int * (8 * n))(*([-1] * 8 * n))
            for i, (ao, bo, co, rci, nki, rai, rbi, ap, bp, kai, kbi) in enumerate(lst):
                ptrs[3 * i], ptrs[3 * i + 1], ptrs[3 * i + 2] = B + ao, B + bo, B + co
                rc[i], nk[i], ra[i], rb[i] = rci, nki, rai, rbi
                for q, v in enumerate(ap): apos[24 * i + q] = v
                for q, v in enumerate(bp): bpos[24 * i + q] = v
                for q, v in enumerate(kai): ka[8 * i + q] = v
                for q, v in enumerate(kbi): kb[8 * i + q] = v
            r = L.c8_plan_small_batch2(self.plan, n, ptrs, rc, nk, apos, bpos, ka, kb, ra, rb, pp)
            if r:
                raise RuntimeError(f"small batch failed {r}")
            nb += 1
        self.records = [(k, b, m, n_, kk, i0 + nb, i1 + nb) for (k, b, m, n_, kk, i0, i1) in self.records]
        self.records.insert(0, ("small", 0, 0, 0, 0, 0, nb))
        for op in ops:
            if op[0] == "perm":
                _, i, o, rout, src, red = op
                r = L.c8_plan_permute(self.plan, B + i, B + o, rout, IntArr(src), IntArr(red), len(red), pp)
            elif op[0] == "tc":
                _, a, b_, c, bb, mm, nn, kk = op
                r = L.c8_plan_tc(self.plan, B + a, B + b_, B + c, bb, mm, nn, kk, pp)
            elif op[0] == "cco":
                _, a, b_, c, bb, mm, nn, kk = op
                r = L.c8_plan_cco(self.plan, B + a, B + b_, B + c, bb, mm, nn, kk, pp)
            else:
                _, a, b_, c, s, bb, mm, nn, kk = op
                r = L.c8_plan_cc(self.plan, B + a, B + b_, B + c, (B + s) if s is not None else None, bb, mm, nn, kk, pp)
            if r:
                raise RuntimeError(f"plan op {op[0]} failed rc={r}")
        L.c8_plan_add_accum(self.plan, B + acc_off, B + root.off, pp)
        self.nops = len(ops) + nb + 1

    def set_prime(self, p):
        assert width_of(p) <= self.width and p < 65536, f"modulus {p} needs width {width_of(p)}"
        v = np.array([p], dtype=np.int32)
        self.L.c8_h2d(self.base + self.p_off, v.ctypes.data, 4, self.stream)
        self.L.c8_memset(self.base + self.acc_off, 0, 4, self.stream)

    def set_leaves(self, buf):
        """buf: uint8 array of 32*nleaf bytes (leaf i at [32i, 32i + 2^rank_i)); 0/1 entries, so with width 2
        the high-byte plane [2^rank_i, 2^(rank_i+1)) of each 64-byte slot is zero."""
        buf = np.ascontiguousarray(buf, dtype=np.uint8)
        if self.width == 2:
            buf = np.concatenate([buf.reshape(-1, 32), np.zeros((buf.size // 32, 32), dtype=np.uint8)], axis=1)
        self._buf = np.ascontiguousarray(buf).reshape(-1)
        self.L.c8_h2d(self.base + self.leaf_off, self._buf.ctypes.data, self._buf.nbytes, self.stream)

    def run(self):
        r = self.L.c8_plan_run(self.plan, self.stream, 1 if self.use_graph else 0)
        if r:
            raise RuntimeError(f"plan run failed {r}")

    def residue(self):
        v = np.zeros(1, dtype=np.int32)
        # the copy synchronizes the stream: an asynchronous kernel failure surfaces here (never read it as 0)
        if self.L.c8_d2h(v.ctypes.data, self.base + self.acc_off, 4, self.stream):
            raise RuntimeError("count8: CUDA error in the plan (see stderr)")
        return int(v[0])

    def free(self):
        self.L.c8_plan_free(self.plan)
        self.L.c8_free(self.base)


class CountSlicer:
    """Clause tensors (0/1 bytes) for each assignment of the sliced variables, vectorized over clauses."""

    def __init__(self, clauses, sliced_vars):
        sl = {v: i for i, v in enumerate(sliced_vars)}
        m = len(clauses)
        self.var_mask = np.zeros(m, dtype=np.uint64)
        self.pos_mask = np.zeros(m, dtype=np.uint64)
        self.labels, self.fidx, self.rank = [], np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
        for ci, c in enumerate(clauses):
            labels, idx, q = [], 0, 0
            for l in c:
                v = abs(l) - 1
                if v in sl:
                    self.var_mask[ci] |= np.uint64(1 << sl[v])
                    if l > 0:
                        self.pos_mask[ci] |= np.uint64(1 << sl[v])
                else:
                    if v in labels:
                        raise ValueError("repeated variable in clause")
                    labels.append(v)
                    if l < 0:
                        idx |= 1 << q
                    q += 1
            self.labels.append(labels)
            self.fidx[ci], self.rank[ci] = idx, len(labels)
        self.base = np.zeros((m, 32), dtype=np.uint8)
        for ci in range(m):
            self.base[ci, : 1 << self.rank[ci]] = 1
        self.rows = np.arange(m)

    def leaves(self, s):
        s = np.uint64(s)
        satisfied = ((~(s ^ self.pos_mask)) & self.var_mask) != 0
        buf = self.base.copy()
        unsat = ~satisfied
        buf[self.rows[unsat], self.fidx[unsat]] = 0
        return buf.reshape(-1)


def count_models(clauses, n, tree, sliced=(), primes=None, small_rc=12, verbose=False, mode="stable", confirm=3,
                 wide=True):
    """Exact model count of a CNF over variables 1..n using a contraction tree (dict) for the einsum
    built from the clause variable lists with `sliced` variables removed. Returns (count, info).

    Moduli: the 54 8-bit moduli (one byte per entry), then, if wide, 16-bit primes (width-2 plan, compiled
    only when the 8-bit range of ~10^109 does not suffice; twice the memory, about 2-4x the time per pass).
    mode="rigorous": use moduli until their product exceeds 2^(#used vars) (always exact).
    mode="stable"  : early-terminating CRT: stop when the reconstruction has not changed for `confirm`
                     extra moduli (error probability ~ prod 1/p_i over the confirming moduli, <= ~1e-7 for 3).
    """
    from .tree_utils import recompute_outputs
    sliced = list(sliced)
    used = sorted({abs(l) - 1 for c in clauses for l in c})
    free_vars = n - len(used)
    slicer = CountSlicer(clauses, sliced)
    ixs = [list(l) for l in slicer.labels]
    d = recompute_outputs(tree, ixs, [])
    need_bits = len(used) + 1           # rigorous bound: count <= 2^(n_used) (free vars are a power of 2)
    if primes is None:
        allp = moduli_desc() + (primes16_desc() if wide else [])
        if mode == "rigorous":
            primes, bits = [], 0.0
            for p in allp:
                if bits > need_bits:
                    break
                primes.append(p); bits += math.log2(p)
            if bits <= need_bits:
                raise ValueError(f"need > {need_bits} bits but the moduli give only {bits:.0f}")
        else:
            primes = allp
    def build(w):
        """Plan of width w with its slicer. A width-2 plan (twice the memory) that does not fit is sliced over
        1..4 more labels (greedy_slice): the count is the same, so the 8-bit residues stay valid."""
        from .slicer import greedy_slice
        from .tree_utils import complexity
        last_err = None
        for k in range(0, 5 if w == 2 else 1):
            sl, dd, extra = slicer, d, []
            if k:
                _, sc = complexity(d, ixs)
                extra, _ = greedy_slice(d, ixs, sc - k, kind="tc")
                sl = CountSlicer(clauses, sliced + extra)
                dd = recompute_outputs(tree, [list(l) for l in sl.labels], [])
            try:
                return CountTree(dd, sl.labels, small_rc=small_rc, width=w), sl, len(extra)
            except MemoryError as e:
                last_err = e
        if w == 1:
            raise last_err
        raise ValueError(f"16-bit plan does not fit even with 4 more sliced labels ({last_err})")

    residues, used_primes = [], []
    last, stable = None, 0
    ct, t_compile, t_run, arena, widths, extra_sliced = None, 0.0, 0.0, 0.0, {}, 0
    run_w, compile_w = {}, {}
    try:
        for p in primes:
            w = width_of(p)
            if ct is None or ct.width < w:
                if ct is not None:
                    ct.free(); ct = None
                t0 = time.time()
                ct, slicer_w, extra_sliced = build(w)
                nsl = 1 << (len(sliced) + extra_sliced)
                leaf_bufs = [slicer_w.leaves(s) for s in range(nsl)] if nsl <= 4096 else None
                t_compile += time.time() - t0
                compile_w[w] = time.time() - t0
                arena = max(arena, ct.arena_bytes)
            t1 = time.time()
            ct.set_prime(p)
            try:
                for s in range(nsl):
                    ct.set_leaves(leaf_bufs[s] if leaf_bufs is not None else slicer_w.leaves(s))
                    ct.run()
                r = ct.residue()
            except RuntimeError as e:   # a CUDA failure in the plan (e.g. out of memory): treat as "does not fit"
                raise MemoryError(f"GPU plan (width {ct.width}) failed: {e}") from e
            residues.append(r); used_primes.append(p)
            t_run += time.time() - t1
            run_w[w] = run_w.get(w, 0.0) + time.time() - t1
            widths[w] = widths.get(w, 0) + 1
            if mode != "rigorous":
                val, _ = crt(residues, used_primes)
                stable = stable + 1 if val == last else 0
                last = val
                if stable >= confirm:
                    break
        else:
            if mode != "rigorous":
                raise ValueError("moduli exhausted before the CRT reconstruction stabilized"
                                 + ("" if wide else " (8-bit only: count > ~10^109)"))
    finally:
        if ct is not None:
            ct.free()
    count, M = crt(residues, used_primes)
    count <<= free_vars
    info = dict(primes=len(used_primes), primes16=widths.get(2, 0), slices=1 << len(sliced),
                extra_sliced16=extra_sliced if widths.get(2) else 0, compile_s=t_compile, run_s=t_run,
                arena_gib=arena / 2 ** 30, per_pass_s=t_run / len(used_primes), run8_s=run_w.get(1, 0.0),
                run16_s=run_w.get(2, 0.0), compile16_s=compile_w.get(2, 0.0))
    return count, info


def profile_tree(clauses, n, tree, sliced=(), prime=251, small_rc=12, width=1):
    """Per-contraction GPU timing of one pass (one slice): returns list of (kind,B,M,N,K,ms)."""
    from .tree_utils import recompute_outputs
    slicer = CountSlicer(clauses, list(sliced))
    d = recompute_outputs(tree, [list(l) for l in slicer.labels], [])
    ct = CountTree(d, slicer.labels, small_rc=small_rc, use_graph=False, width=width)
    ct.set_prime(prime); ct.set_leaves(slicer.leaves(0)); ct.run(); ct.L.c8_sync_all()
    L = ct.L
    e0, e1 = L.c8_event_create(), L.c8_event_create()
    out = []
    for kind, b, m, nn, k, i0, i1 in ct.records:
        L.c8_event_record(e0, -1)
        L.c8_plan_run_range(ct.plan, i0, i1, -1)
        L.c8_event_record(e1, -1)
        out.append((kind, b, m, nn, k, L.c8_event_elapsed(e0, e1)))
    ct.free()
    return out
