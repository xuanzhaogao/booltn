"""Boolean tensor-network contraction engine on the GPU (drives libbooltn.so via ctypes).

Tensors are bit-packed: a tensor with labels [l0, l1, ..., l_{r-1}] has 2^r bits; label l_p is
bit p of the linear index (l0 fastest). Every contraction is planned as
    permute(A), permute(B) -> batched bool GEMM (tensor-core or CUDA-core) -> C
"""
import ctypes, os, time
import numpy as np

_LIB = None
_HERE = os.path.dirname(os.path.abspath(__file__))
# built by `make -C booltn/kernels` into booltn/booltn/lib/; override with $BOOLTN_LIB_DIR
LIB_DIR = os.environ.get("BOOLTN_LIB_DIR", os.path.join(_HERE, "lib"))
LIBPATH = os.path.join(LIB_DIR, "libbooltn.so")

IntArr = lambda xs: (ctypes.c_int * max(1, len(xs)))(*xs)


def lib():
    global _LIB
    if _LIB is None:
        if not os.path.exists(LIBPATH):
            raise OSError(f"{LIBPATH} not found: build it with `make -C booltn/kernels` (CUDA 12.8, sm_90a)")
        L = ctypes.CDLL(LIBPATH)
        L.bt_malloc.restype = ctypes.c_void_p
        L.bt_malloc.argtypes = [ctypes.c_size_t]
        L.bt_free.argtypes = [ctypes.c_void_p]
        L.bt_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        L.bt_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        L.bt_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        L.bt_event_create.restype = ctypes.c_void_p
        L.bt_event_record.argtypes = [ctypes.c_void_p]
        L.bt_event_elapsed.restype = ctypes.c_float
        L.bt_event_elapsed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.bt_mem_used.restype = ctypes.c_size_t
        L.bt_permute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.bt_gemm_cc.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
        L.bt_gemm_tc.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5
        L.bt_contract_generic.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        L.bt_set_perm_impl.argtypes = [ctypes.c_int]
        L.bt_pad_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        assert L.bt_init() == 0
        _LIB = L
    return _LIB


def nwords(r):
    return 1 if r < 5 else 1 << (r - 5)


class BTensor:
    """Bit-packed boolean tensor on the GPU."""

    def __init__(self, labels, ptr=None):
        self.labels = list(labels)
        self.r = len(self.labels)
        self.ptr = ptr if ptr is not None else lib().bt_malloc(4 * nwords(self.r))
        if not self.ptr:
            raise MemoryError(f"bt_malloc failed for rank {self.r}")

    def free(self):
        if self.ptr:
            lib().bt_free(self.ptr)
            self.ptr = None

    @staticmethod
    def from_bool(arr, labels):
        """arr: numpy bool array with arr.ndim == len(labels); axis i <-> labels[i]."""
        r = len(labels)
        # bit position p <-> labels[p]; C-order flatten makes the LAST axis fastest, so reverse axes
        flat = np.ascontiguousarray(np.transpose(arr, axes=list(range(r))[::-1])).reshape(-1).astype(np.uint8)
        packed = np.packbits(flat, bitorder="little")
        buf = np.zeros(4 * nwords(r), dtype=np.uint8)
        buf[: len(packed)] = packed
        t = BTensor(labels)
        lib().bt_h2d(t.ptr, buf.ctypes.data, buf.nbytes)
        return t

    @staticmethod
    def from_words(words, labels):
        t = BTensor(labels)
        w = np.ascontiguousarray(words, dtype=np.uint32)
        lib().bt_h2d(t.ptr, w.ctypes.data, w.nbytes)
        return t

    def to_bool(self):
        """Return numpy bool array with axis i <-> self.labels[i]."""
        buf = np.zeros(nwords(self.r), dtype=np.uint32)
        lib().bt_d2h(buf.ctypes.data, self.ptr, buf.nbytes)
        bits = np.unpackbits(buf.view(np.uint8), bitorder="little")[: 1 << self.r].astype(bool)
        return np.transpose(bits.reshape((2,) * self.r), axes=list(range(self.r))[::-1]) if self.r else bits.reshape(())


def permute(t, new_labels, reduce_labels=(), free_input=False):
    """Return a tensor with label order new_labels; labels in reduce_labels are OR-reduced."""
    if list(new_labels) == t.labels and not reduce_labels:
        return t
    pos = {l: i for i, l in enumerate(t.labels)}
    out = BTensor(new_labels)
    src = [pos[l] for l in new_labels]
    red = [pos[l] for l in reduce_labels]
    rc = lib().bt_permute(t.ptr, t.r, out.ptr, out.r, IntArr(src), IntArr(red), len(red))
    if rc < 0:
        raise RuntimeError(f"bt_permute failed {rc}")
    if free_input:
        t.free()
    return out


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.records = []  # (kind, B, M, N, K, ms_perm, ms_gemm)


STATS = Stats()

TC_MIN_K = 7  # min contracted labels to route to tensor cores (rows must be >= 128 bits)


def plan(La, Lb, Lc):
    sb, sc, sa = set(Lb), set(Lc), set(La)
    Bt = [l for l in La if l in sb and l in sc]
    Kt = [l for l in La if l in sb and l not in sc]
    Mt = [l for l in La if l not in sb and l in sc]
    Nt = [l for l in Lb if l not in sa and l in sc]
    Ra = [l for l in La if l not in sb and l not in sc]
    Rb = [l for l in Lb if l not in sa and l not in sc]
    return Bt, Kt, Mt, Nt, Ra, Rb


# fitted kernel cost model (see py/fit_cost2.py, results/bench_kernels_model2.json)
def kernel_costs(b, m, n, k):
    """Modeled seconds per kernel option for a contraction with label counts (B, M, N, K), n >= m."""
    a_bits, b_bits, c_bits = 2.0 ** (b + m + k), 2.0 ** (b + n + k), 2.0 ** (b + m + n)
    moved = a_bits + b_bits + c_bits
    costs = {"cc": 6.25e-14 * moved + 1.97e-14 * 2.0 ** (b + m + n + k)}
    if n >= 5 and b <= 30:
        pad_cells = 2.0 ** (b + max(m, 6) + max(n, 6) + max(k, 8))
        if k >= 7:
            costs["tc"] = 2.78e-14 * moved + 7.6e-17 * pad_cells + 2.14e-13 * c_bits
        elif k >= 3:
            pad_bits = 128.0 * (2.0 ** (b + m) + 2.0 ** (b + n))
            costs["tcp"] = (2.78e-14 * (pad_bits + c_bits) + 7.6e-17 * pad_cells + 2.14e-13 * c_bits
                            + 2 * pad_bits / 8e12 + 8e-6)
    return costs


def choose_kind(Bt, Kt, Mt, Nt):
    """Pick kernel and operand roles. Returns (swap A<->B, kind in {tc, tcp, cc}).

    tc : 1-bit tensor cores (K >= 7 labels, N >= 5 labels)
    tcp: tensor cores with operand rows zero-padded to 128 bits (3 <= K <= 6 labels)
    cc : CUDA-core masked-OR (any shape; N < 5 packs several rows per output word)
    Choice = argmin of the fitted cost model.
    """
    m, n = len(Mt), len(Nt)
    swap = m > n  # put the larger free index set on the N (word-packed) side
    if swap:
        m, n = n, m
    costs = kernel_costs(len(Bt), m, n, len(Kt))
    kind = min(costs, key=costs.get)
    return swap, kind


def order_shared(A_labels, B_labels, Bt, Kt):
    """Order the shared label groups (batch, contracted) by their positions in the LARGER operand, so the
    larger operand's permutation is cheaper (often an identity or a block copy)."""
    big = A_labels if len(A_labels) >= len(B_labels) else B_labels
    pos = {l: i for i, l in enumerate(big)}
    return sorted(Bt, key=pos.get), sorted(Kt, key=pos.get)


def contract(A, B, out_labels, timing=False, free_inputs=False, force=None):
    """Bool contraction C[out] = OR over summed labels of A & B. Returns C (label order chosen by planner)."""
    L = lib()
    Bt, Kt, Mt, Nt, Ra, Rb = plan(A.labels, B.labels, out_labels)
    tmp = []
    if timing:
        e0, e1, e2 = L.bt_event_create(), L.bt_event_create(), L.bt_event_create()
        L.bt_event_record(e0)
    if Ra:
        A = permute(A, [l for l in A.labels if l not in Ra], Ra)
        tmp.append(A)
    if Rb:
        B = permute(B, [l for l in B.labels if l not in Rb], Rb)
        tmp.append(B)
    swap, kind0 = choose_kind(Bt, Kt, Mt, Nt)
    if swap:
        A, B, Mt, Nt = B, A, Nt, Mt
    kind = force or kind0
    Bt, Kt = order_shared(A.labels, B.labels, Bt, Kt)
    if kind in ("tc", "tcp"):
        Ao, Bo, Co = Kt + Mt + Bt, Kt + Nt + Bt, Nt + Mt + Bt
    elif kind == "cc":
        Ao, Bo, Co = Kt + Mt + Bt, Nt + Kt + Bt, Nt + Mt + Bt
    else:
        Ao, Bo, Co = A.labels, B.labels, Nt + Mt + Bt
    A2 = permute(A, Ao)
    B2 = permute(B, Bo)
    if kind == "tcp":
        kk = len(Kt)
        dummy = [("pad", i) for i in range(7 - kk)]
        A3 = BTensor(Kt + dummy + Mt + Bt); B3 = BTensor(Kt + dummy + Nt + Bt)
        L.bt_pad_rows(A2.ptr, A3.ptr, len(Mt) + len(Bt), kk)
        L.bt_pad_rows(B2.ptr, B3.ptr, len(Nt) + len(Bt), kk)
        for t in (A2, B2):
            if t is not A and t is not B:
                t.free()
        A2, B2 = A3, B3
    if timing:
        L.bt_event_record(e1)
    C = BTensor(Co)
    if kind == "tc":
        rc = L.bt_gemm_tc(A2.ptr, B2.ptr, C.ptr, len(Bt), len(Mt), len(Nt), len(Kt), -1)
    elif kind == "tcp":
        rc = L.bt_gemm_tc(A2.ptr, B2.ptr, C.ptr, len(Bt), len(Mt), len(Nt), 7, -1)
    elif kind == "cc":
        rc = L.bt_gemm_cc(A2.ptr, B2.ptr, C.ptr, len(Bt), len(Mt), len(Nt), len(Kt))
    else:
        pa = {l: i for i, l in enumerate(A2.labels)}
        pb = {l: i for i, l in enumerate(B2.labels)}
        a_pos = [pa.get(l, -1) for l in Co]
        b_pos = [pb.get(l, -1) for l in Co]
        rc = L.bt_contract_generic(A2.ptr, B2.ptr, C.ptr, len(Co), IntArr(a_pos), IntArr(b_pos), len(Kt),
                                   IntArr([pa[l] for l in Kt]), IntArr([pb[l] for l in Kt]))
    if rc < 0:
        raise RuntimeError(f"gemm kind={kind} failed rc={rc} B={len(Bt)} M={len(Mt)} N={len(Nt)} K={len(Kt)}")
    if timing:
        L.bt_event_record(e2)
        STATS.records.append((kind, len(Bt), len(Mt), len(Nt), len(Kt),
                              L.bt_event_elapsed(e0, e1), L.bt_event_elapsed(e1, e2)))
    for t in (A2, B2):
        if t is not A and t is not B:
            t.free()
    for t in tmp:
        t.free()
    if free_inputs:
        A.free()
        B.free()
    return C


# ------------------------------------------------------------------------------------------
# SAT tensor networks
# ------------------------------------------------------------------------------------------
def clause_tensor_words(clause, fixed=None):
    """Bits of a clause tensor over its (unfixed) variables. fixed: dict var->0/1 (sliced variables).

    Returns (labels, words) or (None, None) if the clause is satisfied by the fixed values (tensor = all ones;
    the caller can drop it only if all its labels are... we keep it as all-ones to preserve the tree).
    """
    fixed = fixed or {}
    labels = [abs(l) - 1 for l in clause if (abs(l) - 1) not in fixed]
    sat_by_fixed = any(((abs(l) - 1) in fixed) and (fixed[abs(l) - 1] == (1 if l > 0 else 0)) for l in clause)
    r = len(labels)
    n = 1 << r
    bits = np.ones(n, dtype=bool)
    if not sat_by_fixed:
        idx = 0
        p = 0
        for l in clause:
            v = abs(l) - 1
            if v in fixed:
                continue
            if l < 0:
                idx |= 1 << p
            p += 1
        bits[idx] = False
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    buf = np.zeros(4 * nwords(r), dtype=np.uint8)
    buf[: len(packed)] = packed
    return labels, buf.view(np.uint32)


def contract_tree(tree, leaves, timing=False):
    """tree: omeco dict; leaves: list of BTensor (consumed). Returns the root BTensor."""

    def rec(d):
        if "tensor_index" in d:
            return leaves[d["tensor_index"]]
        a = rec(d["args"][0])
        b = rec(d["args"][1])
        return contract(a, b, d["eins"]["iy"], timing=timing, free_inputs=True)

    return rec(tree)


def solve_sat(clauses, n, tree, sliced_vars=(), timing=False, early_exit=True, verbose=False):
    """Eager reference path (one Python-driven contraction at a time, slow; for debugging).
    Use ``decide`` for the compiled fast path. Returns (sat, slices_done, seconds)."""
    from .tree_utils import recompute_outputs
    sliced_vars = list(sliced_vars)
    ixs_sl = [[abs(l) - 1 for l in c if (abs(l) - 1) not in set(sliced_vars)] for c in clauses]
    tr = recompute_outputs(tree, ixs_sl, [])
    t0 = time.time()
    result = False
    nslices = 1 << len(sliced_vars)
    done = 0
    for s in range(nslices):
        fixed = {v: (s >> i) & 1 for i, v in enumerate(sliced_vars)}
        leaves = []
        for c in clauses:
            labels, words = clause_tensor_words(c, fixed)
            leaves.append(BTensor.from_words(words, labels))
        root = contract_tree(tr, leaves, timing=timing)
        val = bool(root.to_bool().reshape(-1)[0])
        root.free()
        done += 1
        result = result or val
        if verbose:
            print(f"  slice {s}/{nslices}: {val}  ({time.time()-t0:.2f}s)")
        if result and early_exit:
            break
    lib().bt_sync()
    return result, done, time.time() - t0


# ------------------------------------------------------------------------------------------
# Compiled plans: whole-tree contraction recorded once (fixed arena addresses), replayed per slice
# ------------------------------------------------------------------------------------------
def _setup_plan_api(L):
    if getattr(L, "_plan_ok", False):
        return
    L.bt_plan_new.restype = ctypes.c_void_p
    L.bt_plan_free.argtypes = [ctypes.c_void_p]
    L.bt_plan_size.argtypes = [ctypes.c_void_p]
    L.bt_plan_permute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    L.bt_plan_cc.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    L.bt_plan_tc.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
    L.bt_plan_generic.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                          ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    L.bt_plan_run.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.bt_plan_run_range.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    L.bt_plan_or_accum.argtypes = [ctypes.c_void_p] * 3
    L.bt_plan_pad_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    L.bt_make_streams.argtypes = [ctypes.c_int]
    L.bt_h2d_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    L.bt_d2h_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    L.bt_plan_run_stream.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.bt_plan_small_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p] + [ctypes.c_void_p] * 6
    L._plan_ok = True


class Arena:
    """Offset allocator over one device buffer; tensors are assigned [offset, offset+bytes) intervals."""

    def __init__(self):
        self.live = []  # (offset, size, id)
        self.top = 0
        self.next_id = 0

    def alloc(self, nbytes):
        nbytes = (nbytes + 255) // 256 * 256
        self.live.sort()
        cur = 0
        for off, sz, _ in self.live:
            if off - cur >= nbytes:
                break
            cur = max(cur, off + sz)
        tid = self.next_id
        self.next_id += 1
        self.live.append((cur, nbytes, tid))
        self.top = max(self.top, cur + nbytes)
        return cur, tid

    def free(self, tid):
        self.live = [x for x in self.live if x[2] != tid]


class PlanTensor:
    def __init__(self, labels, off, tid):
        self.labels = list(labels)
        self.r = len(self.labels)
        self.off = off
        self.tid = tid


class CompiledTree:
    """Records the whole contraction of `tree` (omeco dict) with leaves of given label lists.

    Usage: ct = CompiledTree(tree, leaf_labels); ct.set_leaves(words_per_leaf (uint32 array, 1 word per
    leaf if all leaves have rank <= 5)); ct.run(); ct.result() -> bool
    """

    def __init__(self, tree, leaf_labels, use_graph=True, small_rc=14):
        L = lib()
        _setup_plan_api(L)
        self.L = L
        self.use_graph = use_graph
        self.records = []  # per op: (kind, B, M, N, K, op_index_start, op_index_end)
        arena = Arena()
        # leaves: one contiguous region; leaf i occupies nwords(rank_i) words (rank <= 16)
        assert all(len(l) <= 16 for l in leaf_labels), "leaf tensors must have rank <= 16"
        nleaf = len(leaf_labels)
        leaf_words = [nwords(len(l)) for l in leaf_labels]
        self.leaf_word_off = np.concatenate([[0], np.cumsum(leaf_words)[:-1]]).astype(np.int64)
        self.leaf_total_words = int(sum(leaf_words))
        leaf_off, _ = arena.alloc(4 * self.leaf_total_words)
        # result flag lives for the whole plan (allocated first, never freed)
        flag_off, _ = arena.alloc(4)
        self.flag_off = flag_off
        leaves = [PlanTensor(l, leaf_off + 4 * int(self.leaf_word_off[i]), None) for i, l in enumerate(leaf_labels)]
        self.nleaf = nleaf
        self.leaf_off = leaf_off
        ops = []  # deferred op specs (need arena base pointer)

        def talloc(labels):
            off, tid = arena.alloc(4 * nwords(len(labels)))
            return PlanTensor(labels, off, tid)

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

        def contract_spec(A, B, out_labels):
            Bt, Kt, Mt, Nt, Ra, Rb = plan(A.labels, B.labels, out_labels)
            nops0 = len(ops)
            tmp = []
            if Ra:
                A2 = permute_spec(A, [l for l in A.labels if l not in Ra], Ra); tmp.append(A2); A = A2
            if Rb:
                B2 = permute_spec(B, [l for l in B.labels if l not in Rb], Rb); tmp.append(B2); B = B2
            swap, kind = choose_kind(Bt, Kt, Mt, Nt)
            if swap:
                A, B, Mt, Nt = B, A, Nt, Mt
            Bt, Kt = order_shared(A.labels, B.labels, Bt, Kt)
            if kind in ("tc", "tcp"):
                Ao, Bo, Co = Kt + Mt + Bt, Kt + Nt + Bt, Nt + Mt + Bt
            elif kind == "cc":
                Ao, Bo, Co = Kt + Mt + Bt, Nt + Kt + Bt, Nt + Mt + Bt
            else:
                Ao, Bo, Co = A.labels, B.labels, Nt + Mt + Bt
            A2 = permute_spec(A, Ao)
            B2 = permute_spec(B, Bo)
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
            if kind == "tc":
                ops.append(("tc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            elif kind == "tcp":
                ops.append(("tc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), 7))
            elif kind == "cc":
                ops.append(("cc", A2.off, B2.off, C.off, len(Bt), len(Mt), len(Nt), len(Kt)))
            else:
                pa = {l: i for i, l in enumerate(A2.labels)}
                pb = {l: i for i, l in enumerate(B2.labels)}
                ops.append(("gen", A2.off, B2.off, C.off, len(Co), [pa.get(l, -1) for l in Co],
                            [pb.get(l, -1) for l in Co], [pa[l] for l in Kt], [pb[l] for l in Kt]))
            for t in (A2, B2):
                if t is not A and t is not B:
                    tfree(t)
            for t in tmp:
                if t is not A2 and t is not B2:
                    tfree(t)
            self.records.append((kind, len(Bt), len(Mt), len(Nt), len(Kt), nops0, len(ops)))
            return C

        # ---- pass 1: small subtrees, executed level-by-level with batched launches (no permutes) ----
        small = {}   # id(node) -> PlanTensor
        levels = {}  # level -> list of op tuples
        self.small_stats = []

        def is_small_shape(la, lb, lc):
            sa, sb, sc = set(la), set(lb), set(lc)
            nk = len((sa | sb) - sc)
            return len(lc) <= small_rc and nk <= 8 and len(la) <= 24 and len(lb) <= 24 and len(lc) + nk <= small_rc + 2

        def mark(d):
            """Returns (labels, level) if d's subtree is small, else None."""
            if "tensor_index" in d:
                return leaves[d["tensor_index"]].labels, 0
            ra = mark(d["args"][0])
            rb = mark(d["args"][1])
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

        def tensor_of(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]]
            return small[id(d)]

        # allocate small outputs (kept for the whole plan; they are tiny) and build per-level op lists
        def build_small(d):
            if "tensor_index" in d or not isinstance(small.get(id(d)), tuple):
                return
            build_small(d["args"][0])
            build_small(d["args"][1])
            lc, lev = small[id(d)]
            A = tensor_of(d["args"][0]); Bt_ = tensor_of(d["args"][1])
            off, _ = arena.alloc(4 * nwords(len(lc)))
            C = PlanTensor(lc, off, None)
            pa = {l: i for i, l in enumerate(A.labels)}
            pb = {l: i for i, l in enumerate(Bt_.labels)}
            summed = [l for l in dict.fromkeys(A.labels + Bt_.labels) if l not in set(lc)]
            levels.setdefault(lev, []).append((A.off, Bt_.off, C.off, len(lc), len(summed),
                                               [pa.get(l, -1) for l in lc], [pb.get(l, -1) for l in lc],
                                               [pa.get(l, -1) for l in summed], [pb.get(l, -1) for l in summed]))
            small[id(d)] = C

        def walk_small(d):
            if "tensor_index" in d:
                return
            if id(d) in small:
                build_small(d)
                return
            walk_small(d["args"][0])
            walk_small(d["args"][1])

        walk_small(tree)
        self.small_levels = levels

        # ---- pass 2: large contractions in post-order ----
        def rec(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]]
            if id(d) in small:
                return small[id(d)]
            a = rec(d["args"][0])
            b = rec(d["args"][1])
            c = contract_spec(a, b, d["eins"]["iy"])
            tfree(a)
            tfree(b)
            return c

        root = rec(tree)
        self.arena_bytes = arena.top
        self.base = L.bt_malloc(self.arena_bytes)
        if not self.base:
            raise MemoryError(f"arena of {self.arena_bytes/2**30:.2f} GiB failed")
        self.root_off = root.off
        self.root_r = root.r
        self.plan = L.bt_plan_new()
        B = self.base
        nsmall_ops = 0
        for lev in sorted(levels):
            lst = levels[lev]
            n = len(lst)
            ptrs = (ctypes.c_void_p * (3 * n))()
            rc = (ctypes.c_int * n)(); nk = (ctypes.c_int * n)()
            apos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n)); bpos = (ctypes.c_int * (24 * n))(*([-1] * 24 * n))
            ka = (ctypes.c_int * (8 * n))(*([-1] * 8 * n)); kb = (ctypes.c_int * (8 * n))(*([-1] * 8 * n))
            for i, (ao, bo, co, rci, nki, ap, bp, kai, kbi) in enumerate(lst):
                ptrs[3 * i], ptrs[3 * i + 1], ptrs[3 * i + 2] = B + ao, B + bo, B + co
                rc[i], nk[i] = rci, nki
                for p, v in enumerate(ap): apos[24 * i + p] = v
                for p, v in enumerate(bp): bpos[24 * i + p] = v
                for p, v in enumerate(kai): ka[8 * i + p] = v
                for p, v in enumerate(kbi): kb[8 * i + p] = v
            rcode = L.bt_plan_small_batch(self.plan, n, ptrs, rc, nk, apos, bpos, ka, kb)
            if rcode:
                raise RuntimeError(f"small batch failed {rcode}")
            self.small_stats.append((lev, n))
            nsmall_ops += n
        self.nsmall = nsmall_ops
        nbatches = len(levels)
        self.records = [(k, b, m, n_, kk, i0 + nbatches, i1 + nbatches) for (k, b, m, n_, kk, i0, i1) in self.records]
        self.records.insert(0, ("small", 0, 0, 0, 0, 0, nbatches))
        for op in ops:
            if op[0] == "perm":
                _, ioff, rin, ooff, rout, src, red = op
                rc = L.bt_plan_permute(self.plan, B + ioff, rin, B + ooff, rout, IntArr(src), IntArr(red), len(red))
            elif op[0] == "tc":
                _, a, b, c, bb, mm, nn, kk = op
                rc = L.bt_plan_tc(self.plan, B + a, B + b, B + c, bb, mm, nn, kk, -1)
            elif op[0] == "pad":
                _, ioff, ooff, lrows, kk = op
                rc = L.bt_plan_pad_rows(self.plan, B + ioff, B + ooff, lrows, kk)
            elif op[0] == "cc":
                _, a, b, c, bb, mm, nn, kk = op
                rc = L.bt_plan_cc(self.plan, B + a, B + b, B + c, bb, mm, nn, kk)
            else:
                _, a, b, c, rc_, ap, bp, ka, kb = op
                rc = L.bt_plan_generic(self.plan, B + a, B + b, B + c, rc_, IntArr(ap), IntArr(bp), len(ka),
                                       IntArr(ka), IntArr(kb))
            if rc != 0:
                raise RuntimeError(f"plan op {op[0]} failed rc={rc}")
        L.bt_plan_or_accum(self.plan, B + flag_off, B + root.off)
        self.nops = len(ops) + 1 + nbatches
        self.reset_flag()

    def reset_flag(self):
        self.L.bt_memset(self.base + self.flag_off, 0, 4)

    def flag(self):
        buf = np.zeros(1, dtype=np.uint32)
        self.L.bt_d2h_stream(buf.ctypes.data, self.base + self.flag_off, 4, self.stream)
        return bool(buf[0])

    stream = -1  # -1 = default plan stream; >= 0 = extra stream index (see bt_make_streams)

    def set_leaves(self, words):
        w = np.ascontiguousarray(words, dtype=np.uint32)
        assert w.size == self.leaf_total_words, (w.size, self.leaf_total_words)
        self._leafbuf = w  # keep alive until the async copy is staged
        self.L.bt_h2d_stream(self.base + self.leaf_off, w.ctypes.data, w.nbytes, self.stream)

    def run(self):
        if self.stream >= 0:
            rc = self.L.bt_plan_run_stream(self.plan, self.stream)
        else:
            rc = self.L.bt_plan_run(self.plan, 1 if self.use_graph else 0)
        if rc:
            raise RuntimeError(f"plan run failed {rc}")

    def result(self):
        buf = np.zeros(1, dtype=np.uint32)
        self.L.bt_d2h(buf.ctypes.data, self.base + self.root_off, 4)
        return bool(buf[0] & 1)

    def time_ops(self):
        """Run the plan op-by-op with events; returns per-contraction (kind,B,M,N,K,ms) list."""
        L = self.L
        out = []
        evs = [L.bt_event_create() for _ in range(2)]
        for kind, b, m, n, k, i0, i1 in self.records:
            L.bt_event_record(evs[0])
            L.bt_plan_run_range(self.plan, i0, i1)
            L.bt_event_record(evs[1])
            out.append((kind, b, m, n, k, L.bt_event_elapsed(evs[0], evs[1])))
        return out

    def free(self):
        self.L.bt_plan_free(self.plan)
        self.L.bt_free(self.base)


class SatSlicer:
    """Vectorized per-slice clause-tensor words (all clauses rank <= 5 after slicing)."""

    def __init__(self, clauses, sliced_vars):
        sl = {v: i for i, v in enumerate(sliced_vars)}
        m = len(clauses)
        self.var_mask = np.zeros(m, dtype=np.uint64)
        self.pos_mask = np.zeros(m, dtype=np.uint64)
        self.full = np.zeros(m, dtype=np.uint32)
        self.fbit = np.zeros(m, dtype=np.uint32)
        self.labels = []
        self.fidx_all = []
        for c_i, c in enumerate(clauses):
            labels, idx, p = [], 0, 0
            for l in c:
                v = abs(l) - 1
                if v in sl:
                    self.var_mask[c_i] |= np.uint64(1 << sl[v])
                    if l > 0:
                        self.pos_mask[c_i] |= np.uint64(1 << sl[v])
                else:
                    labels.append(v)
                    if l < 0:
                        idx |= 1 << p
                    p += 1
            r = len(labels)
            self.labels.append(labels)
            self.full[c_i] = np.uint32((1 << (1 << r)) - 1) if r < 5 else np.uint32(0xFFFFFFFF)
            self.fbit[c_i] = np.uint32(1 << idx) if r <= 5 else np.uint32(0)
            self.fidx_all.append(idx)
        # word layout (multi-word leaves for clauses with > 5 unsliced literals)
        ranks = [len(l) for l in self.labels]
        nws = [1 if r <= 5 else 1 << (r - 5) for r in ranks]
        offs = np.concatenate([[0], np.cumsum(nws)[:-1]]).astype(np.int64)
        self.total_words = int(sum(nws))
        self.wide = [(ci, int(offs[ci]), nws[ci], self.fidx_all[ci]) for ci in range(m) if ranks[ci] > 5]
        self.narrow_idx = np.array([ci for ci in range(m) if ranks[ci] <= 5], dtype=np.int64)
        self.narrow_pos = offs[self.narrow_idx]

    def words(self, s):
        """Concatenated leaf words for slice s: one word per clause of rank <= 5, 2^(r-5) words otherwise."""
        s = np.uint64(s)
        satisfied = ((~(s ^ self.pos_mask)) & self.var_mask) != 0
        narrow = np.where(satisfied, self.full, self.full & ~self.fbit).astype(np.uint32)
        if not self.wide:
            return narrow
        out = np.empty(self.total_words, dtype=np.uint32)
        out[self.narrow_pos] = narrow[self.narrow_idx]
        for ci, pos, nw, fi in self.wide:
            out[pos:pos + nw] = 0xFFFFFFFF
            if not satisfied[ci]:
                out[pos + (fi >> 5)] &= np.uint32(~(1 << (fi & 31)) & 0xFFFFFFFF)
        return out


# ------------------------------------------------------------------------------------------


def decide(clauses, tree, sliced=(), small_rc=14, early_exit=True, max_slices=None, use_graph=True):
    """Decide satisfiability on the GPU with a compiled plan (the fast path behind ``run_sat``).

    ``tree`` is an omeco-format dict for the clause network (labels = 0-based variables, as built by
    ``satgen.sat_einsum``); its internal outputs are recomputed for the network with the ``sliced``
    variables removed. Each slice (assignment of the sliced variables) replays one CUDA graph and the
    answer is OR-accumulated on the device. Returns ``(sat, slices_done, info)`` with
    ``info = dict(compile_s, run_s, arena_gib, nslices)``. UNSAT is only proven if all slices ran.
    """
    from .tree_utils import recompute_outputs
    sliced = list(sliced)
    slicer = SatSlicer(clauses, sliced)
    d = recompute_outputs(tree, [list(l) for l in slicer.labels], [])
    t = time.time()
    ct = CompiledTree(d, slicer.labels, use_graph=use_graph, small_rc=small_rc)
    t_compile = time.time() - t
    nsl = 1 << len(sliced)
    todo = nsl if max_slices is None else min(nsl, max_slices)
    ct.reset_flag()
    t = time.time()
    done = 0
    for s in range(todo):
        ct.set_leaves(slicer.words(s))
        ct.run()
        done += 1
        if early_exit and s % 16 == 15 and ct.flag():
            break
    sat = bool(ct.flag())
    lib().bt_sync()
    info = dict(compile_s=t_compile, run_s=time.time() - t, arena_gib=ct.arena_bytes / 2 ** 30, nslices=nsl)
    ct.free()
    return sat, done, info
