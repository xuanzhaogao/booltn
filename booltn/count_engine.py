"""Exact model counting (#SAT) by tensor-network contraction on the GPU (libcount8.so).

Each pass computes the count modulo a prime p <= 251 (one byte per tensor entry, int8 tensor cores / dp4a);
the exact count is rebuilt by CRT over enough primes. The compiled plan (CUDA graph) is shared by all primes:
p lives in device memory and every kernel reads it.

Layout conventions match engine.py: a tensor with labels [l0..l_{r-1}] has 2^r entries, l0 fastest.
"""
import ctypes, os, math, time
import numpy as np
from .engine import Arena, PlanTensor, plan as classify, LIB_DIR

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
        L.c8_plan_free.argtypes = [VP]
        L.c8_plan_permute.argtypes = [VP, VP, VP, ctypes.c_int, VP, VP, ctypes.c_int, VP]
        L.c8_plan_cc.argtypes = [VP, VP, VP, VP, VP] + [ctypes.c_int] * 4 + [VP]
        L.c8_plan_tc.argtypes = [VP, VP, VP, VP] + [ctypes.c_int] * 4 + [VP]
        L.c8_plan_small_batch.argtypes = [VP, ctypes.c_int, VP] + [VP] * 6 + [VP]
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

    def __init__(self, tree, leaf_labels, small_rc=12, use_graph=True, tc_min_k=5):
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
        leaf_off, _ = arena.alloc(32 * nleaf)
        self.p_off, self.acc_off, self.leaf_off, self.nleaf = p_off, acc_off, leaf_off, nleaf
        self.leaf_ranks = [len(l) for l in leaf_labels]
        leaves = [PlanTensor(l, leaf_off + 32 * i, None) for i, l in enumerate(leaf_labels)]
        ops = []

        def nbytes(r):
            return 1 << r

        def talloc(labels):
            off, tid = arena.alloc(nbytes(len(labels)))
            return PlanTensor(labels, off, tid)

        def tfree(t):
            if t.tid is not None:
                arena.free(t.tid)

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

        def tensor_of(d):
            return leaves[d["tensor_index"]] if "tensor_index" in d else small[id(d)]

        def build_small(d):
            if "tensor_index" in d or not isinstance(small.get(id(d)), tuple):
                return
            build_small(d["args"][0]); build_small(d["args"][1])
            lc, lev = small[id(d)]
            A, B = tensor_of(d["args"][0]), tensor_of(d["args"][1])
            off, _ = arena.alloc(nbytes(len(lc)))
            pa = {l: i for i, l in enumerate(A.labels)}
            pb = {l: i for i, l in enumerate(B.labels)}
            summed = [l for l in dict.fromkeys(A.labels + B.labels) if l not in set(lc)]
            levels.setdefault(lev, []).append((A.off, B.off, off, len(lc), len(summed),
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
                A = permute_spec(A, [l for l in A.labels if l not in Ra], Ra); tmp.append(A)
            if Rb:
                B = permute_spec(B, [l for l in B.labels if l not in Rb], Rb); tmp.append(B)
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
            A2, B2 = permute_spec(A, Ao), permute_spec(B, Bo)
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

        def rec(d):
            if "tensor_index" in d:
                return leaves[d["tensor_index"]]
            if id(d) in small:
                return small[id(d)]
            a, b = rec(d["args"][0]), rec(d["args"][1])
            c = contract_spec(a, b, d["eins"]["iy"])
            tfree(a); tfree(b)
            return c

        root = rec(tree)
        assert root.r == 0, f"root must be a scalar, got rank {root.r}"
        self.arena_bytes = arena.top
        self.base = L.c8_malloc(self.arena_bytes)
        if not self.base:
            raise MemoryError(f"arena {self.arena_bytes/2**30:.2f} GiB")
        B = self.base
        pp = B + p_off
        self.plan = L.c8_plan_new()
        nb = 0
        for lev in sorted(levels):
            lst = levels[lev]; n = len(lst)
            ptrs = (VP * (3 * n))()
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
            r = L.c8_plan_small_batch(self.plan, n, ptrs, rc, nk, apos, bpos, ka, kb, pp)
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
        v = np.array([p], dtype=np.int32)
        self.L.c8_h2d(self.base + self.p_off, v.ctypes.data, 4, self.stream)
        self.L.c8_memset(self.base + self.acc_off, 0, 4, self.stream)

    def set_leaves(self, buf):
        """buf: uint8 array of 32*nleaf bytes (leaf i at [32i, 32i + 2^rank_i))."""
        self._buf = np.ascontiguousarray(buf, dtype=np.uint8)
        self.L.c8_h2d(self.base + self.leaf_off, self._buf.ctypes.data, self._buf.nbytes, self.stream)

    def run(self):
        r = self.L.c8_plan_run(self.plan, self.stream, 1 if self.use_graph else 0)
        if r:
            raise RuntimeError(f"plan run failed {r}")

    def residue(self):
        v = np.zeros(1, dtype=np.int32)
        self.L.c8_d2h(v.ctypes.data, self.base + self.acc_off, 4, self.stream)
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


def count_models(clauses, n, tree, sliced=(), primes=None, small_rc=12, verbose=False, mode="stable", confirm=3):
    """Exact model count of a CNF over variables 1..n using a contraction tree (dict) for the einsum
    built from the clause variable lists with `sliced` variables removed. Returns (count, info).

    mode="rigorous": use primes until their product exceeds 2^(#used vars) (always exact; may run out of
                     8-bit primes for > ~400 used variables).
    mode="stable"  : early-terminating CRT: stop when the reconstruction has not changed for `confirm`
                     extra primes (error probability ~ prod 1/p_i over the confirming primes, ~1e-7 for 3).
    """
    from .tree_utils import recompute_outputs
    sliced = list(sliced)
    used = sorted({abs(l) - 1 for c in clauses for l in c})
    free_vars = n - len(used)
    slicer = CountSlicer(clauses, sliced)
    ixs = [list(l) for l in slicer.labels]
    d = recompute_outputs(tree, ixs, [])
    t0 = time.time()
    ct = CountTree(d, slicer.labels, small_rc=small_rc)
    t_compile = time.time() - t0
    # rigorous bound: count <= 2^(n_used) per... use 2^n_used (free vars handled exactly by a power of 2)
    need_bits = len(used) + 1
    if primes is None:
        if mode == "rigorous":
            primes, bits = [], 0.0
            for p in primes_desc():
                if bits > need_bits:
                    break
                primes.append(p); bits += math.log2(p)
            if bits <= need_bits:
                raise ValueError(f"need > {need_bits} bits but 8-bit primes give only {bits:.0f}")
        else:
            primes = primes_desc()
    nsl = 1 << len(sliced)
    leaf_bufs = [slicer.leaves(s) for s in range(nsl)] if nsl <= 4096 else None
    residues, used_primes = [], []
    t1 = time.time()
    last, stable = None, 0
    for p in primes:
        ct.set_prime(p)
        for s in range(nsl):
            ct.set_leaves(leaf_bufs[s] if leaf_bufs is not None else slicer.leaves(s))
            ct.run()
        residues.append(ct.residue()); used_primes.append(p)
        if mode != "rigorous":
            val, _ = crt(residues, used_primes)
            stable = stable + 1 if val == last else 0
            last = val
            if stable >= confirm:
                break
    else:
        if mode != "rigorous":
            raise ValueError("8-bit primes exhausted before the CRT reconstruction stabilized")
    primes = used_primes
    t_run = time.time() - t1
    count, M = crt(residues, primes)
    count <<= free_vars
    info = dict(primes=len(primes), slices=nsl, compile_s=t_compile, run_s=t_run, arena_gib=ct.arena_bytes / 2 ** 30,
                nops=ct.nops, per_pass_s=t_run / len(primes))
    ct.free()
    return count, info


def profile_tree(clauses, n, tree, sliced=(), prime=251, small_rc=12):
    """Per-contraction GPU timing of one pass (one slice): returns list of (kind,B,M,N,K,ms)."""
    from .tree_utils import recompute_outputs
    slicer = CountSlicer(clauses, list(sliced))
    d = recompute_outputs(tree, [list(l) for l in slicer.labels], [])
    ct = CountTree(d, slicer.labels, small_rc=small_rc, use_graph=False)
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
