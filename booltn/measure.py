"""Fast (tc, sc) of the root contraction tree restricted to a partial assignment (branching lookahead).

``TreeMeasure(tree, clauses, n).tcsc(lits)`` returns the same numbers as
``complexity(recompute_outputs(prune_tree(tree, keep), ixs, []), ixs)`` for the residual of ``clauses`` under
``lits`` (see ``BranchSolver._measure``), computed by ``kernels/measure.c`` (host code, built into
``lib/libbtmeasure.so`` by ``make -C kernels``) with bitset label sets: about 100x faster than the Python
tree walks. ``available()`` is False when the library is not built; callers then use the Python path.
The same library holds ``btx_implied`` (``kernels/gf2.c``), the C path of ``reduce.XorSystem``.
"""
import ctypes
import os

import numpy as np

from .engine import LIB_DIR

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        p = os.path.join(LIB_DIR, "libbtmeasure.so")
        if not os.path.exists(p):
            _LIB = False
            return _LIB
        L = ctypes.CDLL(p)
        P = ctypes.c_void_p
        L.btm_tcsc.argtypes = [ctypes.c_int, P, P, P, P, P, ctypes.c_int, P, P, P, P, P,
                               ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32)]
        L.btm_tcsc.restype = ctypes.c_int
        if hasattr(L, "btm_iy"):               # ordered output labels (TreeMeasure.big_labels / .tree)
            L.btm_iy.argtypes = [ctypes.c_int, P, P, P, P, P, ctypes.c_int, P, P, P, P, P, P, P, ctypes.c_int, P]
            L.btm_iy.restype = ctypes.c_int
        if hasattr(L, "btx_implied"):          # kernels/gf2.c (XorSystem.implied); absent in an older build
            I = ctypes.c_int
            L.btx_implied.argtypes = [I, I, P, P, P, P, P, P, P, P, P, P, I, P, P, P, P, P]
            L.btx_implied.restype = I
        _LIB = L
    return _LIB


def available():
    return bool(_lib())


class TreeMeasure:
    def __init__(self, tree, clauses, n):
        """tree: omeco dict over `clauses` (leaf i = clauses[i]); n: number of variables (DIMACS 1..n)."""
        leaf, left, right = [], [], []
        stack = [(tree, False)]
        while stack:                       # iterative post-order (left, right, node)
            d, done = stack.pop()
            if "tensor_index" in d:
                leaf.append(d["tensor_index"]); left.append(-1); right.append(-1)
                d["_j"] = len(leaf) - 1
            elif done:
                leaf.append(-1); left.append(d["args"][0].pop("_j")); right.append(d["args"][1].pop("_j"))
                d["_j"] = len(leaf) - 1
            else:
                stack += [(d, True), (d["args"][1], False), (d["args"][0], False)]
        tree.pop("_j", None)
        self.nn = len(leaf)
        self.leaf = np.array(leaf, dtype=np.int32)
        self.left = np.array(left, dtype=np.int32)
        self.right = np.array(right, dtype=np.int32)
        self.cptr = np.zeros(len(clauses) + 1, dtype=np.int32)
        self.cptr[1:] = np.cumsum([len(c) for c in clauses])
        self.clit = np.array([l for c in clauses for l in c], dtype=np.int32)
        self.n = n
        W = n // 64 + 1
        self.sub = np.zeros(self.nn * W, dtype=np.uint64)
        self.ext = np.zeros(self.nn * W, dtype=np.uint64)
        self.eff = np.zeros(self.nn, dtype=np.int32)
        self.terms = np.zeros(self.nn, dtype=np.float64)
        self.val = np.zeros(n + 1, dtype=np.int8)

    def tcsc(self, lits):
        L = _lib()
        val = self.val
        val[:] = 0
        if lits:
            a = np.asarray(lits, dtype=np.int64)
            val[np.abs(a)] = np.where(a > 0, 1, -1)
        tc, sc = ctypes.c_double(), ctypes.c_int32()
        p = lambda x: x.ctypes.data
        L.btm_tcsc(self.nn, p(self.leaf), p(self.left), p(self.right), p(self.cptr), p(self.clit), self.n, p(val),
                   p(self.sub), p(self.ext), p(self.eff), p(self.terms), ctypes.byref(tc), ctypes.byref(sc))
        self._iy = None
        return tc.value, sc.value

    # --- after tcsc(lits), for the same assignment ---------------------------------------------------------
    def has_iy(self):
        return bool(_lib()) and hasattr(_lib(), "btm_iy")

    def keep(self):
        """Clause indices of the live leaves (clauses not satisfied), ascending (BranchSolver._residual)."""
        j = np.nonzero((self.eff == np.arange(self.nn)) & (self.leaf >= 0))[0]
        return np.sort(self.leaf[j])

    def residual(self):
        """BranchSolver._residual: the unassigned literals of each live clause, in clause and literal order."""
        lv = self.val[np.abs(self.clit)] * np.sign(self.clit).astype(np.int8)
        cl = np.repeat(np.arange(len(self.cptr) - 1), np.diff(self.cptr))
        sat = np.zeros(len(self.cptr) - 1, dtype=bool)
        sat[cl[lv > 0]] = True
        m = (lv == 0) & ~sat[cl]
        flat = self.clit[m].tolist()
        cnt = np.bincount(cl[m], minlength=len(sat))[~sat]
        ends = np.cumsum(cnt).tolist()
        starts = [0] + ends[:-1]
        return [flat[a:b] for a, b in zip(starts, ends)]

    def iy(self):
        """(off, len, arena list): output labels of each live internal node in recompute_outputs order."""
        if self._iy is None:
            L = _lib()
            p = lambda x: x.ctypes.data
            if not hasattr(self, "off"):
                self.off = np.zeros(self.nn, dtype=np.int32)
                self.len = np.zeros(self.nn, dtype=np.int32)
                self.stamp = np.zeros(self.n + 1, dtype=np.int32)
                self.arena = np.zeros(max(1024, 16 * self.nn), dtype=np.int32)
            while True:
                used = L.btm_iy(self.nn, p(self.leaf), p(self.left), p(self.right), p(self.cptr), p(self.clit),
                                self.n, p(self.val), p(self.sub), p(self.ext), p(self.eff), p(self.off),
                                p(self.len), p(self.arena), len(self.arena), p(self.stamp))
                if used >= 0:
                    break
                self.arena = np.zeros(2 * len(self.arena), dtype=np.int32)
            self._iy = (self.off, self.len, self.arena[:used].tolist())
        return self._iy

    def big_labels(self, sc_target):
        """slicer._big_labels of the restricted tree: labels of output tensors larger than sc_target, by summed
        2^(size - sc_target), ties in first-occurrence order (post-order, iy order)."""
        off, ln, ar = self.iy()
        j = np.nonzero((self.eff == np.arange(self.nn)) & (self.leaf < 0) & (ln > sc_target))[0]
        w = {}
        for o, k in zip(off[j].tolist(), ln[j].tolist()):
            x = 2.0 ** (k - sc_target)
            for l in ar[o:o + k]:
                w[l] = w.get(l, 0) + x
        return sorted(w, key=lambda l: -w[l])

    def tree(self, ixs):
        """recompute_outputs(prune_tree(tree, keep()), ixs, []): ixs[i] = labels of the i-th live clause."""
        off, ln, ar = self.iy()
        live = np.nonzero(self.eff == np.arange(self.nn))[0].tolist()
        keep = self.keep()
        pos = dict(zip(keep.tolist(), range(len(keep))))
        leaf, off, ln = self.leaf.tolist(), off.tolist(), ln.tolist()
        stack = []                      # (dict, its labels as an operand)
        for j in live:                  # post-order of the pruned tree
            c = leaf[j]
            if c >= 0:
                i = pos[c]
                stack.append(({"tensor_index": i}, list(ixs[i])))
            else:
                r, lb = stack.pop()
                l, la = stack.pop()
                out = ar[off[j]:off[j] + ln[j]]
                stack.append(({"args": [l, r], "eins": {"ixs": [la, lb], "iy": out}}, out))
        return stack[-1][0] if stack else None
