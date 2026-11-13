"""Fast (tc, sc) of the root contraction tree restricted to a partial assignment (branching lookahead).

``TreeMeasure(tree, clauses, n).tcsc(lits)`` returns the same numbers as
``complexity(recompute_outputs(prune_tree(tree, keep), ixs, []), ixs)`` for the residual of ``clauses`` under
``lits`` (see ``BranchSolver._measure``), computed by ``kernels/measure.c`` (host code, built into
``lib/libbtmeasure.so`` by ``make -C kernels``) with bitset label sets: about 100x faster than the Python
tree walks. ``available()`` is False when the library is not built; callers then use the Python path.
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
        return tc.value, sc.value
