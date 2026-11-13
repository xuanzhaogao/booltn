"""Correctness tests for the GPU exact model counter against brute force and closed forms."""
import sys, time
import numpy as np
import omeco
from booltn.satgen import random_ksat, sat_einsum
from booltn.hard_families import tseitin_grid
from booltn.count_engine import count_models


def brute(clauses, n):
    # vectorized over all 2^n assignments
    X = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    ok = np.ones(1 << n, dtype=bool)
    for c in clauses:
        sat = np.zeros(1 << n, dtype=bool)
        for l in c:
            sat |= X[:, abs(l) - 1] if l > 0 else ~X[:, abs(l) - 1]
        ok &= sat
    return int(ok.sum())


def domino(n):
    """Domino tilings of an n x n board: var per edge, exactly one per cell."""
    cells = [(i, j) for i in range(n) for j in range(n)]
    edges = []
    for (i, j) in cells:
        if i + 1 < n: edges.append(((i, j), (i + 1, j)))
        if j + 1 < n: edges.append(((i, j), (i, j + 1)))
    var = {e: k + 1 for k, e in enumerate(edges)}
    inc = {c: [] for c in cells}
    for e in edges:
        inc[e[0]].append(var[e]); inc[e[1]].append(var[e])
    cl = []
    for c in cells:
        xs = inc[c]
        cl.append(list(xs))
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                cl.append([-xs[a], -xs[b]])
    return cl, len(edges)


def tree_for(clauses, n, sc=30):
    ixs, iy, sizes = sat_einsum(clauses, n)
    t = omeco.optimize_code(ixs, [], sizes, omeco.TreeSA(ntrials=2, niters=10, score=omeco.ScoreFunction(sc_target=sc, sc_weight=4.0)))
    return t.to_dict()


def check(name, clauses, n, want, small_rc=12, sliced=()):
    d = tree_for(clauses, n)
    t = time.time()
    got, info = count_models(clauses, n, d, sliced=sliced, small_rc=small_rc)
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got} want {want} ({info['primes']} primes, {time.time()-t:.2f}s)", flush=True)
    return ok


if __name__ == "__main__":
    allok = True
    for n, alpha, seed in [(12, 2.0, 1), (16, 3.0, 2), (18, 2.5, 3), (20, 2.0, 4), (20, 3.5, 5), (16, 4.26, 6)]:
        cl = random_ksat(n, round(alpha * n), 3, seed)
        for src in (12, 0):  # with and without batched small contractions
            allok &= check(f"random3sat n={n} a={alpha} small_rc={src}", cl, n, brute(cl, n), small_rc=src)
    # free variables (unused) multiply the count by 2^free
    cl = random_ksat(14, 20, 3, 7); cl = [[l for l in c] for c in cl]
    allok &= check("free vars (n=18, 14 used)", cl, 18, brute(cl, 18))
    for L in [3, 4, 6]:
        cl, n = tseitin_grid(L, seed=1)  # odd total charge -> 0
        allok &= check(f"tseitin odd {L}x{L}", cl, n, 0)
    for n, want in [(4, 36), (6, 6728), (8, 12988816), (10, 258584046368)]:
        cl, nv = domino(n)
        allok &= check(f"domino tilings {n}x{n}", cl, nv, want)
    # slicing: sum over slices
    cl = random_ksat(20, 50, 3, 9)
    allok &= check("random3sat n=20 sliced [0,1,2]", cl, 20, brute(cl, 20), sliced=[0, 1, 2])
    print("ALL PASSED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
