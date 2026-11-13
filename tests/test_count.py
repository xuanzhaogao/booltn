"""Correctness tests for the GPU exact model counter against brute force and closed forms."""
import sys, time
import numpy as np
import omeco
from booltn.satgen import random_ksat, sat_einsum
from booltn.hard_families import tseitin_grid
from booltn.count_engine import count_models, primes16_desc


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


def hardcore_grid(h, w):
    """Independent sets of the h x w grid graph: clause (-u, -v) per edge."""
    var = lambda i, j: i * w + j + 1
    cl = []
    for i in range(h):
        for j in range(w):
            if i + 1 < h: cl.append([-var(i, j), -var(i + 1, j)])
            if j + 1 < w: cl.append([-var(i, j), -var(i, j + 1)])
    return cl, h * w


def hardcore_count(h, w):
    """Exact number of independent sets by the row transfer matrix."""
    states = [s for s in range(1 << w) if s & (s >> 1) == 0]
    vec = {s: 1 for s in states}
    for _ in range(h - 1):
        vec = {t: sum(c for s_, c in vec.items() if s_ & t == 0) for t in states}
    return sum(vec.values())


def tree_for(clauses, n, sc=30):
    ixs, iy, sizes = sat_einsum(clauses, n)
    if len(ixs) == 1:
        return {"tensor_index": 0}
    t = omeco.optimize_code(ixs, [], sizes, omeco.TreeSA(ntrials=2, niters=10, score=omeco.ScoreFunction(sc_target=sc, sc_weight=4.0)))
    return t.to_dict()


def check(name, clauses, n, want, small_rc=12, sliced=(), primes=None):
    d = tree_for(clauses, n)
    t = time.time()
    got, info = count_models(clauses, n, d, sliced=sliced, small_rc=small_rc, primes=primes)
    ok = got == want
    show = (lambda x: str(x)) if want < 10 ** 30 else (lambda x: f"{str(x)[:12]}...e{len(str(x)) - 1}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {show(got)} want {show(want)} ({info['primes']} moduli, "
          f"{info['primes16']} of them 16-bit, {time.time()-t:.2f}s)", flush=True)
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
    # one-clause networks (the tree is a lone leaf; branch-and-reduce leaves can be this small)
    for cl, nv, want in [([[1, -2]], 2, 3), ([[1, -2, 3]], 5, 28), ([[-4]], 4, 8)]:
        for P in (None, primes16_desc()[:6]):
            allok &= check(f"one clause {cl} n={nv}" + (" 16-bit" if P else ""), cl, nv, want, primes=P)
    # 16-bit residues only (width-2 plans for every kernel path)
    P16 = primes16_desc()[:12]
    for n, alpha, seed in [(20, 2.0, 4), (18, 2.5, 3)]:
        cl = random_ksat(n, round(alpha * n), 3, seed)
        for src in (12, 0):
            allok &= check(f"16-bit random3sat n={n} a={alpha} small_rc={src}", cl, n, brute(cl, n), small_rc=src, primes=P16)
    for n, want in [(8, 12988816), (10, 258584046368)]:
        cl, nv = domino(n)
        for src in (12, 0):
            allok &= check(f"16-bit domino {n}x{n} small_rc={src}", cl, nv, want, small_rc=src, primes=P16)
    allok &= check("16-bit random3sat n=20 sliced [0,1,2]", cl := random_ksat(20, 50, 3, 9), 20, brute(cl, 20),
                   sliced=[0, 1, 2], primes=P16)
    # beyond the 8-bit CRT range (~10^109.2): the 54 8-bit moduli, then 16-bit primes
    cl, nv = hardcore_grid(66, 10)
    for src in (12, 0):
        allok &= check(f"independent sets 66x10 (> 10^109) small_rc={src}", cl, nv, hardcore_count(66, 10), small_rc=src)
    print("ALL PASSED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
