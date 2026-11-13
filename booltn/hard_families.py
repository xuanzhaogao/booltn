"""Structured SAT / #SAT instance families with low treewidth (CDCL-hard or counting-hard).

  tseitin_grid(L, seed)               Tseitin parity on an L x L grid (odd total charge -> UNSAT)
  mutilated_chessboard(n)             domino tiling of an n x n board minus two opposite corners (UNSAT, n even)
  plaquette_csp(L, nforbid, seed)     random CSP: each 2x2 plaquette forbids `nforbid` assignments
  noisy_tseitin_grid(L, nforbid, ...)  Tseitin parity plus random forbidden local assignments

All return (clauses, n) with DIMACS literals. The CaDiCaL-vs-TN scan that used to live here is
experiments/families/cdcl_vs_tn.py.
"""
import random


def tseitin_grid(L, seed=0):
    """Tseitin formula on an L x L grid: edge variables, XOR at each vertex = charge; odd total -> UNSAT."""
    rng = random.Random(seed)
    edges = []
    def v(i, j): return i * L + j
    for i in range(L):
        for j in range(L):
            if i + 1 < L: edges.append((v(i, j), v(i + 1, j)))
            if j + 1 < L: edges.append((v(i, j), v(i, j + 1)))
    var = {e: k + 1 for k, e in enumerate(edges)}
    inc = {u: [] for u in range(L * L)}
    for e in edges:
        inc[e[0]].append(var[e]); inc[e[1]].append(var[e])
    charge = [rng.randrange(2) for _ in range(L * L)]
    if sum(charge) % 2 == 0:
        charge[0] ^= 1
    clauses = []
    for u in range(L * L):
        xs = inc[u]
        d = len(xs)
        for a in range(1 << d):
            par = bin(a).count("1") & 1
            if par != charge[u]:  # forbid assignment a
                clauses.append([(-x if (a >> k) & 1 else x) for k, x in enumerate(xs)])
    return clauses, len(edges)


def mutilated_chessboard(n):
    """Domino tiling of an n x n board minus two opposite corners (UNSAT for even n)."""
    cells = [(i, j) for i in range(n) for j in range(n) if (i, j) not in ((0, 0), (n - 1, n - 1))]
    cs = set(cells)
    edges = []
    for (i, j) in cells:
        if (i + 1, j) in cs: edges.append(((i, j), (i + 1, j)))
        if (i, j + 1) in cs: edges.append(((i, j), (i, j + 1)))
    var = {e: k + 1 for k, e in enumerate(edges)}
    inc = {c: [] for c in cells}
    for e in edges:
        inc[e[0]].append(var[e]); inc[e[1]].append(var[e])
    clauses = []
    for c in cells:
        xs = inc[c]
        clauses.append(list(xs))                       # at least one
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                clauses.append([-xs[a], -xs[b]])      # at most one
    return clauses, len(edges)


def plaquette_csp(L, nforbid, seed=0):
    """Random CSP on an L x L grid: every 2x2 plaquette forbids `nforbid` random assignments (4-clauses).
    No XOR / cardinality structure; treewidth ~ L."""
    rng = random.Random(seed)
    def v(i, j): return i * L + j + 1
    clauses = []
    for i in range(L - 1):
        for j in range(L - 1):
            xs = [v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1)]
            for a in rng.sample(range(16), nforbid):
                clauses.append([(-x if (a >> k) & 1 else x) for k, x in enumerate(xs)])
    return clauses, L * L


def noisy_tseitin_grid(L, nforbid, seed=0, odd=False):
    """Tseitin parity on an L x L grid where every vertex additionally forbids `nforbid` of its
    parity-satisfying local assignments (random). Mixes global parity with non-XOR local constraints."""
    rng = random.Random(seed)
    edges = []
    def v(i, j): return i * L + j
    for i in range(L):
        for j in range(L):
            if i + 1 < L: edges.append((v(i, j), v(i + 1, j)))
            if j + 1 < L: edges.append((v(i, j), v(i, j + 1)))
    var = {e: k + 1 for k, e in enumerate(edges)}
    inc = {u: [] for u in range(L * L)}
    for e in edges:
        inc[e[0]].append(var[e]); inc[e[1]].append(var[e])
    charge = [rng.randrange(2) for _ in range(L * L)]
    if (sum(charge) % 2 == 1) != odd:
        charge[0] ^= 1
    clauses = []
    for u in range(L * L):
        xs = inc[u]; d = len(xs)
        good = [a for a in range(1 << d) if (bin(a).count("1") & 1) == charge[u]]
        bad = set(a for a in range(1 << d) if (bin(a).count("1") & 1) != charge[u])
        k = min(nforbid, len(good) - 1)
        bad |= set(rng.sample(good, k))
        for a in sorted(bad):
            clauses.append([(-x if (a >> b) & 1 else x) for b, x in enumerate(xs)])
    return clauses, len(edges)
