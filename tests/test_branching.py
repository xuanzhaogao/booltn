"""Branch-and-reduce (booltn.branching) vs minisat, with GPU tensor-network leaves.

CDCL probes are switched off and sc_leaf is small, so instances must branch and every answer comes from
reductions or GPU leaves. Also checks each root reducer on a family it should decide on its own.
"""
import sys
from pysat.solvers import Minisat22
from booltn.satgen import random_ksat
from booltn.hard_families import tseitin_grid, mutilated_chessboard, noisy_tseitin_grid
from booltn.branching import BranchSolver


def truth(cl):
    with Minisat22(bootstrap_with=cl) as s:
        return "SAT" if s.solve() else "UNSAT"


ok = True
cases = [(f"random 3-SAT n={n} seed={s}", random_ksat(n, round(4.26 * n), 3, s), n) for n in (50, 60) for s in (1, 2, 3, 4)]
cases += [("noisy tseitin 8 (even)", *noisy_tseitin_grid(8, 2, 1)), ("noisy tseitin 7 (even)", *noisy_tseitin_grid(7, 1, 3))]
for name, cl, n in cases:
    r = BranchSolver(cl, n, sc_leaf=12, cdcl_root=0, cdcl_node=0, use_xor=False, use_lp=False, ntrials=1, niters=5).solve()
    want = truth(cl)
    good = r.status == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name:26s} {r.status:5s} (minisat {want}) by {r.decided_by:6s} "
          f"nodes {r.stats['nodes']:3d} branches {r.stats['branches']:3d} GPU leaves {r.stats['leaves']:3d}", flush=True)
for name, (cl, n), by in [("tseitin 10 (odd)", tseitin_grid(10, 1), "xor"), ("chessboard 10", mutilated_chessboard(10), "lp")]:
    r = BranchSolver(cl, n, cdcl_root=0).solve()
    good = r.status == "UNSAT" and r.decided_by == by
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name:26s} {r.status} by {r.decided_by} (expected {by})")
# counting with leaves that overflow the CRT range: moduli 101..113 reconstruct counts up to ~10^4 (two moduli
# plus three confirming ones), so larger leaf counts must be split further; the exact total must still come out
import numpy as np


def brute(cl, n):
    X = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    okm = np.ones(1 << n, dtype=bool)
    for c in cl:
        s = np.zeros(1 << n, dtype=bool)
        for l in c:
            s |= X[:, abs(l) - 1] if l > 0 else ~X[:, abs(l) - 1]
        okm &= s
    return int(okm.sum())


for name, (cl, n) in [("random 3-SAT n=22 m=30", (random_ksat(22, 30, 3, 5), 22)), ("random 4-SAT n=22 m=60", (random_ksat(22, 60, 4, 6), 22))]:
    want = brute(cl, n)
    r = BranchSolver(cl, n, mode="count", sc_leaf=20, cdcl_root=0, cdcl_node=0, ntrials=1, niters=5,
                     count_moduli=[101, 103, 107, 109, 113]).solve()
    good = r.count == want and r.stats["leaf_splits"] > 0
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name:26s} count {r.count} (brute force {want}), {r.stats['leaf_splits']} leaf splits, "
          f"{r.stats['leaves']} leaves")
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
