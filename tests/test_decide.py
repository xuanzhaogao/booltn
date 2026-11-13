"""engine.decide (compiled fast path, the library API) vs minisat on random 3-SAT, unsliced and with a few
slices (booltn.slicer.greedy_slice to min(width - 2, 30) -> small arenas, at most a few thousand slices)."""
import sys
from pysat.solvers import Minisat22
from booltn.satgen import random_ksat, sat_einsum
from booltn.ordering import optimize_order
from booltn.tree_utils import complexity
from booltn.slicer import greedy_slice
from booltn import engine as E

ok = True
for n, alpha, seed, sliced_run in [(40, 4.26, 1, False), (50, 4.26, 2, True), (48, 4.0, 3, True), (52, 4.6, 4, True)]:
    cl = random_ksat(n, round(alpha * n), 3, seed)
    ixs, iy, sizes = sat_einsum(cl, n)
    tree, _, _ = optimize_order(ixs, sizes, sc_target=30, ntrials=2, niters=10)
    sc = complexity(tree, ixs)[1]
    sliced = greedy_slice(tree, ixs, min(sc - 2, 30), kind="tc")[0] if sliced_run else []
    with Minisat22(bootstrap_with=cl) as S:
        truth = S.solve()
    sat, done, info = E.decide(cl, tree, sliced, early_exit=False)
    good = sat == truth and done == info["nslices"]
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} n={n} alpha={alpha} seed={seed}: sc={sc}, {len(sliced)} sliced, {done} slices, "
          f"gpu={sat} minisat={truth} ({info['run_s']*1e3:.1f} ms)", flush=True)
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
