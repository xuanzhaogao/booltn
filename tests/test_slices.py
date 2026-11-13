"""Per-slice agreement of the three execution paths with minisat under assumptions.

Random 3-SAT (n=50, alpha=4.26, seed 2) ordered by omeco and sliced to a small space target, then for
every slice: minisat with the sliced variables as assumptions == eager contraction (contract_tree) ==
compiled plan (CompiledTree, without and with CUDA-graph replay); the vectorized slicer leaves equal the
per-clause ones. (Merges the former debug_sat.py / debug_sat2.py.)
"""
import sys
import numpy as np
from pysat.solvers import Minisat22
from booltn.satgen import random_ksat, sat_einsum
from booltn.ordering import optimize_order, prepare_tree
from booltn import engine as E

n, seed = 50, 2
cl = random_ksat(n, round(4.26 * n), 3, seed)
ixs, iy, sizes = sat_einsum(cl, n)
d, sliced, _ = optimize_order(ixs, sizes, sc_target=18, sc_slice=12, ntrials=2, niters=10)
sliced = sliced[:5]  # at most 32 slices
d, _ = prepare_tree(d, ixs, sliced)
print(f"sliced vars {sliced} -> {2 ** len(sliced)} slices")
slicer = E.SatSlicer(cl, sliced)
plans = {g: E.CompiledTree(d, slicer.labels, use_graph=g) for g in (False, True)}
ok = True
for s in range(2 ** len(sliced)):
    fixed = {v: (s >> i) & 1 for i, v in enumerate(sliced)}
    with Minisat22(bootstrap_with=cl) as S:
        truth = S.solve(assumptions=[(v + 1) if b else -(v + 1) for v, b in fixed.items()])
    leaves = [E.BTensor.from_words(w, l) for (l, w) in (E.clause_tensor_words(c, fixed) for c in cl)]
    root = E.contract_tree(d, leaves)
    eager = bool(root.to_bool().reshape(-1)[0])
    root.free()
    w_plan = slicer.words(s)
    w_eager = np.array([E.clause_tensor_words(c, fixed)[1][0] for c in cl], dtype=np.uint32)
    res = {}
    for g, ct in plans.items():
        ct.set_leaves(w_plan); ct.run()
        res[g] = bool(ct.result())
    good = truth == eager == res[False] == res[True] and np.array_equal(w_plan, w_eager)
    ok &= good
    print(f"slice {s:2d}: minisat={truth} eager={eager} plan={res[False]} graph={res[True]} "
          f"leaves_equal={np.array_equal(w_plan, w_eager)} {'PASS' if good else 'FAIL'}")
for ct in plans.values():
    ct.free()
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
