"""Wide clauses (k = 6, 7: clause tensors larger than one word) end to end, with and without slicing, vs minisat."""
import sys
import omeco
from booltn.satgen import random_ksat, sat_einsum
from booltn.tree_utils import recompute_outputs
from booltn import engine as E
from pysat.solvers import Minisat22
ok = True
for n, m, k, seed, sliced in [(24, 180, 7, 1, []), (24, 250, 7, 2, []), (22, 120, 6, 3, [0, 1]), (26, 300, 7, 4, [2, 5, 7])]:
    cl = random_ksat(n, m, k, seed)
    ixs, iy, sizes = sat_einsum(cl, n)
    tree = omeco.optimize_code(ixs, [], sizes, omeco.TreeSA(ntrials=2, niters=10))
    with Minisat22(bootstrap_with=cl) as S:
        truth = S.solve()
    slicer = E.SatSlicer(cl, sliced)
    d = recompute_outputs(tree.to_dict(), [list(l) for l in slicer.labels], [])
    ct = E.CompiledTree(d, slicer.labels, small_rc=12)
    ct.reset_flag()
    for s in range(1 << len(sliced)):
        ct.set_leaves(slicer.words(s)); ct.run()
    res = ct.flag(); ct.free()
    ok &= res == truth
    print(f"{'PASS' if res == truth else 'FAIL'} {k}-SAT n={n} m={m} sliced={sliced}: gpu={res} minisat={truth}")
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
