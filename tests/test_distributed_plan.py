"""Distribution planner + numpy emulation against minisat (CPU only, no GPU needed; needs omeco, python-sat).

Random 3-SAT and grid 3-SAT instances are ordered for a logical space cap + p, planned for 2^p GPUs with
a local cap, and every global slice is executed by the numpy emulator; the OR over the slices must equal
minisat with the sliced variables as assumptions.
"""
import sys
sys.setrecursionlimit(100000)
from pysat.solvers import Minisat22
from booltn.satgen import random_ksat, grid_ksat, sat_einsum
from booltn.ordering import optimize_order, prepare_tree
from booltn.tree_utils import complexity
from booltn import distributed as D
from booltn.tree_utils import H100_MODEL

# cost models that make exchanges attractive on tiny networks (no latency, no "small is free" shortcut)
FREE_LINK = dict(bw=1e30, latency=0.0)
TINY = dict(H100_MODEL, small_rank=0, launch=0.0)

ok = True
cases = [("rand", 34, 1, 10, 1), ("rand", 34, 2, 10, 2), ("rand", 36, 3, 9, 3), ("grid", 7, 1, 9, 1),
         ("grid", 7, 2, 9, 2), ("grid", 8, 3, 10, 3), ("rand", 30, 4, 8, 2), ("grid", 8, 5, 9, 1),
         ("grid", 9, 6, 10, 2), ("rand", 38, 7, 10, 1), ("grid", 8, 8, 8, 3), ("rand", 40, 9, 11, 2)]
for fam, size, seed, cap, p in cases:
    if fam == "rand":
        cl, n = random_ksat(size, round(4.26 * size), 3, seed), size
    else:
        cl, n = grid_ksat(size, 3.0, 3, 1, seed)
    ixs, iy, sizes = sat_einsum(cl, n)
    tree, sliced, _ = optimize_order(ixs, sizes, sc_target=cap + p + 4, sc_slice=cap + p, ntrials=2, niters=8)
    d, ixs_sl = prepare_tree(tree, ixs, sliced)
    tc, sc = complexity(d, ixs_sl)
    sch = D.plan_distribution(d, ixs_sl, p, cap, link=FREE_LINK, model=TINY, min_xrank=1)
    ref = D.plan_distribution(d, ixs_sl, p, cap, link=FREE_LINK, model=TINY, exchanges=False)
    if sch is None:
        print(f"{fam} {size} s{seed} cap {cap} p {p}: sc={sc:.0f} no feasible schedule (skipped)")
        continue
    tot, rmax = sch.model_s, sch.max_local_rank()
    nsl = min(1 << len(sliced), 16)
    good = rmax <= cap
    for s in range(nsl):
        fixed = {v: (s >> i) & 1 for i, v in enumerate(sliced)}
        with Minisat22(bootstrap_with=cl) as S:
            truth = S.solve(assumptions=[(v + 1) if b else -(v + 1) for v, b in fixed.items()])
        got = D.emulate(sch, cl, n, sliced, s)
        good &= got == truth
    ok &= good
    print(f"{fam} {size} s{seed} cap {cap} p {p}: sc={sc:.0f} tc={tc:.1f} slices checked {nsl}, "
          f"{len(sch.exchanges)} exchanges ({''.join('d' if x['newtop'] is None else 'l' for x in sch.exchanges)}), "
          f"local max rank {rmax}, model {tot:.3g}s (no exchange: {'-' if ref is None else format(ref.model_s, '.3g')}s) "
          f"{'PASS' if good else 'FAIL'}")
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
