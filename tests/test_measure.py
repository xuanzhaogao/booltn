"""booltn.measure (kernels/measure.c) vs the Python tree walks of BranchSolver._measure: (tc, sc) of the root
tree restricted to random partial assignments must be identical (tc bit for bit). CPU only."""
import random
import sys
import time
from booltn.satgen import random_ksat
from booltn.hard_families import tseitin_grid
from booltn.branching import BranchSolver
from booltn.measure import TreeMeasure, available


def main():
    if not available():
        print("FAIL libbtmeasure.so not built (make -C kernels measure)")
        sys.exit(1)
    ok = True
    cases = [(f"random 3-SAT n={n} seed={s}", random_ksat(n, round(4.3 * n), 3, s), n) for n in (60, 150) for s in (1, 2)]
    cases += [("random 5-SAT n=40", random_ksat(40, 600, 5, 3), 40), ("tseitin 8", *tseitin_grid(8, 1))]
    for name, cl, n in cases:
        bs = BranchSolver(cl, n, cdcl_root=0, cdcl_node=0, cms_root_time=0, cms_node_confl=0, mip_root_time=0,
                          use_xor=False, use_lp=False, ntrials=1, niters=5, leaf="skip")
        bs.t0 = time.time()
        assert bs._root() is None
        tm = TreeMeasure(bs.tree, bs.clauses, bs.n)
        vs = sorted({abs(l) for c in bs.clauses for l in c})
        rng = random.Random(7)
        bad = 0
        for _ in range(300):
            k = rng.choice([0, 1, 3, 10, len(vs) // 3, len(vs)])
            lits = list(bs.root_lits) + [v if rng.random() < .5 else -v for v in rng.sample(vs, min(k, len(vs)))]
            bad += bs._measure(lits)[3:] != tm.tcsc(lits)
        ok &= bad == 0
        print(f"{'PASS' if bad == 0 else 'FAIL'} {name:24s} {tm.nn} tree nodes, 300 assignments, {bad} mismatches")
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
