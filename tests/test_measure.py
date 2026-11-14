"""booltn.measure (kernels/measure.c) vs the Python tree walks (BranchSolver._residual, prune_tree,
recompute_outputs, complexity, slicer._big_labels): for the root tree restricted to random partial assignments,
(tc, sc) must be identical (tc bit for bit), and so must the live clauses, the residual clauses, the restricted
tree (output labels in the same order) and the branching candidates. CPU only."""
import random
import sys
import time
from booltn.satgen import random_ksat
from booltn.hard_families import tseitin_grid
from booltn.branching import BranchSolver
from booltn.measure import TreeMeasure, available
from booltn.slicer import _big_labels
from booltn.tree_utils import complexity, prune_tree, recompute_outputs


def py_measure(bs, lits):
    """The Python path of BranchSolver._measure: (keep, res, d, ixs, tc, sc)."""
    keep, res = bs._residual(lits)
    if not res:
        return keep, res, None, None, 0.0, 0
    ixs = [[abs(l) - 1 for l in c] for c in res]
    d = recompute_outputs(prune_tree(bs.tree, keep), ixs, [])
    return (keep, res, d, ixs) + complexity(d, ixs)


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
            keep, res, d, ixs, tc, sc = py_measure(bs, lits)
            bad += (tc, sc) != tm.tcsc(lits)
            bad += keep != tm.keep().tolist()
            if res and tm.has_iy():                # btm_iy: library built from this version
                bad += res != tm.residual()
                bad += d != tm.tree(ixs)
                for tgt in {0, 3, 8, max(0, sc - 3)}:
                    bad += _big_labels(d, ixs, tgt) != tm.big_labels(tgt)
            m = bs._measure(lits)                    # BranchSolver's own path (C when built)
            bad += (m[0] != res) + (m[1] != d) + (tuple(m[3:]) != (tc, sc))
        ok &= bad == 0
        print(f"{'PASS' if bad == 0 else 'FAIL'} {name:24s} {tm.nn} tree nodes, 300 assignments, {bad} mismatches "
              "(tc/sc, live clauses, residual, tree, candidates)")
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
