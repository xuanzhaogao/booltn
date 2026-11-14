"""Write a case file for test_distributed.py (CPU only; needs omeco, and python-sat for the ground truth).

  python make_distributed_case.py out.json family arg seed --sc 21 [--trials 2] [--truth minisat|none]

family: rand (arg = n, 3-SAT at ratio 4.26), grid (arg = L), tseitin, chess, plaq (see booltn.hard_families).
The order is optimized with omeco (TreeSA + TreeSASlicer) for a space cap of 2^sc bits after slicing.
The JSON holds clauses, n, tree, sliced, truth (bool or null) and meta (family, arg, seed, sc, tc, model_s).
"""
import sys, json, argparse
sys.setrecursionlimit(200000)
from booltn.satgen import random_ksat, grid_ksat, sat_einsum
from booltn import hard_families as HF
from booltn.tree_utils import complexity, tree_h100_time
from booltn.ordering import optimize_order, prepare_tree


def instance(family, arg, seed):
    if family == "grid":
        return grid_ksat(arg, 3.0, 3, 1, seed)
    if family == "rand":
        return random_ksat(arg, round(4.26 * arg), 3, seed), arg
    if family == "tseitin":
        return HF.tseitin_grid(arg, seed)
    if family == "chess":
        return HF.mutilated_chessboard(arg)
    if family == "plaq":
        return HF.plaquette_csp(arg, 4, seed)
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("family"); ap.add_argument("arg", type=int); ap.add_argument("seed", type=int)
    ap.add_argument("--sc", type=int, required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--niters", type=int, default=20)
    ap.add_argument("--truth", default="minisat")
    a = ap.parse_args()
    clauses, n = instance(a.family, a.arg, a.seed)
    ixs, iy, sizes = sat_einsum(clauses, n)
    best = None
    for _ in range(a.trials):
        tree, sliced, _ = optimize_order(ixs, sizes, a.sc + 6, sc_slice=a.sc, niters=a.niters)
        d, ixs_sl = prepare_tree(tree, ixs, sliced)
        tc, sc = complexity(d, ixs_sl)
        secs = tree_h100_time(d, ixs_sl) * 2.0 ** len(sliced)
        if sc <= a.sc and (best is None or secs < best[0]):
            best = (secs, tree, sliced, tc, sc)
    if best is None:
        sys.exit(f"no order within sc = {a.sc} found in {a.trials} trials")
    secs, tree, sliced, tc, sc = best
    truth = None
    if a.truth == "minisat":
        from pysat.solvers import Minisat22
        with Minisat22(bootstrap_with=clauses) as S:
            truth = bool(S.solve())
    with open(a.out, "w") as f:
        json.dump(dict(clauses=clauses, n=n, tree=tree, sliced=sliced, truth=truth,
                       meta=dict(family=a.family, arg=a.arg, seed=a.seed, sc=sc, tc=tc, model_s=secs)), f)
    print(f"{a.out}: n={n} m={len(clauses)} sc={sc:.0f} tc/slice={tc:.2f} sliced={len(sliced)} model {secs:.3g}s truth={truth}")


if __name__ == "__main__":
    main()
