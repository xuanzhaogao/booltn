"""Exact model counting (#SAT) on the GPU (CLI: ``booltn-count``).

  python -m booltn.run_count --cnf f.cnf [--sc 32] [--tree tree.json] [--rigorous]

Steps: contraction order (omeco TreeSA, or --tree) -> slicing to --sc -> one compiled int8 plan replayed once
per prime p <= 251 (count mod p, summed over slices) -> CRT. Default stopping rule: the reconstruction is
unchanged for 3 more primes; --rigorous uses enough primes to cover 2^n. Prints one JSON line.
"""
import os, json, time, math, argparse
from .satgen import read_dimacs, sat_einsum
from .tree_utils import complexity
from .ordering import optimize_order, prepare_tree, load_tree, save_tree
from . import count_engine as CE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--sc", type=float, default=32, help="max tensor rank (bytes = 2^sc) after slicing")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--save-tree", default=None)
    ap.add_argument("--rigorous", action="store_true")
    ap.add_argument("--ntrials", type=int, default=8)
    ap.add_argument("--niters", type=int, default=30)
    ap.add_argument("--gpus", default=None, help="spread the prime passes over GPUs: k or a list like 0,1,2")
    args = ap.parse_args()
    T0 = time.time()
    clauses, n = read_dimacs(args.cnf)
    ixs, iy, sizes = sat_einsum(clauses, n)
    if args.tree:
        (d, sliced), t_opt = load_tree(args.tree), 0.0
    else:
        d, sliced, t_opt = optimize_order(ixs, sizes, args.sc + 3, sc_slice=args.sc,
                                          ntrials=args.ntrials, niters=args.niters)
    if args.save_tree:
        save_tree(args.save_tree, d, sliced)
    dd, ixs_sl = prepare_tree(d, ixs, sliced)
    tc, sc = complexity(dd, ixs_sl)
    if args.gpus:
        from .multigpu import GpuPool, parse_gpus, count_sliced
        with GpuPool(parse_gpus(args.gpus)) as pool:
            cnt, minfo = count_sliced(pool, clauses, n, d, sliced, mode="rigorous" if args.rigorous else "stable")
        info = dict(primes=minfo["primes"], compile_s=0.0, run_s=minfo["wall"], per_pass_s=minfo["wall"] / minfo["primes"],
                    arena_gib=float("nan"), gpus=minfo["gpus"])
    else:
        cnt, info = CE.count_models(clauses, n, d, sliced=sliced, mode="rigorous" if args.rigorous else "stable")
    total = time.time() - T0
    l10 = math.log10(cnt) if cnt > 0 else float("-inf")
    print(json.dumps(dict(instance=os.path.basename(args.cnf), vars=n, clauses=len(clauses), tc=round(tc, 2), sc=sc,
                          slices=2 ** len(sliced), count=str(cnt), log10=round(l10, 4), primes=info["primes"],
                          opt_s=round(t_opt, 2), compile_s=round(info["compile_s"], 2), gpu_run_s=round(info["run_s"], 4),
                          per_pass_s=round(info["per_pass_s"], 5), arena_gib=round(info["arena_gib"], 3),
                          total_wall_s=round(total, 2), gpus=info.get("gpus", 1))))


if __name__ == "__main__":
    main()
