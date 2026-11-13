"""Branch-and-reduce SAT solver with GPU tensor-network leaves (CLI: ``booltn-branch``).

  python -m booltn.run_branch --cnf f.cnf [--sc-leaf 32] [--no-xor] [--no-lp] [--cdcl-root 100000] [-v]
  python -m booltn.run_branch --cnf f.cnf --count --gpus 8          # exact #SAT, leaves on 8 GPUs

Root reductions (propagation, XOR Gauss-Jordan, cardinality LP, budgeted CDCL), then depth-first branching
on the variables of the largest tensors until each branch's network fits sc_leaf, deciding the leaves on
the GPU. Prints one JSON line (status, which component decided it, statistics).
"""
import argparse, json, math, os, time
from .satgen import read_dimacs
from .branching import BranchSolver


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--sc-leaf", type=int, default=32, help="largest tensor (log2 bits) contracted at a leaf")
    ap.add_argument("--no-xor", action="store_true")
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--lp-max-depth", type=int, default=None, help="run the LP only down to this depth")
    ap.add_argument("--cdcl-root", type=int, default=None, help="conflict budget at the root (0 = off; default 100000, "
                    "2000 with --count where a probe can only prove UNSAT)")
    ap.add_argument("--cdcl-node", type=int, default=2000, help="conflict budget per node (0 = off)")
    ap.add_argument("--candidates", type=int, default=12, help="branching candidates evaluated by lookahead")
    ap.add_argument("--cms-root-time", type=float, default=60.0, help="CryptoMiniSat (Gauss) probe at the root, seconds (0 = off)")
    ap.add_argument("--mip-root-time", type=float, default=None,
                    help="HiGHS MIP probe over the LP rows at the root, seconds (0 = off; default 60 decide, 0 count)")
    ap.add_argument("--cms-node", type=int, default=2000, help="CryptoMiniSat probe per node, conflicts (0 = off)")
    ap.add_argument("--order", choices=["auto", "treesa", "greedy", "gpu", "best"], default="auto",
                    help="root contraction order: omeco (auto/treesa/greedy), GPU TreeSA (gpu, needs treesa_gpu), "
                         "or both, keeping the lower tc (best)")
    ap.add_argument("--tree", default=None, help="contraction tree of the root network (JSON {tree})")
    ap.add_argument("--leaf", choices=["gpu", "cdcl"], default="gpu")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--count", action="store_true", help="exact model counting (#SAT) instead of decision")
    ap.add_argument("--gpus", default=None, help="leaves on several GPUs: k (first k visible) or a list like 0,1,2")
    ap.add_argument("--leaf-cache", default=None, help="--count: JSON-lines journal of leaf counts; a run that stopped "
                    "resumes from it when started again with the same file")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    t = time.time()
    cl, n = read_dimacs(a.cnf)
    tree = json.load(open(a.tree))["tree"] if a.tree else None
    pool = None
    if a.gpus:
        from .multigpu import GpuPool, parse_gpus
        pool = GpuPool(parse_gpus(a.gpus))
    if a.cdcl_root is None:
        a.cdcl_root = 2000 if a.count else 100000
    try:
        r = BranchSolver(cl, n, sc_leaf=a.sc_leaf, use_xor=not a.no_xor, use_lp=not a.no_lp, cdcl_root=a.cdcl_root,
                         cdcl_node=a.cdcl_node, lp_max_depth=a.lp_max_depth, candidates=a.candidates, order=a.order,
                         tree=tree, leaf=a.leaf, time_limit=a.time_limit, mode="count" if a.count else "decide",
                         pool=pool, cms_root_time=a.cms_root_time, cms_node_confl=a.cms_node,
                         mip_root_time=a.mip_root_time, leaf_cache=a.leaf_cache, verbose=a.verbose).solve()
    finally:
        if pool is not None:
            pool.close()
    st = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.stats.items()}
    extra = {} if r.count is None else dict(count=str(r.count), log10=round(math.log10(r.count), 4) if r.count else None)
    print(json.dumps(dict(instance=os.path.basename(a.cnf), status=r.status, decided_by=r.decided_by, **extra,
                          total_s=round(time.time() - t, 3), gpus=len(pool) if pool else 1, **st)), flush=True)


if __name__ == "__main__":
    main()
