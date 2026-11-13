"""Decide satisfiability by boolean tensor-network contraction on the GPU (CLI: ``booltn-sat``).

  python -m booltn.run_sat --cnf f.cnf --expect none --sc 36            # any DIMACS file
  python -m booltn.run_sat --n 60 --alpha 4.26 --seed 3 --sc 32          # random 3-SAT
  python -m booltn.run_sat --grid 24 --alpha 3 --seed 1 --tree trees/grid_L24_a3_s1_sc33.json --timing

Steps: instance -> contraction order (omeco TreeSA, or --tree) -> slicing to the --sc target -> compile the
tree once (CompiledTree: CUDA graph + arena) -> replay it for every slice, OR-accumulating the answer on the
device. Without --expect the answer is checked against minisat. Reports bool-ops/s and, with --timing, a
per-kernel breakdown.
"""
import time, math, argparse, collections
from .satgen import random_ksat, sat_einsum, read_dimacs, grid_ksat
from .tree_utils import contractions, complexity
from .ordering import optimize_order, prepare_tree, load_tree, save_tree
from . import engine as E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=4.26)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cnf", type=str, default=None)
    ap.add_argument("--grid", type=int, default=None, help="local 3-SAT on an LxL grid (alpha clauses/var)")
    ap.add_argument("--expect", choices=["sat", "unsat", "none"], default=None,
                    help="known answer (skips the minisat ground-truth run, e.g. for CDCL-hard instances)")
    ap.add_argument("--sc", type=float, default=30, help="space target (log2 bits) after slicing")
    ap.add_argument("--sc-opt", type=float, default=None, help="sc target used during TreeSA (default sc+6)")
    ap.add_argument("--max-slices", type=int, default=None)
    ap.add_argument("--tree", type=str, default=None, help="load tree JSON {tree, sliced} instead of optimizing")
    ap.add_argument("--save-tree", type=str, default=None)
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--timing", action="store_true", help="per-contraction event timing (adds sync overhead)")
    ap.add_argument("--eager", action="store_true", help="use the eager (python per-contraction) path instead of a compiled plan")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--concurrent", type=int, default=1, help="run this many slices concurrently (one plan+arena per stream)")
    ap.add_argument("--small-rc", type=int, default=14, help="max output rank for batched small contractions (0 = off)")
    ap.add_argument("--gpus", default=None, help="spread the slices over GPUs: k or a list like 0,1,2")
    args = ap.parse_args()

    if args.cnf:
        clauses, n = read_dimacs(args.cnf)
    elif args.grid:
        clauses, n = grid_ksat(args.grid, args.alpha, args.k, 1, args.seed)
    else:
        n = args.n
        clauses = random_ksat(n, round(args.alpha * n), args.k, args.seed)
    ixs, iy, sizes = sat_einsum(clauses, n)

    if args.expect == "none":
        truth = None
        print(f"instance: n={n} m={len(clauses)}; no ground truth")
    elif args.expect:
        truth = args.expect == "sat"
        print(f"instance: n={n} m={len(clauses)}; expected {'SAT' if truth else 'UNSAT'} (by construction)")
    else:
        from pysat.solvers import Minisat22
        t = time.time()
        with Minisat22(bootstrap_with=clauses) as s:
            truth = s.solve()
        print(f"instance: n={n} m={len(clauses)} k={args.k}; minisat says {'SAT' if truth else 'UNSAT'} ({time.time()-t:.3f}s)")

    if args.tree:
        (d, sliced), t_opt = load_tree(args.tree), 0.0
    else:
        d, sliced, t_opt = optimize_order(ixs, sizes, args.sc_opt or args.sc + 6, sc_slice=args.sc)
    if args.save_tree:
        save_tree(args.save_tree, d, sliced)
    d, ixs_sl = prepare_tree(d, ixs, sliced)
    cs = contractions(d, ixs_sl)
    ops_slice = sum(2.0 ** (1 + sum(c)) for c in cs)
    tc, sc = complexity(d, ixs_sl)
    print(f"order: opt {t_opt:.1f}s; per slice tc=2^{tc:.2f} sc=2^{sc:.0f} ops/slice=2^{math.log2(ops_slice):.2f}; "
          f"{len(sliced)} sliced vars -> {2**len(sliced)} slices")

    nsl = 2 ** len(sliced)
    max_slices = args.max_slices or nsl
    if args.gpus:
        from .multigpu import GpuPool, parse_gpus, decide_sliced
        with GpuPool(parse_gpus(args.gpus)) as pool:
            res, done, info = decide_sliced(pool, clauses, d, sliced, small_rc=args.small_rc,
                                            early_exit=not args.no_early_exit, max_slices=max_slices)
        full = done == nsl or res
        print(f"GPU result: {'SAT' if res else 'UNSAT'}" + ("" if full else f" (partial: {done}/{nsl} slices)") +
              (f" [{'OK' if res == truth else 'MISMATCH'}]" if full and truth is not None else ""))
        print(f"time {info['wall']:.3f}s for {done} slices on {len(info['per_gpu'])} GPU workers (incl. compile); "
              f"bool-ops {ops_slice*done:.3e} -> {ops_slice*done/info['wall']/1e12:.1f} TOPS; per GPU (id, slices, s): {info['per_gpu']}")
        return
    E.lib()
    E.STATS.reset()
    if not args.eager:
        run_plan(clauses, d, sliced, ops_slice, truth, max_slices, nsl, args)
        return
    # eager path (one Python-driven contraction at a time; for debugging)
    if max_slices < nsl:
        # benchmark mode: contract the first max_slices slices
        sl_vars = sliced
        t0 = time.time()
        res = False
        for s in range(max_slices):
            fixed = {v: (s >> i) & 1 for i, v in enumerate(sl_vars)}
            leaves = [E.BTensor.from_words(w, l) for (l, w) in (E.clause_tensor_words(c, fixed) for c in clauses)]
            root = E.contract_tree(d, leaves, timing=args.timing)
            res = res or bool(root.to_bool().reshape(-1)[0])
            root.free()
        E.lib().bt_sync()
        dt = time.time() - t0
        done = max_slices
    else:
        res, done, dt = E.solve_sat(clauses, n, d, sliced, timing=args.timing, early_exit=not args.no_early_exit)
    print(f"GPU result: {'SAT' if res else 'UNSAT'} (partial over {done}/{nsl} slices)" if done < nsl and not res
          else f"GPU result: {'SAT' if res else 'UNSAT'}  [{'OK' if res == truth else 'MISMATCH'}]")
    print(f"time {dt:.3f}s for {done} slices -> {dt/done*1e3:.2f} ms/slice; "
          f"bool-ops {ops_slice*done:.3e} -> {ops_slice*done/dt/1e12:.1f} TOPS (end-to-end incl. host)")
    if args.timing and E.STATS.records:
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for kind, b, m, nn, k, tp, tg in E.STATS.records:
            a = agg[kind]
            a[0] += 1; a[1] += tp; a[2] += tg; a[3] += 2.0 ** (1 + b + m + nn + k)
        tot = sum(a[1] + a[2] for a in agg.values())
        print(f"GPU kernel time {tot:.1f} ms total (perm+gemm), host/launch overhead = {dt*1e3 - tot:.1f} ms")
        for kind, a in sorted(agg.items()):
            print(f"  {kind:4s}: {a[0]:6d} contractions, perm {a[1]:9.2f} ms, gemm {a[2]:9.2f} ms, "
                  f"ops {a[3]:.3e} -> {a[3]/max(a[2],1e-9)/1e9:.1f} TOPS (gemm only)")
        recs = sorted(E.STATS.records, key=lambda r: -(r[5] + r[6]))[:12]
        print("  slowest contractions (kind B M N K perm_ms gemm_ms TOPS):")
        for kind, b, m, nn, k, tp, tg in recs:
            print(f"    {kind:4s} B={b:2d} M={m:2d} N={nn:2d} K={k:2d}  perm {tp:8.3f}  gemm {tg:8.3f}  "
                  f"{2.0**(1+b+m+nn+k)/((tp+tg)*1e-3)/1e12:8.1f} TOPS")


def run_plan_concurrent(clauses, d, sliced, ops_slice, truth, max_slices, nsl, args):
    """k independent compiled plans (own arenas), each replayed on its own stream; slices round-robin."""
    k = args.concurrent
    slicer = E.SatSlicer(clauses, sliced)
    L = E.lib()
    E._setup_plan_api(L)
    L.bt_make_streams(k)
    cts = []
    for i in range(k):
        ct = E.CompiledTree(d, slicer.labels, use_graph=True, small_rc=args.small_rc)
        ct.stream = i
        ct.set_leaves(slicer.words(0)); ct.run()
        cts.append(ct)
    L.bt_sync_all()
    for ct in cts:
        ct.reset_flag()
    L.bt_sync_all()
    print(f"concurrent plans: {k} x arena {cts[0].arena_bytes/2**30:.3f} GiB")
    t0 = time.time()
    done = 0
    for s in range(max_slices):
        ct = cts[s % k]
        ct.set_leaves(slicer.words(s))
        ct.run()
        done += 1
    L.bt_sync_all()
    dt = time.time() - t0
    res = any(ct.flag() for ct in cts)
    full = done == nsl
    print(f"GPU result: {'SAT' if res else 'UNSAT'}" + ("" if full else f" (partial: {done}/{nsl} slices)") +
          (f" [{'OK' if res == truth else 'MISMATCH'}]" if full else ""))
    print(f"time {dt:.3f}s for {done} slices -> {dt/done*1e3:.3f} ms/slice; bool-ops {ops_slice*done:.3e} -> "
          f"{ops_slice*done/dt/1e12:.1f} TOPS end-to-end (concurrency {k})")
    for ct in cts:
        ct.free()


def run_plan(clauses, d, sliced, ops_slice, truth, max_slices, nsl, args):
    if args.concurrent > 1:
        return run_plan_concurrent(clauses, d, sliced, ops_slice, truth, max_slices, nsl, args)
    slicer = E.SatSlicer(clauses, sliced)
    t = time.time()
    ct = E.CompiledTree(d, slicer.labels, use_graph=not args.no_graph, small_rc=args.small_rc)
    t_compile = time.time() - t
    print(f"compiled plan: {ct.nops} kernel launches ({ct.nsmall} small contractions in {len(ct.small_levels)} batched levels), "
          f"arena {ct.arena_bytes/2**30:.3f} GiB, compile {t_compile:.2f}s")
    # warm-up (graph capture) on slice 0
    ct.set_leaves(slicer.words(0)); ct.run(); E.lib().bt_sync()
    if args.timing:
        recs = ct.time_ops()
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for kind, b, m, n, k, ms in recs:
            a = agg[kind]; a[0] += 1; a[1] += ms; a[2] += 2.0 ** (1 + b + m + n + k)
        tot = sum(a[1] for a in agg.values())
        print(f"per-op timing of one slice: {tot:.2f} ms total GPU time (incl. permutes)")
        for kind, a in sorted(agg.items()):
            print(f"  {kind:4s}: {a[0]:5d} contractions  {a[1]:8.2f} ms  {a[2]:.3e} ops -> {a[2]/max(a[1],1e-9)/1e9:8.1f} TOPS")
        print("  heaviest contractions (kind B M N K ms TOPS):")
        for kind, b, m, n, k, ms in sorted(recs, key=lambda r: -r[5])[:10]:
            print(f"    {kind:4s} B={b:2d} M={m:2d} N={n:2d} K={k:2d} {ms:8.3f} ms {2.0**(1+b+m+n+k)/(ms*1e-3)/1e12:8.1f} TOPS")
    ct.reset_flag()
    t0 = time.time()
    res, done = False, 0
    early = not args.no_early_exit
    for s in range(max_slices):
        ct.set_leaves(slicer.words(s))
        ct.run()
        done += 1
        if early and s % 16 == 15 and ct.flag():
            break
    res = ct.flag()
    E.lib().bt_sync()
    dt = time.time() - t0
    full = done == nsl or res
    status = f"{'SAT' if res else 'UNSAT'}" + ("" if full else f" (partial: {done}/{nsl} slices)")
    check = (f"[{'OK' if res == truth else 'MISMATCH'}]" if truth is not None else "") if full else ""
    print(f"GPU result: {status} {check}")
    print(f"time {dt:.3f}s for {done} slices -> {dt/done*1e3:.3f} ms/slice; bool-ops {ops_slice*done:.3e} -> "
          f"{ops_slice*done/dt/1e12:.1f} TOPS end-to-end")
    ct.free()


if __name__ == "__main__":
    main()
