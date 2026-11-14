"""Distributed contraction on the GPUs (booltn.distributed.DistPlan) against the single-GPU engine.

  python test_distributed.py CASE.json ... [--p 1] [--devices 0,1] [--force] [--max-slices 16] [--no-graph]

Needs booltn/lib/libbooltn_mg.so (make -C kernels mg). Each case file (clauses, n, tree, sliced, truth;
written by make_distributed_case.py, CPU only) is planned for 2^p GPUs with local cap = sc - p, so that
the largest tensors only fit when they are split. Every global slice (at most --max-slices) is contracted
by the distributed plan and by engine.CompiledTree on one GPU; both must agree, and the OR over all slices
must equal the stored ground truth when all slices ran. --force uses a cost model that makes exchanges
attractive on small networks (to exercise them). Ranks may share devices: --p 3 --devices 0,1,1,0,1,0,0,1
runs 8 ranks on 2 GPUs with every exchange between the two devices.
"""
import sys, json, time, argparse
sys.setrecursionlimit(200000)
import numpy as np
from booltn import engine as E, distributed as D
from booltn.ordering import prepare_tree
from booltn.satgen import sat_einsum
from booltn.tree_utils import H100_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--p", type=int, default=1)
ap.add_argument("--devices", default=None)
ap.add_argument("--force", action="store_true")
ap.add_argument("--max-slices", type=int, default=16)
ap.add_argument("--no-graph", action="store_true", help="launch kernels directly instead of replaying CUDA graphs")
a = ap.parse_args()
devices = [int(x) for x in a.devices.split(",")] if a.devices else None
kw = dict(link=dict(bw=1e30, latency=0.0), model=dict(H100_MODEL, small_rank=0, launch=0.0)) if a.force else {}

ok = True
for f in a.cases:
    J = json.load(open(f))
    cl, n, sliced = J["clauses"], J["n"], J["sliced"]
    ixs, _, _ = sat_einsum(cl, n)
    d, ixs_sl = prepare_tree(J["tree"], ixs, sliced)
    cap = int(J["meta"]["sc"]) - a.p
    sch = D.plan_distribution(d, ixs_sl, a.p, cap, **kw)
    if sch is None:
        print(f"{f}: no feasible schedule for p={a.p} cap={cap} (skipped)")
        continue
    dp = D.DistPlan(cl, n, sliced, sch, devices=devices, use_graph=not a.no_graph)
    slicer = E.SatSlicer(cl, sliced)
    ct = E.CompiledTree(d, slicer.labels, use_graph=not a.no_graph)
    nsl = 1 << len(sliced)
    todo = min(nsl, a.max_slices)
    good, anysat = True, False
    for s in range(todo):
        ct.reset_flag(); ct.set_leaves(slicer.words(s)); ct.run(); ref = ct.flag()
        dp.reset_flags()
        dp.run_slice(s)
        got = any(dp.flags())
        good &= got == ref
        anysat |= got
    if todo == nsl and J.get("truth") is not None:
        good &= anysat == J["truth"]
    ok &= good
    nx = len(sch.exchanges)
    print(f"{'PASS' if good else 'FAIL'} {f}: p={a.p} cap={cap} devices={dp.devices} {todo}/{nsl} slices, "
          f"{nx} exchanges ({sum(x['newtop'] is None for x in sch.exchanges)} merging partial sums), "
          f"max exchanged rank {max([x['rank'] for x in sch.exchanges], default=0)}, {dp.nsegs} segments, "
          f"arena {dp.prog.arena_bytes / 2**20:.1f} MiB/GPU, peer {dp.direct}", flush=True)
    dp.free(); ct.free()
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
