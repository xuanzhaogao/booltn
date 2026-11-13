"""Multi-GPU execution: one worker process per GPU, independent work items.

The engine's work splits into independent pieces: slices (OR / sum), primes of the #SAT counter (one
pass per prime) and branch-and-reduce leaves. Each worker process pins one GPU through
CUDA_VISIBLE_DEVICES before the CUDA library is loaded and caches its compiled plan, so a network is
compiled once per GPU.

    from booltn.multigpu import GpuPool, decide_sliced, count_sliced
    with GpuPool() as pool:                                   # all visible GPUs (or GpuPool([0, 1, 2]))
        sat, done = decide_sliced(pool, clauses, tree, sliced)
        count, info = count_sliced(pool, clauses, n, tree, sliced)

The same pool executes branch-and-reduce leaves: BranchSolver(..., pool=pool).
For testing on one GPU, GpuPool([0, 0]) runs two workers on the same device.
"""
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import time


def visible_gpus():
    """GPU ids this process may use (CUDA_VISIBLE_DEVICES, else nvidia-smi)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return []


def flatten_tree(tree):
    """Pickle-safe, non-recursive form of an omeco tree dict: post-order list of ("L", tensor_index) and
    ("N", left, right, iy) entries (children by list index). Deep trees overflow pickle's recursion limit."""
    out, idx, stack = [], {}, [(tree, False)]
    while stack:
        d, done = stack.pop()
        if "tensor_index" in d:
            idx[id(d)] = len(out)
            out.append(("L", d["tensor_index"]))
        elif done:
            idx[id(d)] = len(out)
            out.append(("N", idx[id(d["args"][0])], idx[id(d["args"][1])], list(d.get("eins", {}).get("iy", []))))
        else:
            stack += [(d, True), (d["args"][1], False), (d["args"][0], False)]
    return out


def unflatten_tree(flat):
    nodes = []
    for e in flat:
        if e[0] == "L":
            nodes.append({"tensor_index": e[1]})
        else:
            nodes.append({"args": [nodes[e[1]], nodes[e[2]]], "eins": {"ixs": [], "iy": e[3]}})
    return nodes[-1]


# ------------------------------------------------------------------------------------------------
# worker side
# ------------------------------------------------------------------------------------------------
_GPU = None
_STOP = None
_PLAN = {}          # key -> (kind, compiled plan, slicer); one cached plan per kind


def _init(gpu_queue, stop):
    global _GPU, _STOP
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))   # tree utilities recurse over deep trees
    _GPU = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU)
    _STOP = stop


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _sat_plan(key, clauses, tree, sliced, small_rc):
    from . import engine as E
    from .tree_utils import recompute_outputs
    old = _PLAN.get("sat")
    if old is not None and old[0] == key:
        return old[1], old[2]
    if old is not None:
        old[1].free()
    slicer = E.SatSlicer(clauses, list(sliced))
    d = recompute_outputs(unflatten_tree(tree), [list(l) for l in slicer.labels], [])
    ct = E.CompiledTree(d, slicer.labels, use_graph=True, small_rc=small_rc)
    _PLAN["sat"] = (key, ct, slicer)
    return ct, slicer


def _w_decide_slices(key, clauses, tree, sliced, slices, small_rc, early_exit):
    """OR over the given slices on this worker's GPU -> (sat, slices_done, gpu, seconds)."""
    t = time.time()
    ct, slicer = _sat_plan(key, clauses, tree, sliced, small_rc)
    ct.reset_flag()
    done = 0
    for i, s in enumerate(slices):
        ct.set_leaves(slicer.words(s))
        ct.run()
        done += 1
        if early_exit and i % 16 == 15 and (ct.flag() or _STOP.is_set()):
            break
    sat = bool(ct.flag())
    if sat and early_exit:
        _STOP.set()
    return sat, done, _GPU, time.time() - t


def _w_count_primes(key, clauses, tree, sliced, primes, small_rc):
    """Residues of the model count of the sliced network for the given primes -> ({p: r}, gpu, seconds)."""
    from . import count_engine as CE
    from .tree_utils import recompute_outputs
    t = time.time()
    old = _PLAN.get("count")
    if old is not None and old[0] == key:
        ct, slicer = old[1], old[2]
    else:
        if old is not None:
            old[1].free()
        slicer = CE.CountSlicer(clauses, list(sliced))
        d = recompute_outputs(unflatten_tree(tree), [list(l) for l in slicer.labels], [])
        ct = CE.CountTree(d, slicer.labels, small_rc=small_rc)
        _PLAN["count"] = (key, ct, slicer)
    nsl = 1 << len(sliced)
    out = {}
    for p in primes:
        ct.set_prime(p)
        for s in range(nsl):
            ct.set_leaves(slicer.leaves(s))
            ct.run()
        out[p] = ct.residue()
    return out, _GPU, time.time() - t


def _w_leaf_decide(res, flat):
    from . import engine as E
    return E.decide(res, unflatten_tree(flat), [], early_exit=False)[0]


def _w_leaf_count(res, nused, flat, mode, moduli=None):
    from .count_engine import count_models
    return count_models(res, nused, unflatten_tree(flat), mode=mode, primes=moduli)[0]


def parse_gpus(spec):
    """--gpus value: an int k (first k visible GPUs) or a comma list of ids (ids may repeat)."""
    if spec is None:
        return None
    s = str(spec)
    if "," in s:
        return [g for g in s.split(",") if g != ""]
    vis = visible_gpus()
    k = int(s)
    if k > len(vis):
        raise ValueError(f"--gpus {k}: only {len(vis)} GPUs visible ({vis})")
    return vis[:k]


# ------------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------------
class GpuPool:
    """Process pool with one worker per entry of `gpus` (ids may repeat to oversubscribe a GPU)."""

    def __init__(self, gpus=None):
        self.gpus = [str(g) for g in (gpus if gpus is not None else visible_gpus())]
        if not self.gpus:
            raise RuntimeError("no GPUs visible")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for g in self.gpus:
            q.put(g)
        self.stop = ctx.Event()
        self.ex = cf.ProcessPoolExecutor(max_workers=len(self.gpus), mp_context=ctx,
                                         initializer=_init, initargs=(q, self.stop))

    def __len__(self):
        return len(self.gpus)

    def submit(self, fn, *args):
        return self.ex.submit(fn, *args)

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def decide_sliced(pool, clauses, tree, sliced, small_rc=14, early_exit=True, max_slices=None):
    """OR over all 2^|sliced| slices, distributed round-robin over the pool's GPUs.
    Returns (sat, slices_done, info) with per-GPU timings."""
    sliced = list(sliced)
    nsl = 1 << len(sliced)
    todo = nsl if max_slices is None else min(nsl, max_slices)
    G = len(pool)
    tree = flatten_tree(tree)
    key = _key("sat", clauses, tree, sliced, small_rc)
    pool.stop.clear()
    t = time.time()
    futs = [pool.submit(_w_decide_slices, key, clauses, tree, sliced, list(range(i, todo, G)), small_rc, early_exit)
            for i in range(G)]
    rs = [f.result() for f in futs]
    sat = any(r[0] for r in rs)
    return sat, sum(r[1] for r in rs), dict(wall=time.time() - t, per_gpu=[(r[2], r[1], round(r[3], 3)) for r in rs])


def count_sliced(pool, clauses, n, tree, sliced=(), small_rc=12, mode="stable", confirm=3):
    """Exact model count; the prime passes are spread over the pool's GPUs (one prime per GPU per round).
    mode="stable": stop when the CRT value is unchanged for `confirm` extra primes; "rigorous": enough
    primes to exceed 2^(#used variables)."""
    from .count_engine import moduli_desc, crt
    sliced = list(sliced)
    used = sorted({abs(l) - 1 for c in clauses for l in c})
    G = len(pool)
    tree = flatten_tree(tree)
    key = _key("count", clauses, tree, sliced, small_rc)
    allp = moduli_desc()
    if mode == "rigorous":
        need, bits, sel = len(used) + 1, 0.0, []
        for p in allp:
            if bits > need:
                break
            sel.append(p); bits += math.log2(p)
        if bits <= need:
            raise ValueError(f"need > {need} bits but 8-bit primes give only {bits:.0f}")
        allp = sel
    t = time.time()
    residues, used_p, last, stable, i = {}, [], None, 0, 0
    while i < len(allp):
        batch = allp[i:i + G]
        i += len(batch)
        futs = [pool.submit(_w_count_primes, key, clauses, tree, sliced, [p], small_rc) for p in batch]
        for f in futs:
            r, _, _ = f.result()
            residues.update(r)
        for p in batch:                      # stability check in prime order, as the 1-GPU counter does
            used_p.append(p)
            if mode != "rigorous":
                val, _ = crt([residues[q] for q in used_p], used_p)
                stable = stable + 1 if val == last else 0
                last = val
        if mode != "rigorous" and stable >= confirm:
            break
    else:
        if mode != "rigorous" and stable < confirm:
            raise ValueError("8-bit moduli exhausted before the CRT reconstruction stabilized (count > ~10^109)")
    count, _ = crt([residues[q] for q in used_p], used_p)
    count <<= n - len(used)
    return count, dict(primes=len(used_p), gpus=G, wall=time.time() - t)
