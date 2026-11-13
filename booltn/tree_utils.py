"""Utilities for contraction trees in omeco's dict format.

Tree dict format (same as omeco ``NestedEinsum.to_dict()``):
  leaf: {"tensor_index": i}
  node: {"args": [left, right], "eins": {"ixs": [labels_left, labels_right], "iy": labels_out}}

All indices are assumed to be boolean (dimension 2), so log2 sizes are label counts.
"""
import math


def leaf_ids(d):
    if "tensor_index" in d:
        return [d["tensor_index"]]
    return leaf_ids(d["args"][0]) + leaf_ids(d["args"][1])


def recompute_outputs(d, ixs, iy):
    """Return a new tree whose internal-node output labels are derived from topology.

    A label is an output of a node iff it appears in a tensor outside the node's
    subtree, or in the final output ``iy``.
    """
    total = {}
    for ix in ixs:
        for l in ix:
            total[l] = total.get(l, 0) + 1
    final = set(iy)

    def rec(t):
        if "tensor_index" in t:
            cnt = {}
            for l in ixs[t["tensor_index"]]:
                cnt[l] = cnt.get(l, 0) + 1
            return {"tensor_index": t["tensor_index"]}, cnt, list(ixs[t["tensor_index"]])
        l, cl, la = rec(t["args"][0])
        r, cr, lb = rec(t["args"][1])
        cnt = dict(cl)
        for k, v in cr.items():
            cnt[k] = cnt.get(k, 0) + v
        seen, out = set(), []
        for x in la + lb:
            if x in seen:
                continue
            seen.add(x)
            if x in final or cnt[x] < total[x]:
                out.append(x)
        return {"args": [l, r], "eins": {"ixs": [la, lb], "iy": out}}, cnt, out

    return rec(d)[0]


def contractions(d, ixs):
    """List of (B, M, N, K) label counts for every pairwise contraction (post-order)."""
    acc = []

    def rec(t):
        if "tensor_index" in t:
            return set(ixs[t["tensor_index"]])
        a = rec(t["args"][0])
        b = rec(t["args"][1])
        y = set(t["eins"]["iy"])
        acc.append((len(a & b & y), len((a - b) & y), len((b - a) & y), len((a & b) - y)))
        return y

    rec(d)
    return acc


def complexity(d, ixs):
    """(tc, sc) in log2 units, omeco convention: tc = log2 sum 2^(B+M+N+K), sc = max tensor size."""
    cs = contractions(d, ixs)
    tc = math.log2(sum(2.0 ** (b + m + n + k) for b, m, n, k in cs)) if cs else 0.0
    sc = max([b + m + n for b, m, n, k in cs] + [len(set(ix)) for ix in ixs])
    return tc, sc


# ---------------------------------------------------------------------------
# GPU cost model (bit-packed bool tensors, Hopper b1 tensor cores)
# Times in seconds; label counts are log2 sizes in bits.
# ---------------------------------------------------------------------------
GPU_MODEL = dict(
    peak_ops=11.9e15,     # b1 wgmma AND-popc, ops/s (2 ops per AND-accumulate), measured on H100 PCIe
    bw_bits=1.6e13,       # HBM bandwidth in bits/s (~2.0 TB/s)
    m0=6, n0=6, k0=8,     # tensor-core tile granularity: M>=64 rows, N>=64, K>=256 bits
    launch=4e-6,          # per-contraction fixed overhead (s)
)


def gpu_time(B, M, N, K, model=GPU_MODEL):
    ops = 2.0 ** (1 + B + max(M, model["m0"]) + max(N, model["n0"]) + max(K, model["k0"]))
    bits = 2.0 ** (B + M + K) + 2.0 ** (B + N + K) + 2.0 ** (B + M + N)
    return max(ops / model["peak_ops"], bits / model["bw_bits"]) + model["launch"]


def tree_gpu_time(d, ixs, model=GPU_MODEL):
    return sum(gpu_time(*c, model=model) for c in contractions(d, ixs))


# ---------------------------------------------------------------------------
# GPU cost model v2: fitted to the booltn kernels on H100 PCIe (results/bench_kernels_model2.json)
#   per large contraction = launches + permutes of both operands + GEMM (best of cc / tc)
#   small contractions (out rank <= 14) run batched per tree level -> ~free
# ---------------------------------------------------------------------------
GPU_MODEL_V2 = dict(
    launch=4e-6,             # per kernel launch inside a CUDA graph (s)
    perm_bw_bits=8.0e12,     # permutation throughput, bits/s each way (~1 TB/s)
    cc_bit=6.25e-14,         # cc: s per bit moved (A+B+C)
    cc_cell=1.97e-14,        # cc: s per 2^(B+M+N+K) cell (~100 TOPS)
    tc_bit=2.78e-14,         # tc: s per bit moved
    tc_pad_cell=7.6e-17,     # tc: s per padded cell 2^(B+max(M,6)+max(N,6)+max(K,8))
    tc_out_bit=2.14e-13,     # tc: s per output bit (threshold + pack epilogue)
    small_rank=14,           # contractions with output rank <= this are batched (no permutes)
    small_cell=1e-12,        # s per (output bit x summed combo) in the batched small kernel
)


def gpu_time_v2(B, M, N, K, model=GPU_MODEL_V2, small=False):
    """Modeled seconds for one pairwise contraction with label counts (B, M, N, K)."""
    if small:
        return model["small_cell"] * 2.0 ** (B + M + N + K)
    a_bits, b_bits, c_bits = 2.0 ** (B + M + K), 2.0 ** (B + N + K), 2.0 ** (B + M + N)
    moved = a_bits + b_bits + c_bits
    perm = 2 * (a_bits + b_bits) / model["perm_bw_bits"]
    cc = model["cc_bit"] * moved + model["cc_cell"] * 2.0 ** (B + M + N + K)
    best = cc
    if K >= 7 and max(M, N) >= 5:
        tc = (model["tc_bit"] * moved
              + model["tc_pad_cell"] * 2.0 ** (B + max(M, 6) + max(N, 6) + max(K, 8))
              + model["tc_out_bit"] * c_bits)
        best = min(cc, tc)
    return 3 * model["launch"] + perm + best


def tree_gpu_time_v2(d, ixs, model=GPU_MODEL_V2):
    """Total modeled seconds for one contraction of the tree (one slice)."""
    total = 0.0
    for (B, M, N, K) in contractions(d, ixs):
        small = (B + M + N) <= model["small_rank"]
        total += gpu_time_v2(B, M, N, K, model, small=small)
    return total
