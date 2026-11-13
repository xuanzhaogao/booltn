"""Greedy slicer for dict-format contraction trees (any optimizer), boolean labels.

Repeatedly slices the label that minimizes  2^(#sliced) * cost(tree with sliced labels removed)
among labels of tensors that exceed the space target, until max tensor rank <= sc_target.
cost = "tc" (2^tc per slice) or "h100" (modeled H100 time per slice, tree_h100_time; old name "gpu").
"""
from .tree_utils import recompute_outputs, contractions, complexity, tree_h100_time


def _cost(d, ixs, kind):
    if kind == "tc":
        cs = contractions(d, ixs)
        return sum(2.0 ** sum(c) for c in cs)
    return tree_h100_time(d, ixs)


def _big_labels(d, ixs, sc_target):
    """Labels occurring in intermediate tensors larger than the target (weighted by size excess)."""
    w = {}

    def rec(t):
        if "tensor_index" in t:
            return
        rec(t["args"][0]); rec(t["args"][1])
        y = t["eins"]["iy"]
        if len(y) > sc_target:
            for l in y:
                w[l] = w.get(l, 0) + 2.0 ** (len(y) - sc_target)
    rec(d)
    return sorted(w, key=lambda l: -w[l])


def greedy_slice(tree, ixs, sc_target, kind="h100", max_candidates=24, verbose=False):
    sliced = []
    while True:
        ixs_sl = [[l for l in ix if l not in set(sliced)] for ix in ixs]
        d = recompute_outputs(tree, ixs_sl, [])
        tc, sc = complexity(d, ixs_sl)
        if sc <= sc_target:
            return sliced, d
        best = None
        for l in _big_labels(d, ixs_sl, sc_target)[:max_candidates]:
            s2 = sliced + [l]
            ixs2 = [[x for x in ix if x not in set(s2)] for ix in ixs]
            d2 = recompute_outputs(tree, ixs2, [])
            _, sc2 = complexity(d2, ixs2)
            c = (2 ** len(s2)) * _cost(d2, ixs2, kind) * (2.0 ** max(0, sc2 - sc_target))
            if best is None or c < best[0]:
                best = (c, l)
        sliced.append(best[1])
        if verbose:
            print(f"  slice label {best[1]}: total cost {best[0]:.3e}")
