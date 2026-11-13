"""Contraction orders: omeco TreeSA + slicing, and tree JSON files.

A tree file is ``{"tree": <omeco dict>, "sliced": [labels]}`` (extra keys are ignored); labels are
0-based variable ids. ``trees/`` in the repository holds the trees used for the published numbers;
GPU TreeSA (the separate ``treesa_gpu`` package) writes the same format via ``runner_json``.
"""
import json
import time

from .tree_utils import recompute_outputs


def optimize_order(ixs, sizes, sc_target, sc_slice=None, ntrials=8, niters=30, sc_weight=4.0):
    """omeco TreeSA with score (sc_target, sc_weight), then omeco TreeSASlicer down to ``sc_slice``
    (log2 of the largest tensor) if the tree exceeds it. Returns (tree_dict, sliced_labels, seconds)."""
    import omeco
    score = omeco.ScoreFunction(sc_target=sc_target, sc_weight=sc_weight)
    t = time.time()
    tree = omeco.optimize_code(ixs, [], sizes, omeco.TreeSA(ntrials=ntrials, niters=niters, score=score))
    sliced, d = [], tree.to_dict()
    if sc_slice is not None and tree.complexity(ixs, sizes).sc > sc_slice:
        sl = omeco.slice_code(tree, ixs, sizes, omeco.TreeSASlicer(score=omeco.ScoreFunction(sc_target=sc_slice)))
        sliced, d = list(sl.slicing()), sl.tree().to_dict()
    return d, sliced, time.time() - t


def sliced_network(ixs, sliced):
    """Drop the sliced labels from every tensor's label list."""
    s = set(sliced)
    return [[l for l in ix if l not in s] for ix in ixs]


def prepare_tree(tree, ixs, sliced):
    """(tree with internal outputs recomputed on the sliced network, sliced ixs)."""
    ixs_sl = sliced_network(ixs, sliced)
    return recompute_outputs(tree, ixs_sl, []), ixs_sl


def load_tree(path):
    """Read a tree file -> (tree_dict, sliced_labels)."""
    with open(path) as f:
        J = json.load(f)
    return J["tree"], list(J.get("sliced", []))


def save_tree(path, tree, sliced, **meta):
    with open(path, "w") as f:
        json.dump(dict(tree=tree, sliced=list(sliced), **meta), f)
