"""Branch-and-reduce SAT solving with tensor-network leaves (TensorBranching for CNF).

    from booltn.branching import BranchSolver
    res = BranchSolver(clauses, n, sc_leaf=32).solve()
    res.status          # "SAT" | "UNSAT" | "UNKNOWN";  res.stats: nodes, leaves, prunes per reducer, times
    res = BranchSolver(clauses, n, mode="count").solve()
    res.count           # exact model count (int); res.status "COUNT" (or "UNKNOWN" on a limit / error)

Root: unit propagation, XOR Gauss-Jordan, LP over detected cardinality constraints and a CDCL probe with a
large budget (booltn.reduce). Each can decide the instance. Otherwise one contraction tree is optimized
for the reduced clause network (omeco TreeSA, or greedy for very large networks).

Search (depth first). A node is a list of assumed literals:
  1. reduce   propagation and XOR implications to a fixpoint; then the LP and small-budget CDCL probes
              (CaDiCaL, then CryptoMiniSat with Gauss-Jordan if CaDiCaL is undecided).
              Any of them may prove the node UNSAT (prune) or find a model (SAT).
  2. measure  the node's network = clauses not yet satisfied, restricted to their free literals; its tree
              is the root tree with satisfied clauses pruned (no re-optimization), giving (tc, sc).
  3. leaf     if sc <= sc_leaf: decide the network on the GPU (booltn.engine.decide).
  4. branch   otherwise pick the variable with the best lookahead among the labels of the largest
              intermediate tensors: both values are propagated (+XOR) and scored by the children's
              tc and sc excess. A value whose propagation conflicts fixes the variable (failed literal)
              instead of branching.
Branching on a variable shrinks every tensor that carries it, as slicing does; unlike uniform slicing,
each branch is reduced again and may be pruned, so the slices are non-uniform.

Multi-GPU: with pool=GpuPool(...) the leaves are contracted asynchronously on all GPUs of the pool.

Counting (mode="count") uses the same search with count-preserving steps only: implied literals are fixed
(same value in every model), reducers may only prune (count 0), both children of a branch are summed,
and a leaf contributes its tensor-network count (booltn.count_engine) times 2^(free variables). When every
remaining clause belongs to a detected XOR, the node is counted in closed form, 2^(#XOR vars - rank)
times 2^(other free variables), without contraction.

Leaf cache (count mode, leaf_cache=path): every GPU leaf count is appended to a JSON-lines journal keyed by a
hash of the leaf's residual formula (the count of a formula does not depend on the path that produced it). A
run that stops (time limit, lost allocation) is resumed by running again with the same journal: the search
replays its internal nodes and takes every leaf already in the journal from there.
"""
import dataclasses
import hashlib
import json
import math
import os
import sys
import time

from .reduce import CDCL, XorSystem, LPRelaxation, detect_xors
from .slicer import _big_labels
from .tree_utils import recompute_outputs, complexity, prune_tree, log2sumexp2


@dataclasses.dataclass
class Result:
    status: str                 # "SAT" | "UNSAT" | "UNKNOWN"
    decided_by: str             # reducer or "gpu" / "cdcl-leaf" / "search"
    stats: dict
    model: list = None          # a satisfying assignment when a CDCL probe found one (DIMACS literals)
    count: int = None           # mode="count": exact number of models


def clean_clauses(clauses):
    """Drop tautologies and duplicate literals / clauses. Returns (clauses, has_empty_clause)."""
    out, seen, empty = [], set(), False
    for c in clauses:
        s = set(c)
        if any(-l in s for l in s):
            continue
        if not s:
            empty = True
            continue
        k = frozenset(s)
        if k not in seen:
            seen.add(k)
            out.append(sorted(s, key=abs))
    return out, empty


def check_model(clauses, model):
    true = set(l for l in model if l)
    return all(any(l in true for l in c) for c in clauses)


class BranchSolver:
    def __init__(self, clauses, n, sc_leaf=32, use_xor=True, use_lp=True, cdcl_root=100000, cdcl_node=2000,
                 lp_max_depth=None, candidates=12, order="auto", ntrials=4, niters=20, tree=None,
                 leaf="gpu", time_limit=3600.0, max_nodes=1000000, mode="decide", count_mode="stable", pool=None,
                 cms_root_time=60.0, cms_node_confl=2000, count_moduli=None, max_splits=16, mip_root_time=None,
                 leaf_cache=None, verbose=0):
        """pool: a booltn.multigpu.GpuPool; leaves then run asynchronously on its GPUs while the search
        continues (decide: the first SAT leaf ends the search; count: leaf counts are summed)."""
        assert mode in ("decide", "count")
        self.pool = pool
        self.cms_root_time, self.cms_node_confl = cms_root_time, cms_node_confl
        # HiGHS MIP over the LP rows at the root (decide: proves UNSAT or finds a model; count: only UNSAT helps)
        self.mip_root_time = (60.0 if mode == "decide" else 0.0) if mip_root_time is None else mip_root_time
        self.count_moduli = count_moduli    # CRT moduli for GPU leaves (default: 8-bit moduli, then 16-bit primes)
        self.max_splits = max_splits        # leaf splits along one path when a leaf count fails (e.g. the 16-bit
                                            # plan, twice the memory, does not fit; each split lowers sc)
        self.mode, self.count_mode = mode, count_mode
        self.orig, self.empty = clean_clauses(clauses)
        self.n = n
        self.sc_leaf = sc_leaf
        self.use_xor, self.use_lp = use_xor, use_lp
        self.cdcl_root, self.cdcl_node = cdcl_root, cdcl_node
        self.lp_max_depth = lp_max_depth
        self.K = candidates
        self.order, self.ntrials, self.niters, self.tree0 = order, ntrials, niters, tree
        self.leaf = leaf
        self.time_limit, self.max_nodes = time_limit, max_nodes
        self.verbose = verbose
        self.st = dict(nodes=0, leaves=0, branches=0, failed_literals=0, max_depth=0,
                       pruned=dict(prop=0, xor=0, lp=0, cdcl=0), xor_closed=0, root_sc=None, root_tc=None,
                       t_root=0.0, t_order=0.0, t_reduce=0.0, t_select=0.0, t_leaf=0.0, leaf_sc_max=0)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(self.orig) + 1000))
        self.cache, self.cache_path = {}, leaf_cache   # residual-formula hash -> model count over its variables
        if leaf_cache and mode == "count" and os.path.exists(leaf_cache):
            for line in open(leaf_cache):
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = int(r["count"])
                except (ValueError, KeyError):      # a line cut short by a killed run
                    pass
        self.st["leaves_cached"], self.st["cache_size"] = 0, len(self.cache)

    @staticmethod
    def _key(res):
        """Hash of a residual clause list, independent of clause and literal order."""
        c = sorted(tuple(sorted(cl)) for cl in res)
        return hashlib.sha256(repr(c).encode()).hexdigest()[:32]

    def _remember(self, key, count):
        if key is None or key in self.cache:
            return
        self.cache[key] = count
        if self.cache_path:
            with open(self.cache_path, "a") as f:     # one short line per leaf: an O_APPEND write
                f.write(json.dumps(dict(key=key, count=str(count))) + "\n")

    # ------------------------------------------------------------------------------------------
    def log(self, *a):
        if self.verbose:
            print(f"[{time.time() - self.t0:8.2f}s]", *a, flush=True)

    def _fixpoint(self, lits):
        """Propagation + XOR implications. Returns (None, lits) or ("UNSAT", reason)."""
        lits = list(lits)
        while True:
            ok, implied = self.cdcl.propagate(lits)
            if not ok:
                return "UNSAT", "prop"
            lits = list(dict.fromkeys(list(lits) + list(implied)))
            if self.xor is None:
                return None, lits
            ok, extra = self.xor.implied(lits)
            if not ok:
                return "UNSAT", "xor"
            have = set(lits)
            new = [l for l in extra if l not in have]
            if not new:
                return None, lits
            lits += new

    def _residual(self, lits):
        """(kept clause indices, residual clauses) of the root clause list under lits."""
        val = {}
        for l in lits:
            val[abs(l)] = l > 0
        keep, res = [], []
        for i, c in enumerate(self.clauses):
            rem, sat = [], False
            for l in c:
                b = val.get(abs(l))
                if b is None:
                    rem.append(l)
                elif b == (l > 0):
                    sat = True
                    break
            if not sat:
                keep.append(i)
                res.append(rem)
        return keep, res

    def _measure(self, lits):
        """Residual network, its tree (root tree pruned) and (tc, sc)."""
        keep, res = self._residual(lits)
        self._keep = keep
        if not res:
            return res, None, None, 0.0, 0
        ixs = [[abs(l) - 1 for l in c] for c in res]
        d = recompute_outputs(prune_tree(self.tree, keep), ixs, [])
        tc, sc = complexity(d, ixs)
        return res, d, ixs, tc, sc

    # ------------------------------------------------------------------------------------------
    def _unsat(self, why):
        if self.mode == "count":
            return Result("COUNT", why, self.st, count=0)
        return Result("UNSAT", why, self.st)

    def _root(self):
        """Root reductions + ordering. Returns a Result if decided, else None."""
        t = time.time()
        if self.empty:
            return self._unsat("empty-clause")
        self.cdcl = CDCL(self.orig, cms=bool(self.cms_root_time or self.cms_node_confl))
        ok, implied = self.cdcl.propagate([])
        if not ok:
            return self._unsat("prop")
        self.xor, self.xor_members = None, set()
        if self.use_xor:
            xs, self.xor_members = detect_xors(self.orig, members=True)
            self.st["n_xor"] = len(xs)
            if xs:
                self.xor = XorSystem(xs)
                if not self.xor.consistent:
                    return self._unsat("xor")
        st, lits = self._fixpoint(implied)
        if st:
            return self._unsat(lits)
        self.lp = None
        if self.use_lp:
            self.lp = LPRelaxation(self.orig, self.n)
            self.st["lp_rows"], self.st["lp_card"] = self.lp.n_rows, self.lp.n_card
            if not self.lp.feasible(lits):
                return self._unsat("lp")
        if self.cdcl_root:
            res, model = self.cdcl.probe(lits, self.cdcl_root, cms_time=self.cms_root_time)
            if res is True and self.mode == "decide":
                return Result("SAT", "cdcl", self.st, model)
            if res is False:
                return self._unsat("cdcl")
        if self.lp is not None and self.mip_root_time and self.lp.n_card > 0:   # pure CNF rows: CDCL is stronger
            tm = time.time()
            res, model = self.lp.mip(lits, time_limit=self.mip_root_time)
            self.st["t_mip"] = round(time.time() - tm, 3)
            if res is False:
                return self._unsat("mip")
            if res is True and self.mode == "decide":
                val = {abs(l): l > 0 for l in model}
                if all(any(val.get(abs(l)) == (l > 0) for l in c) for c in self.orig):
                    return Result("SAT", "mip", self.st, model)
        self.root_lits = lits
        # root network: clauses not satisfied by the root literals, restricted to free literals
        self.clauses = self.orig
        keep, res = self._residual(lits)
        self.clauses = res
        self.is_xor = [frozenset(self.orig[j]) in self.xor_members for j in keep]
        self.st["t_root"] = time.time() - t
        self.log(f"root: {len(self.orig)} clauses -> {len(res)} after {len(lits)} root literals; "
                 f"xor {self.st.get('n_xor', 0)}, lp rows {self.st.get('lp_rows', 0)}")
        if not res:
            if self.mode == "count":
                return Result("COUNT", "prop", self.st, count=1 << self._nfree(lits))
            return Result("SAT", "prop", self.st)
        if self.mode == "count" and self.xor is not None and all(self.is_xor):
            c = self._xor_count(lits)
            if c is not None:
                return Result("COUNT", "xor-closed-form", self.st, count=c)
        # contraction tree of the root network
        t = time.time()
        ixs = [[abs(l) - 1 for l in c] for c in res]
        if self.tree0 is not None:
            self.tree = self.tree0
        else:
            cands = []
            if self.order != "gpu":
                import omeco
                sizes = {l: 2 for ix in ixs for l in ix}
                # TreeSA up to 25000 tensors (domino 32x32, 6788 clauses: 3.4 s, sc 33 vs greedy sc 54)
                use_sa = self.order in ("treesa", "best") or (self.order == "auto" and len(ixs) <= 25000)
                opt = (omeco.TreeSA(ntrials=self.ntrials, niters=self.niters,
                                    score=omeco.ScoreFunction(sc_target=self.sc_leaf, sc_weight=4.0))
                       if use_sa else omeco.GreedyMethod())
                cands.append(omeco.optimize_code(ixs, [], sizes, opt).to_dict())
            if self.order in ("gpu", "best"):
                # GPU TreeSA (optional package treesa_gpu): on grid 3-SAT L30 a3.0 at sc target 34 it gives sc 50,
                # tc 2^65.7 vs omeco's sc 53, tc 2^73.4
                import treesa_gpu
                g, _ = treesa_gpu.optimize(ixs, [], nchains=1024, niters=50, sc_target=self.sc_leaf, sc_weight=4.0,
                                           mode="tc")
                cands.append(g)
            cands = [recompute_outputs(c, ixs, []) for c in cands]
            # lowest sc first: the leaves needed grow with sc - sc_leaf (grid L26 a2.5: 24 leaves from an sc-40
            # tree, more than 28 from an sc-42 tree with 16x lower tc); then lowest tc
            self.tree = min(cands, key=lambda c: complexity(c, ixs)[::-1])
        self.tree = recompute_outputs(self.tree, ixs, [])
        self.st["root_tc"], self.st["root_sc"] = complexity(self.tree, ixs)
        self.st["t_order"] = time.time() - t
        self.log(f"root tree: tc 2^{self.st['root_tc']:.1f} sc 2^{self.st['root_sc']} ({self.st['t_order']:.1f}s)")
        return None

    def _nfree(self, lits):
        """Number of unassigned variables among 1..n."""
        return self.n - len({abs(l) for l in lits})

    def _xor_count(self, lits):
        """Closed-form count when every remaining clause is an XOR clause: solutions of the XOR system
        (with the assigned XOR variables as units) times 2^(unassigned variables outside the system)."""
        ok, rk = self.xor.rank(lits)
        if not ok:
            return 0
        assigned = {abs(l) - 1 for l in lits}
        outside = self.n - len(self.xor.vars) - len(assigned - self.xor.vars)
        self.st["xor_closed"] += 1
        return 1 << (len(self.xor.vars) - rk + outside)

    def _leaf_count(self, res, d, lits, moduli=None):
        t = time.time()
        self.st["leaves"] += 1
        used = sorted({abs(l) for c in res for l in c})
        key = self._key(res) if self.cache_path else None
        if key in self.cache:
            self.st["leaves_cached"] += 1
            return self.cache[key] << (self._nfree(lits) - len(used))
        if self.leaf == "gpu":
            from .count_engine import count_models
            c, _ = count_models(res, len(used), d, mode=self.count_mode, primes=self.count_moduli)
        else:   # "cdcl": enumerate models over the used variables (tests only)
            from pysat.solvers import Minisat22
            c = 0
            with Minisat22(bootstrap_with=res) as s:
                while s.solve():
                    m = s.get_model()
                    c += 1
                    s.add_clause([-m[v - 1] for v in used])
        self._remember(key, c)
        self.st["t_leaf"] += time.time() - t
        return c << (self._nfree(lits) - len(used))

    def _submit_leaf(self, res, d, lits):
        """Asynchronous leaf on the pool -> appended to self.pending as (future, shift)."""
        from . import multigpu as MG
        self.st["leaves"] += 1
        if self.mode == "count":
            used = sorted({abs(l) for c in res for l in c})
            key = self._key(res) if self.cache_path else None
            if key in self.cache:
                import concurrent.futures as cf
                fut = cf.Future()
                fut.set_result(self.cache[key])
                self.st["leaves_cached"] += 1
            else:
                fut = self.pool.submit(MG._w_leaf_count, res, len(used), MG.flatten_tree(d), self.count_mode,
                                       self.count_moduli)
            self.pending.append((fut, self._nfree(lits) - len(used), lits, (self._depth, self._splits), key))
        else:
            self.pending.append((self.pool.submit(MG._w_leaf_decide, res, MG.flatten_tree(d)), 0, lits,
                                 (self._depth, self._splits), None))

    def _drain(self, block=False, keep=0):
        """Collect finished leaves (all if block, else until at most `keep` are pending).
        Returns ("SAT", None) when a decide leaf is SAT, ("ERR", msg) on a leaf error, ("LIMIT", None) when the
        time limit passes while blocked on a leaf, else (None, count)."""
        import concurrent.futures as cf
        t = time.time()
        got, out = 0, None
        while self.pending:
            fut, shift, lits, (depth, splits), key = self.pending[0]
            if not fut.done() and not block and len(self.pending) <= keep:
                break
            try:
                r = fut.result(timeout=max(1.0, self.time_limit - (time.time() - self.t0)))
            except cf.TimeoutError:
                out = ("LIMIT", None)
                break
            except (ValueError, MemoryError) as e:    # count beyond the moduli, or the leaf plan does not fit
                self.pending.pop(0)
                if self.mode == "count" and splits < self.max_splits:   # leaf count failed: split this leaf
                    self.st["leaf_splits"] += 1
                    self._stack.append((lits, depth, True, splits + 1))
                    continue
                out = ("ERR", f"{type(e).__name__}: {e}")
                break
            self.pending.pop(0)
            if self.mode == "count":
                self._remember(key, r)
                got += r << shift
            elif r:
                out = ("SAT", None)
                break
        self.st["t_leaf"] += time.time() - t
        return out if out else (None, got)

    def _leaf(self, res, d):
        t = time.time()
        self.st["leaves"] += 1
        if self.leaf == "gpu":
            from . import engine as E
            sat = E.decide(res, d, [], early_exit=False)[0]
        else:   # "cdcl": exact CDCL on the residual (for testing without an H100)
            from pysat.solvers import Solver
            with Solver(name="cadical195", bootstrap_with=res) as s:
                sat = s.solve()
        self.st["t_leaf"] += time.time() - t
        return sat

    def _select(self, lits, force=False):
        """Best branching variable by lookahead -> ("branch", [(lits0, tc0), (lits1, tc1)])
        or ("forced", lits') when a value fails, or ("unsat", reason) when both fail.
        force: branch even if the network already fits sc_leaf (a leaf count overflowed the CRT range)."""
        res, d, ixs, tc, sc = self._measure(lits)
        cands = _big_labels(d, ixs, self.sc_leaf if not force else max(0, sc - 3))[: self.K]
        best = None
        for v in cands:
            kids = []
            for lit in (v + 1, -(v + 1)):
                st, l2 = self._fixpoint(lits + [lit])
                if st:
                    self.st["failed_literals"] += 1
                    st2, l3 = self._fixpoint(lits + [-lit])
                    return ("unsat", l3) if st2 else ("forced", l3)
                _, _, _, tc2, sc2 = self._measure(l2)
                kids.append((l2, tc2, sc2))
            score = log2sumexp2([k[1] + max(0, k[2] - self.sc_leaf) for k in kids])
            if best is None or score < best[0]:
                best = (score, v, kids)
        return "branch", sorted(best[2], key=lambda k: k[1])

    # ------------------------------------------------------------------------------------------
    def solve(self):
        self.t0 = time.time()
        r = self._root()
        if r is not None:
            return self._finish(r)
        stack = self._stack = [(self.root_lits, 0, False, 0)]  # (literals, depth, force a branch, overflow splits)
        total = 0
        counting = self.mode == "count"
        self.pending = []
        self.st["leaf_splits"] = 0
        while stack or (self.pool is not None and self.pending):
            if not stack:              # only asynchronous leaves left: they may re-queue split nodes
                kind, val = self._drain(block=True)
                if kind == "LIMIT":
                    return self._finish(Result("UNKNOWN", "limit", self.st, count=None))
                if kind == "SAT":
                    return self._finish(Result("SAT", "gpu", self.st))
                if kind == "ERR":
                    self.st["error"] = val
                    return self._finish(Result("UNKNOWN", "leaf-error", self.st))
                total += val
                continue
            if self.pool is not None and self.pending:
                kind, val = self._drain(keep=2 * len(self.pool))
                if kind == "LIMIT":
                    return self._finish(Result("UNKNOWN", "limit", self.st, count=None))
                if kind == "SAT":
                    return self._finish(Result("SAT", "gpu", self.st))
                if kind == "ERR":
                    self.st["error"] = val
                    return self._finish(Result("UNKNOWN", "leaf-error", self.st))
                total += val
            if time.time() - self.t0 > self.time_limit or self.st["nodes"] >= self.max_nodes:
                return self._finish(Result("UNKNOWN", "limit", self.st, count=None))
            lits, depth, force, splits = stack.pop()
            self._depth, self._splits = depth, splits
            self.st["nodes"] += 1
            self.st["max_depth"] = max(self.st["max_depth"], depth)
            while True:     # node loop (repeats after a failed literal)
                t = time.time()
                st, out = self._fixpoint(lits)
                if st is None and self.lp is not None and (self.lp_max_depth is None or depth <= self.lp_max_depth):
                    if not self.lp.feasible(out):
                        st, out = "UNSAT", "lp"
                if st is None and self.cdcl_node and depth > 0:
                    res, model = self.cdcl.probe(out, self.cdcl_node, cms_confl=self.cms_node_confl)
                    if res is True and not counting:
                        self.st["t_reduce"] += time.time() - t
                        return self._finish(Result("SAT", "cdcl", self.st, model))
                    if res is False:
                        st, out = "UNSAT", "cdcl"
                self.st["t_reduce"] += time.time() - t
                if st:
                    self.st["pruned"][out] += 1
                    break
                lits = out
                res, d, ixs, tc, sc = self._measure(lits)
                if not res:
                    if not counting:
                        return self._finish(Result("SAT", "search", self.st))
                    total += 1 << self._nfree(lits)
                    break
                if counting and self.xor is not None and all(self.is_xor[i] for i in self._keep):
                    total += self._xor_count(lits)
                    break
                if sc <= self.sc_leaf and not force:
                    self.st["leaf_sc_max"] = max(self.st["leaf_sc_max"], sc)
                    if self.pool is not None:
                        self._submit_leaf(res, d, lits)
                        break
                    if counting:
                        try:
                            total += self._leaf_count(res, d, lits)
                            break
                        except (ValueError, MemoryError) as e:   # count beyond the moduli / plan too large: split
                            if splits >= self.max_splits:
                                self.st["error"] = f"{e}; still too large after {splits} splits"
                                return self._finish(Result("UNKNOWN", "leaf-error", self.st))
                            self.st["leaf_splits"] += 1
                            force, splits = True, splits + 1
                            self._splits = splits
                    else:
                        if self._leaf(res, d):
                            return self._finish(Result("SAT", self.leaf, self.st))
                        break
                t = time.time()
                kind, val = self._select(lits, force=force)
                self.st["t_select"] += time.time() - t
                if kind == "unsat":
                    self.st["pruned"][val if isinstance(val, str) else "prop"] += 1
                    break
                if kind == "forced":
                    lits = val
                    continue
                self.st["branches"] += 1
                self.log(f"depth {depth}: tc 2^{tc:.1f} sc 2^{sc} -> branch "
                         f"(children tc {', '.join(f'2^{k[1]:.1f}' for k in val)})")
                for l2, _, _ in reversed(val):    # cheaper child is explored first
                    stack.append((l2, depth + 1, False, splits))
                break
        if counting:
            return self._finish(Result("COUNT", "search", self.st, count=total))
        return self._finish(Result("UNSAT", "search", self.st))

    def _finish(self, r):
        for fut, _, _, _, key in getattr(self, "pending", []):   # stopped early: keep leaves that did finish
            if key is not None and fut.done() and not fut.cancelled() and fut.exception() is None:
                self._remember(key, fut.result())
        r.stats["time"] = time.time() - self.t0
        if hasattr(self, "cdcl"):
            r.stats["t_cdcl"] = self.cdcl.time
        if getattr(self, "xor", None) is not None:
            r.stats["t_xor"] = self.xor.time
        if getattr(self, "lp", None) is not None:
            r.stats["t_lp"] = self.lp.time + self.lp.setup_time
        if r.model is not None:
            assert check_model(self.orig, r.model), "CDCL model does not satisfy the formula"
        return r
