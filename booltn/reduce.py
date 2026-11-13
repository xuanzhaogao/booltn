"""Reductions for branch-and-reduce SAT solving (used by booltn.branching).

Every reducer sees the original CNF plus the literals assumed at a search node (DIMACS literals) and
either decides the node or returns implied literals:

  CDCL         unit propagation and budgeted CDCL on one incremental solver (pysat CaDiCaL), with the
               node's literals as assumptions. Learned clauses are shared by all nodes. Optionally a
               second probe with CryptoMiniSat (Gauss-Jordan on recovered XORs inside CDCL), which
               solves XOR-heavy instances that plain CDCL and separate Gauss cannot.
  XorSystem    XOR constraints recovered from the CNF + Gauss-Jordan elimination over GF(2): detects
               parity conflicts (Tseitin, Urquhart) and implied units.
  LPRelaxation LP over the clauses plus syntactically detected cardinality constraints
               (booltn.card_detect), solved with HiGHS: an infeasible LP proves the node UNSAT
               (mutilated chessboard, pigeonhole, sgen).

All of them are sound for decision SAT. Propagation and pruning of UNSAT nodes also preserve model
counts; nothing here substitutes variables.
"""
import collections
import time


# ------------------------------------------------------------------------------------------------
# CDCL: propagation + budgeted search
# ------------------------------------------------------------------------------------------------
class CDCL:
    """One incremental pysat solver for unit propagation and budgeted probes under assumptions."""

    def __init__(self, clauses, name="cadical195", cms=False):
        from pysat.solvers import Solver
        self.s = Solver(name=name, bootstrap_with=clauses)
        self.time = 0.0
        self.cms = None
        if cms:
            import pycryptosat
            self.cms = pycryptosat.Solver(threads=1)
            self.cms.add_clauses([c for c in clauses if c])

    def propagate(self, lits):
        """Unit propagation under `lits`. Returns (ok, implied literals incl. lits); ok=False on conflict."""
        t = time.time()
        ok, implied = self.s.propagate(assumptions=list(lits))
        self.time += time.time() - t
        return ok, implied

    def probe(self, lits, budget, cms_confl=None, cms_time=None):
        """Budgeted CDCL under `lits` -> (True, model) | (False, None) | (None, None). If CaDiCaL is
        undecided and a CryptoMiniSat probe is enabled, CMS gets `cms_confl` conflicts / `cms_time` s."""
        t = time.time()
        if budget is None:
            res = self.s.solve(assumptions=list(lits))
        else:
            self.s.conf_budget(int(budget))
            res = self.s.solve_limited(assumptions=list(lits))
        model = self.s.get_model() if res else None
        self.last_by = "cadical" if res is not None else None    # which solver decided the last probe
        if res is None and self.cms is not None and (cms_confl or cms_time):
            kw = {}
            if cms_confl:
                kw["confl_limit"] = int(cms_confl)
            if cms_time:
                kw["time_limit"] = float(cms_time)
            sat, sol = self.cms.solve(assumptions=list(lits), **kw)
            if sat is not None:
                res = sat
                self.last_by = "cms"
                model = [i if sol[i] else -i for i in range(1, len(sol)) if sol[i] is not None] if sat else None
        self.time += time.time() - t
        return res, model

    def delete(self):
        self.s.delete()


# ------------------------------------------------------------------------------------------------
# XOR detection + Gauss-Jordan over GF(2)
# ------------------------------------------------------------------------------------------------
def detect_xors(clauses, max_k=10, members=False):
    """XOR constraints encoded directly in the CNF: a set of k variables (k <= max_k) carrying all
    2^(k-1) clauses that forbid the assignments of one parity. Returns [(vars (0-based), rhs)] with
    XOR(vars) = rhs. A variable set whose clauses forbid both parities yields two rows (inconsistent).
    With members=True also returns the set of clauses (frozensets of literals) that encode the XORs."""
    groups = collections.defaultdict(set)
    for c in clauses:
        vs = frozenset(abs(l) - 1 for l in c)
        if 2 <= len(c) <= max_k and len(vs) == len(c):
            groups[vs].add(frozenset(c))
    out, mem = [], set()
    for vs, cls in groups.items():
        k = len(vs)
        need = 1 << (k - 1)
        if len(cls) < need:
            continue
        par = collections.Counter(sum(1 for l in c if l < 0) & 1 for c in cls)
        # a clause with p negative literals forbids an assignment of parity p -> the XOR is 1 - p
        for p in (0, 1):
            if par[p] == need:
                out.append((sorted(vs), 1 - p))
                mem.update(c for c in cls if sum(1 for l in c if l < 0) & 1 == p)
    return (out, mem) if members else out


class XorSystem:
    """GF(2) system in reduced row echelon form; rows are Python ints (bit v = variable v)."""

    def __init__(self, xors):
        self.rows = {}          # pivot var -> (mask, rhs); each pivot appears in exactly one row
        self.consistent = True
        self.vars = set()
        for vs, rhs in xors:
            mask = 0
            for v in vs:
                mask |= 1 << v
            self.vars.update(vs)
            if not self._add(self.rows, mask, rhs):
                self.consistent = False
                break
        self.n_xors = len(xors)
        self.time = 0.0

    @staticmethod
    def _add(rows, mask, rhs):
        """Add mask = rhs to the RREF system `rows` in place. Returns False on inconsistency."""
        # reduce by existing pivots
        m = mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            m ^= low
            if v in rows:
                pm, pr = rows[v]
                mask ^= pm
                rhs ^= pr
                m = mask & ~((low << 1) - 1)        # continue with higher bits of the updated mask
        if mask == 0:
            return rhs == 0
        piv = (mask & -mask).bit_length() - 1
        bit = 1 << piv
        for v, (pm, pr) in list(rows.items()):     # eliminate the new pivot from all rows (RREF)
            if pm & bit:
                rows[v] = (pm ^ mask, pr ^ rhs)
        rows[piv] = (mask, rhs)
        return True

    def rank(self, lits):
        """(consistent, rank) of the system plus unit equations for the assigned XOR variables."""
        if not self.consistent:
            return False, None
        t = time.time()
        rows = dict(self.rows)
        for l in lits:
            v = abs(l) - 1
            if v in self.vars and not self._add(rows, 1 << v, int(l > 0)):
                self.time += time.time() - t
                return False, None
        self.time += time.time() - t
        return True, len(rows)

    def implied(self, lits):
        """Add the assigned literals (those on XOR variables) as unit equations.
        Returns (ok, implied literals on XOR variables not in lits)."""
        if not self.consistent:
            return False, []
        t = time.time()
        rows = dict(self.rows)
        assigned = {}
        for l in lits:
            v = abs(l) - 1
            if v in self.vars:
                assigned[v] = l > 0
        for v, b in assigned.items():
            if not self._add(rows, 1 << v, int(b)):
                self.time += time.time() - t
                return False, []
        out = []
        for v, (pm, pr) in rows.items():
            if pm & (pm - 1) == 0 and v not in assigned:     # single-variable row: v = pr
                out.append((v + 1) if pr else -(v + 1))
        self.time += time.time() - t
        return True, out


# ------------------------------------------------------------------------------------------------
# LP relaxation with detected cardinality constraints (HiGHS)
# ------------------------------------------------------------------------------------------------
class LPRelaxation:
    """Feasibility LP: 0 <= x <= 1, one row per (detected) constraint sum(literals) >= rhs."""

    def __init__(self, clauses, n, card=True):
        import highspy
        from .card_detect import detect
        t = time.time()
        if card:
            cons, self.absorbed = detect(clauses)
        else:
            cons, self.absorbed = [(sorted(c), 1) for c in set(frozenset(c) for c in clauses)], 0
        self.n_card = sum(1 for lits, rhs in cons if rhs > 1 or len(lits) > 2 and rhs == len(lits) - 1)
        self.h = h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        inf = highspy.kHighsInf
        self.n = n
        for _ in range(n):
            h.addVar(0.0, 1.0)
        for lits, rhs in cons:
            idx = [abs(l) - 1 for l in lits]
            val = [1.0 if l > 0 else -1.0 for l in lits]
            nneg = sum(1 for l in lits if l < 0)
            h.addRow(float(rhs - nneg), inf, len(idx), idx, val)
        self.n_rows = len(cons)
        self.setup_time = time.time() - t
        self.time = 0.0
        self._status = highspy.HighsModelStatus

    def feasible(self, lits=()):
        """False iff the LP with the literals fixed is infeasible (=> the node is UNSAT)."""
        import numpy as np
        t = time.time()
        h = self.h
        idx = np.array([abs(l) - 1 for l in lits], dtype=np.int32)
        if len(idx):
            val = np.array([1.0 if l > 0 else 0.0 for l in lits])
            h.changeColsBounds(len(idx), idx, val, val)
        h.run()
        st = h.getModelStatus()
        if len(idx):
            h.changeColsBounds(len(idx), idx, np.zeros(len(idx)), np.ones(len(idx)))
        self.time += time.time() - t
        return st != self._status.kInfeasible

    def mip(self, lits=(), time_limit=60.0):
        """0/1 feasibility of the same rows (HiGHS MIP: branch and bound with cuts) under the literals.
        Returns (True, model) with a model as DIMACS literals, (False, None) if infeasible, (None, None)
        if undecided within time_limit. (HiGHS's thread pool is process-global: its size cannot be changed once
        the LP has run, so no `threads` option here.)"""
        import highspy
        import numpy as np
        t = time.time()
        m = highspy.Highs()
        m.setOptionValue("output_flag", False)
        m.setOptionValue("time_limit", float(time_limit))
        m.passModel(self.h.getModel())
        n = self.n
        m.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))
        idx = np.array([abs(l) - 1 for l in lits], dtype=np.int32)
        if len(idx):
            val = np.array([1.0 if l > 0 else 0.0 for l in lits])
            m.changeColsBounds(len(idx), idx, val, val)
        m.run()
        st = m.getModelStatus()
        self.time += time.time() - t
        if st == self._status.kInfeasible:
            return False, None
        if st == self._status.kOptimal:
            x = m.getSolution().col_value
            return True, [(v + 1) if x[v] > 0.5 else -(v + 1) for v in range(n)]
        return None, None
