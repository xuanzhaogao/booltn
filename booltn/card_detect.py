"""Syntactic cardinality detection: CNF -> OPB with recovered cardinality constraints.

Rules (in the spirit of Biere, Le Berre, Lonca & Manthey, "Detecting cardinality constraints in CNF",
SAT 2014, syntactic variant):
  * k-subset rule: if for a literal set L (|L| = s) every w-subset of L is a clause, then
    "at least s-w+1 literals of L are true". With all-negative literals this is at-most-k
    (k = w-1); all-positive gives at-least-k.
  * AMO rule (w = 2): greedy clique cover of the literal conflict graph (binary clause l1 | l2 = edge
    between -l1 and -l2); each clique C (|C| >= 3) of literals becomes sum(C) <= 1.
Every clause not absorbed by a detected constraint is kept as `sum(literals) >= 1`.

  python -m booltn.card_detect in.cnf out.opb      # prints detection statistics
"""
import sys, os, itertools, collections, math
from .satgen import read_dimacs


def detect(clauses, max_set=128, subset_budget=5000):
    """max_set caps a detected literal set; growing a set of size s by one literal checks C(s, w-1) subsets,
    which must stay <= subset_budget (w = 3: s <= 100; w = 4: 32; w = 5: 20). A fixed cap of 12 split the
    at-most-2 constraints of tph8-10 (17-21 pigeons per hole) and left their LP feasible."""
    cset = set(frozenset(c) for c in clauses)
    used = set()
    cons = []  # (literal list, rhs): sum of literals >= rhs  (literal -v means (1 - x_v))
    # ---- k-subset rule for widths >= 3 ----
    by_w = collections.defaultdict(list)
    for c in cset:
        if len(c) >= 3:
            by_w[len(c)].append(c)
    for w, cls in by_w.items():
        lit_clauses = collections.defaultdict(list)
        for c in cls:
            for l in c:
                lit_clauses[l].append(c)
        for c in cls:
            if c in used:
                continue
            L = set(c)
            # grow L greedily with literals l such that all (w-1)-subsets of L plus l are clauses
            grown = True
            while grown and len(L) < max_set and math.comb(len(L), w - 1) <= subset_budget:
                grown = False
                cand = collections.Counter(l for x in L for cc in lit_clauses[x] for l in cc if l not in L)
                for l, _ in cand.most_common():
                    if -l in L:
                        continue
                    if all(frozenset(T) | {l} in cset for T in itertools.combinations(L, w - 1)):
                        L.add(l); grown = True
                        break
            if len(L) > w:
                covered = [frozenset(T) for T in itertools.combinations(L, w)]
                if all(t in cset for t in covered):
                    used.update(covered)
                    cons.append((sorted(L), len(L) - w + 1))
    # ---- AMO cliques: a binary clause (l1 | l2) says the literals -l1, -l2 are not both true; nodes are
    #      literals, so the rule is invariant under polarity flips ("shuffled-as" instances) ----
    nbr = collections.defaultdict(set)
    for c in cset:
        if len(c) == 2:
            a, b = sorted(-l for l in c)
            nbr[a].add(b); nbr[b].add(a)
    edges_left = {frozenset((a, b)) for a in nbr for b in nbr[a]}
    for v in sorted(nbr, key=lambda x: -len(nbr[x])):
        while True:
            # start a clique from an uncovered edge at v, extend greedily
            starts = [b for b in nbr[v] if frozenset((v, b)) in edges_left]
            if not starts:
                break
            C = {v, starts[0]}
            for u in sorted(nbr[v] & nbr[starts[0]], key=lambda x: -len(nbr[x])):
                if all(u in nbr[x] for x in C):
                    C.add(u)
            if len(C) < 3:
                break
            for a, b in itertools.combinations(C, 2):
                e = frozenset((a, b))
                edges_left.discard(e)
                used.add(frozenset((-a, -b)))
            cons.append((sorted(-x for x in C), len(C) - 1))  # sum(not u) >= |C|-1  <=>  sum u <= 1 (u literals)
    rest = [sorted(c) for c in cset if c not in used]
    for c in rest:
        cons.append((c, 1))
    return cons, len(cset) - len(rest)


def write_opb(cons, n, path):
    with open(path, "w") as f:
        f.write(f"* #variable= {n} #constraint= {len(cons)}\n")
        for lits, rhs in cons:
            # sum over literals: x_v or ~x_v
            terms = " ".join(f"+1 {'x' if l > 0 else '~x'}{abs(l)}" for l in lits)
            f.write(f"{terms} >= {rhs} ;\n")


if __name__ == "__main__":
    cl, n = read_dimacs(sys.argv[1])
    cons, absorbed = detect(cl)
    ncard = sum(1 for l, r in cons if r > 1 or (len(l) > 2 and r == len(l) - 1))
    write_opb(cons, n, sys.argv[2])
    print(f"{os.path.basename(sys.argv[1])}: {len(cl)} clauses -> {len(cons)} constraints "
          f"({absorbed} clauses absorbed into {len(cons) - (len(cl) - absorbed)} cardinality constraints)")
