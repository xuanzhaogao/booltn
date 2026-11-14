"""booltn.card_detect: the k-subset rule recovers an at-most-k constraint, the AMO rule a clique, and the work
budget stops the k-subset rule (the remaining clauses stay clauses) without changing anything below it. CPU only."""
import itertools
import sys
from booltn.card_detect import detect


def main():
    ok = True
    # at most 2 of x1..x7 true: every 3-subset of the negated literals is a clause
    amk = [[-a, -b, -c] for a, b, c in itertools.combinations(range(1, 8), 3)]
    # at most 1 of x8..x12: pairwise binary clauses (AMO clique)
    amo = [[-a, -b] for a, b in itertools.combinations(range(8, 13), 2)]
    other = [[1, 2, 8], [3, -9, 10]]
    cl = amk + amo + other

    st = {}
    cons, absorbed = detect(cl, stats=st)
    want_k = (sorted(range(-7, 0)), 7 - 3 + 1)            # sum(not x_i) >= 5  <=>  sum x_i <= 2
    want_1 = (sorted(range(-12, -7)), 5 - 1)               # sum(not x_i) >= 4  <=>  sum x_i <= 1
    good = want_k in cons and want_1 in cons and absorbed == len(amk) + len(amo) and not st["budget_hit"]
    print(f"{'ok  ' if good else 'FAIL'} default: {len(cons)} constraints, {absorbed} clauses absorbed, work {st['work']}")
    ok &= good

    st0 = {}
    cons0, absorbed0 = detect(cl, work_budget=0, stats=st0)
    good = (st0["budget_hit"] and want_k not in cons0 and want_1 in cons0 and absorbed0 == len(amo)
            and all((sorted(c), 1) in cons0 for c in amk))
    print(f"{'ok  ' if good else 'FAIL'} work_budget 0: k-subset rule off, {absorbed0} absorbed (AMO only)")
    ok &= good

    stb = {}
    consb, _ = detect(cl, work_budget=st["work"], stats=stb)   # a budget the run does not exceed: same output
    good = consb == cons and not stb["budget_hit"]
    print(f"{'ok  ' if good else 'FAIL'} work_budget = work: identical output")
    ok &= good

    stn = {}
    consn, _ = detect(cl, work_budget=None, stats=stn)
    good = consn == cons and stn["work"] == st["work"]
    print(f"{'ok  ' if good else 'FAIL'} work_budget None: identical output")
    ok &= good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
