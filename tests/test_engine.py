"""Correctness tests for the bool TN engine against numpy references."""
import sys, random
import numpy as np
from booltn import engine as E


def ref_contract(a, la, b, lb, lc):
    """numpy reference: OR-AND contraction via integer einsum."""
    letters = {}
    def let(l):
        if l not in letters:
            letters[l] = chr(ord("a") + len(letters)) if len(letters) < 26 else chr(ord("A") + len(letters) - 26)
        return letters[l]
    spec = "".join(let(l) for l in la) + "," + "".join(let(l) for l in lb) + "->" + "".join(let(l) for l in lc)
    return np.einsum(spec, a.astype(np.int64), b.astype(np.int64)) > 0


def check(nb, nm, nn, nk, nra=0, nrb=0, density=0.3, force=None, seed=0):
    rng = np.random.default_rng(seed)
    labs = list(range(100, 100 + nb + nm + nn + nk + nra + nrb))
    rng.shuffle(labs)
    Bt = labs[:nb]; Mt = labs[nb:nb + nm]; Nt = labs[nb + nm:nb + nm + nn]
    Kt = labs[nb + nm + nn:nb + nm + nn + nk]
    Ra = labs[nb + nm + nn + nk:nb + nm + nn + nk + nra]
    Rb = labs[nb + nm + nn + nk + nra:]
    la = Bt + Mt + Kt + Ra; lb = Nt + Kt + Bt + Rb
    random.Random(seed).shuffle(la); random.Random(seed + 1).shuffle(lb)
    lc = Bt + Mt + Nt
    random.Random(seed + 2).shuffle(lc)
    # density so that the OR over K is nontrivial
    p = min(0.5, max(0.02, (0.7 / max(1, 2 ** nk)) ** 0.5))
    a = rng.random((2,) * len(la)) < p
    b = rng.random((2,) * len(lb)) < p
    A = E.BTensor.from_bool(a, la)
    B = E.BTensor.from_bool(b, lb)
    C = E.contract(A, B, lc, force=force)
    got = C.to_bool()
    # reorder got to lc
    perm = [C.labels.index(l) for l in lc]
    got = np.transpose(got, perm) if len(lc) else got
    want = ref_contract(a, la, b, lb, lc)
    ok = np.array_equal(got, want)
    kind = force or "auto"
    print(f"{'PASS' if ok else 'FAIL'} B={nb} M={nm} N={nn} K={nk} Ra={nra} Rb={nrb} kind={kind} density(C)={want.mean() if want.size else want:.3f}")
    for t in (A, B, C):
        t.free()
    return ok


def check_permute(r, seed=0):
    rng = np.random.default_rng(seed)
    labs = list(range(r))
    a = rng.random((2,) * r) < 0.5
    A = E.BTensor.from_bool(a, labs)
    newl = labs[:]
    random.Random(seed).shuffle(newl)
    P = E.permute(A, newl)
    got = P.to_bool()
    want = np.transpose(a, [labs.index(l) for l in newl])
    ok = np.array_equal(got, want)
    print(f"{'PASS' if ok else 'FAIL'} permute r={r} {newl[:8]}...")
    A.free(); P.free()
    return ok


if __name__ == "__main__":
    allok = True
    for r, s in [(3, 0), (5, 1), (6, 2), (10, 3), (14, 4), (14, 5), (18, 6), (20, 7), (22, 8), (24, 9), (24, 10)]:
        allok &= check_permute(r, s)
    # structured permutations at r=22 for the tiled kernel
    rng = np.random.default_rng(5)
    a = rng.random((2,) * 22) < 0.5
    A = E.BTensor.from_bool(a, list(range(22)))
    base = list(range(22))
    tests = [base[:5] + base[5:][::-1], base[::-1], [3, 1, 4, 0, 2] + base[5:], base[10:] + base[:10],
             [7, 12, 0, 1, 2] + [x for x in base if x not in (7, 12, 0, 1, 2)], base[5:10] + base[:5] + base[10:]]
    for newl in tests:
        P = E.permute(A, newl)
        ok = np.array_equal(P.to_bool(), np.transpose(a, newl)); allok &= ok
        print(f"{'PASS' if ok else 'FAIL'} tiled permute {newl[:8]}...")
        P.free()
    A.free()
    # identity low bits (word copy) and in-word permutation modes
    rng = np.random.default_rng(9)
    a = rng.random((2,) * 12) < 0.5
    A = E.BTensor.from_bool(a, list(range(12)))
    for newl in [[0, 1, 2, 3, 4, 9, 7, 8, 5, 6, 11, 10], [3, 1, 4, 0, 2, 5, 6, 7, 8, 9, 10, 11], [4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5]]:
        P = E.permute(A, newl)
        ok = np.array_equal(P.to_bool(), np.transpose(a, newl)); allok &= ok
        print(f"{'PASS' if ok else 'FAIL'} permute mode test {newl}")
        P.free()
    A.free()
    cases = [
        # (B, M, N, K, Ra, Rb, force)
        (0, 3, 5, 2, 0, 0, "cc"), (2, 4, 6, 3, 0, 0, "cc"), (3, 1, 5, 0, 0, 0, "cc"), (1, 2, 7, 6, 0, 0, "cc"),
        (2, 0, 5, 1, 0, 0, "cc"), (1, 4, 5, 9, 0, 0, "cc"),
        (0, 7, 8, 7, 0, 0, "tc"), (1, 7, 8, 8, 0, 0, "tc"), (2, 6, 6, 7, 0, 0, "tc"), (1, 8, 9, 10, 0, 0, "tc"),
        (2, 3, 5, 7, 0, 0, "tc"), (0, 9, 7, 11, 0, 0, "tc"), (1, 5, 6, 8, 0, 0, "tc"),
        (2, 2, 2, 2, 0, 0, "gen"), (1, 3, 1, 3, 0, 0, None), (3, 0, 0, 4, 0, 0, None), (0, 0, 0, 5, 0, 0, None),
        (2, 3, 4, 2, 1, 2, None), (2, 6, 3, 2, 0, 0, None), (2, 1, 6, 8, 0, 0, None),
        (3, 3, 4, 4, 0, 0, "cc"), (2, 4, 1, 3, 0, 0, "cc"), (1, 5, 0, 2, 0, 0, "cc"), (2, 3, 2, 6, 0, 0, "cc"),
        (3, 2, 3, 1, 0, 0, None), (2, 8, 3, 5, 0, 0, None),
        (6, 2, 2, 5, 0, 0, None), (5, 1, 1, 3, 0, 0, None), (4, 0, 0, 2, 0, 0, None), (7, 1, 2, 0, 0, 0, None),
        (3, 0, 1, 4, 0, 0, None), (1, 1, 1, 1, 0, 0, None), (0, 1, 2, 1, 0, 0, None), (2, 2, 3, 6, 1, 1, None),
        (2, 1, 2, 14, 0, 0, "cc"), (1, 0, 5, 14, 0, 0, "cc"), (1, 2, 6, 13, 0, 0, "cc"), (0, 3, 3, 16, 0, 0, "cc"),
        (0, 6, 7, 15, 0, 0, "tc"), (1, 6, 5, 14, 0, 0, "tc"), (0, 7, 8, 13, 0, 0, "tc"),
        (1, 6, 8, 6, 0, 0, "tcp"), (2, 5, 7, 5, 0, 0, "tcp"), (0, 7, 9, 4, 0, 0, "tcp"), (1, 3, 6, 3, 0, 0, "tcp"),
    ]
    for i, (nb, nm, nn, nk, ra, rbb, f) in enumerate(cases):
        allok &= check(nb, nm, nn, nk, ra, rbb, force=f, seed=i)
    print("ALL PASSED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
