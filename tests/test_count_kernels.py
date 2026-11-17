"""Kernel-level tests of libcount8: tc8 (int8 tensor cores), cc8 (dp4a, split-K), perm8 (with sums) vs numpy,
for byte residues (width 1, moduli <= 256) and 16-bit residues in two byte planes (width 2, moduli < 2^16)."""
import sys, ctypes, time
import numpy as np
from booltn import count_engine as CE

L = CE.lib()
VP = ctypes.c_void_p


def dev(arr):
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    p = L.c8_malloc(max(a.nbytes, 4))
    L.c8_h2d(p, a.ctypes.data, a.nbytes, -1)
    return p


def host(p, n):
    out = np.zeros(n, dtype=np.uint8)
    L.c8_d2h(out.ctypes.data, p, n, -1)
    return out


def planes(x, w):
    """flat array of residues -> device byte layout: width 1 bytes; width 2 low-byte plane then high-byte plane"""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    return x.astype(np.uint8) if w == 1 else np.concatenate([(x & 255).astype(np.uint8), (x >> 8).astype(np.uint8)])


def dev_w(x, w):
    return dev(planes(x, w))


def host_w(p, n, w):
    b = host(p, n * w).astype(np.int64)
    return b if w == 1 else b[:n] + 256 * b[n:]


def gemm_test(kind, bb, mm, nn, kk, prime, seed=0, w=1):
    rng = np.random.default_rng(seed)
    Bn, M, N, K = 1 << bb, 1 << mm, 1 << nn, 1 << kk
    A = rng.integers(0, prime, size=(Bn, M, K), dtype=np.int64)
    Bm = rng.integers(0, prime, size=(Bn, N, K), dtype=np.int64)
    if seed % 2 == 1:   # extreme values: all entries p - 1 (largest partial sums)
        A[:], Bm[:] = prime - 1, prime - 1
    want = np.einsum("bmk,bnk->bmn", A, Bm) % prime   # int64: K * (2^16)^2 < 2^63
    pA = dev_w(A, w)
    pB = dev_w(np.transpose(Bm, (0, 2, 1)), w) if kind == "cco" else dev_w(Bm, w)
    pC = L.c8_malloc(Bn * M * N * w)
    pp = L.c8_malloc(4)
    pv = np.array([prime], dtype=np.int32); L.c8_h2d(pp, pv.ctypes.data, 4, -1)
    scratch = L.c8_malloc(4 * Bn * M * N) if kind == "cc_split" else None
    plan = L.c8_plan_new_w(w)
    if kind == "tc":
        rc = L.c8_plan_tc(plan, pA, pB, pC, bb, mm, nn, kk, pp)
    elif kind == "cco":
        rc = L.c8_plan_cco(plan, pA, pB, pC, bb, mm, nn, kk, pp)
    else:
        rc = L.c8_plan_cc(plan, pA, pB, pC, scratch, bb, mm, nn, kk, pp)
    assert rc == 0, rc
    t = time.time()
    assert L.c8_plan_run(plan, -1, 0) == 0
    L.c8_sync_all()
    dt = time.time() - t
    got = host_w(pC, Bn * M * N, w).reshape(Bn, M, N)
    ok = np.array_equal(got, want.astype(np.int64))
    print(f"{'PASS' if ok else 'FAIL'} {kind:8s} w={w} B={bb} M={mm} N={nn} K={kk} p={prime}  ({dt*1e3:.1f} ms)"
          + ("" if ok else f"  mismatches={int((got != want).sum())}"), flush=True)
    for q in (pA, pB, pC, pp) + ((scratch,) if scratch else ()):
        L.c8_free(q)
    L.c8_plan_free(plan)
    return ok


def ccog_test(bb, mm, nn, kk, prime, seed=0, w=1):
    """ccog (ccov with A, B, C in random label orders; the lowest LNV positions of B and C are N labels 0..LNV-1)."""
    rng = np.random.default_rng(seed)
    lnv = {16: 4, 8: 3, 4: 2}[(16 if kk <= 3 else 8) if w == 1 else (8 if kk <= 3 else 4)]
    Bl = [("b", i) for i in range(bb)]; Ml = [("m", i) for i in range(mm)]
    Nl = [("n", i) for i in range(nn)]; Kl = [("k", i) for i in range(kk)]
    def shuffled(xs):
        xs = list(xs); rng.shuffle(xs); return xs
    LA = shuffled(Kl + Ml + Bl)
    LB = Nl[:lnv] + shuffled(Nl[lnv:] + Kl + Bl)
    LC = Nl[:lnv] + shuffled(Nl[lnv:] + Ml + Bl)
    vals = {}
    def tensor(labels):
        x = rng.integers(0, prime, size=(2,) * len(labels), dtype=np.int64)   # axis i <-> labels[i]
        if seed % 2 == 1:
            x[...] = prime - 1
        return x
    lA, lB = Bl + Ml + Kl, Bl + Kl + Nl          # logical axis orders
    A = tensor(lA); Bm = tensor(lB)
    allL = sorted(set(lA) | set(lB), key=str)
    letter = {l: chr(97 + i) for i, l in enumerate(allL)}
    lC = Bl + Ml + Nl
    spec = "".join(letter[l] for l in lA) + "," + "".join(letter[l] for l in lB) + "->" + "".join(letter[l] for l in lC)
    want = np.einsum(spec, A, Bm) % prime
    def to_dev(x, logical, layout):
        # device position p <-> layout[p] (p = 0 fastest): C-order flatten with axes layout[r-1], ..., layout[0]
        ax = [logical.index(l) for l in layout[::-1]]
        return np.transpose(x, ax).reshape(-1) if layout else x.reshape(-1)
    pA = dev_w(to_dev(A, lA, LA), w); pB = dev_w(to_dev(Bm, lB, LB), w)
    pC = L.c8_malloc(max(4, (1 << (bb + mm + nn)) * w))
    pp = L.c8_malloc(4)
    pv = np.array([prime], dtype=np.int32); L.c8_h2d(pp, pv.ctypes.data, 4, -1)
    pos = []
    for l in Bl: pos += [LA.index(l), LB.index(l), LC.index(l)]
    for l in Ml: pos += [LA.index(l), LC.index(l)]
    for l in Nl: pos += [LB.index(l), LC.index(l)]
    for l in Kl: pos += [LA.index(l), LB.index(l)]
    plan = L.c8_plan_new_w(w)
    rc = L.c8_plan_ccog(plan, pA, pB, pC, bb, mm, nn, kk, CE.IntArr(pos), pp)
    assert rc == 0, rc
    assert L.c8_plan_run(plan, -1, 0) == 0
    L.c8_sync_all()
    got = host_w(pC, 1 << (bb + mm + nn), w)
    ok = np.array_equal(got, to_dev(want, lC, LC))
    print(f"{'PASS' if ok else 'FAIL'} ccog     w={w} B={bb} M={mm} N={nn} K={kk} p={prime}", flush=True)
    for q in (pA, pB, pC, pp):
        L.c8_free(q)
    L.c8_plan_free(plan)
    return ok


def perm_test(r, nred, prime, seed=0, w=1):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, prime, size=(2,) * r, dtype=np.int64)  # axis order: slowest..fastest = label r-1..0
    labs = list(range(r))
    red = labs[:nred]
    keep = [l for l in labs if l not in red]
    rng2 = np.random.default_rng(seed + 1)
    newl = list(rng2.permutation(keep))
    # numpy reference: axis i of x corresponds to label (r-1-i)
    axes_of = {l: r - 1 - l for l in labs}
    y = x.sum(axis=tuple(axes_of[l] for l in red)) % prime if red else x
    rem = [l for l in labs if l not in red]  # remaining labels in label order
    rem_axes = sorted(rem, key=lambda l: -l)  # y's axes are labels in descending order
    want = np.transpose(y, [rem_axes.index(l) for l in newl[::-1]]).astype(np.int64).reshape(-1)
    pin = dev_w(x.reshape(-1), w)
    pout = L.c8_malloc(max(w << len(newl), 4))
    pp = L.c8_malloc(4); pv = np.array([prime], dtype=np.int32); L.c8_h2d(pp, pv.ctypes.data, 4, -1)
    plan = L.c8_plan_new_w(w)
    src = [labs.index(l) for l in newl]
    rc = L.c8_plan_permute(plan, pin, pout, len(newl), CE.IntArr(src), CE.IntArr(red), len(red), pp)
    assert rc == 0
    L.c8_plan_run(plan, -1, 0); L.c8_sync_all()
    got = host_w(pout, 1 << len(newl), w)
    ok = np.array_equal(got, want)
    print(f"{'PASS' if ok else 'FAIL'} perm8 w={w} r={r} nred={nred} p={prime}", flush=True)
    for q in (pin, pout, pp):
        L.c8_free(q)
    L.c8_plan_free(plan)
    return ok


if __name__ == "__main__":
    ok = True
    for args in [(0, 7, 8, 5, 251), (1, 6, 7, 6, 241), (2, 7, 9, 8, 97), (0, 8, 8, 10, 251), (1, 5, 6, 4, 13),
                 (0, 9, 10, 7, 251), (3, 6, 6, 5, 3), (0, 7, 7, 15, 251)]:
        ok &= gemm_test("tc", *args)
    for args in [(0, 3, 4, 2, 251), (2, 2, 5, 0, 241), (1, 4, 3, 1, 97), (0, 5, 5, 6, 251), (2, 0, 0, 3, 7)]:
        ok &= gemm_test("cc", *args)
    for args in [(0, 2, 3, 12, 251), (1, 1, 2, 14, 241), (0, 0, 0, 16, 251)]:
        ok &= gemm_test("cc_split", *args)
    # long K, small output (cck): several M x N tiles, partial tiles, K slices, extreme entries (odd seeds)
    for i, args in enumerate([(2, 3, 4, 13, 251), (0, 4, 5, 10, 256), (0, 2, 4, 18, 241), (3, 1, 1, 9, 7), (1, 0, 6, 12, 256)]):
        ok &= gemm_test("cc_split", *args, seed=i)
        ok &= gemm_test("cc_split", *args[:4], (65521, 65519, 40009, 257)[i % 4], seed=i, w=2)
    for args in [(0, 3, 4, 2, 251), (2, 0, 6, 0, 241), (1, 4, 5, 4, 97), (3, 2, 4, 3, 7), (0, 5, 10, 1, 251)]:
        ok &= gemm_test("cco", *args)
    # small-K outer-product kernel (ccov, dp4a): every K, row blocks (M > 16 rows), N = 16 bytes, extreme entries
    # (odd seeds), modulus 256; then the same with the older cco8 / cco16 kernels (c8_set_cco_impl(1))
    CCO_MORE = [(0, 7, 4, 4, 251, 1), (2, 6, 5, 3, 256, 1), (1, 1, 12, 2, 241, 0), (0, 0, 4, 0, 7, 0),
                (3, 4, 6, 4, 256, 3), (1, 5, 4, 1, 255, 5), (0, 3, 8, 3, 2, 0)]
    for impl in (2, 1):
        L.c8_set_cco_impl(impl)
        for bb, mm, nn, kk, p, sd in CCO_MORE:
            ok &= gemm_test("cco", bb, mm, nn, kk, p, seed=sd)
            ok &= gemm_test("cco", bb, mm, nn, kk, (65521, 65519, 40009, 257)[sd % 4], seed=sd, w=2)
    L.c8_set_cco_impl(2)
    for bb, mm, nn, kk, p, sd in CCO_MORE + [(2, 3, 9, 2, 251, 2), (1, 2, 7, 4, 241, 4), (0, 6, 6, 1, 256, 7)]:
        ok &= ccog_test(bb, mm, nn, kk, p, seed=sd)
        ok &= ccog_test(bb, mm, nn, kk, (65521, 65519, 40009, 257)[sd % 4], seed=sd, w=2)
    for args in [(10, 0, 251), (12, 2, 241), (8, 3, 97), (14, 0, 251), (6, 1, 5), (20, 0, 251), (16, 0, 13), (9, 0, 7)]:
        ok &= perm_test(*args)
    # 16-bit residues in byte planes (width 2); odd seeds use all entries p - 1
    P16 = (65521, 65519, 40009, 257)
    for i, args in enumerate([(0, 7, 8, 5), (1, 6, 7, 6), (2, 7, 9, 8), (0, 8, 8, 10), (1, 5, 6, 4), (0, 9, 10, 7),
                              (3, 6, 6, 5), (0, 7, 7, 15), (0, 7, 8, 16)]):
        ok &= gemm_test("tc", *args, P16[i % 4], seed=i, w=2)
    for i, args in enumerate([(0, 3, 4, 2), (2, 2, 5, 0), (1, 4, 3, 1), (0, 5, 5, 6), (2, 0, 0, 3)]):
        ok &= gemm_test("cc", *args, P16[i % 4], seed=i, w=2)
    for i, args in enumerate([(0, 2, 3, 12), (1, 1, 2, 14), (0, 0, 0, 16)]):
        ok &= gemm_test("cc_split", *args, P16[i % 4], seed=i, w=2)
    for i, args in enumerate([(0, 3, 4, 2), (2, 0, 6, 0), (1, 4, 5, 4), (3, 2, 4, 3), (0, 5, 10, 1)]):
        ok &= gemm_test("cco", *args, P16[i % 4], seed=i, w=2)
    for i, args in enumerate([(10, 0), (12, 2), (8, 3), (14, 0), (6, 1), (20, 0), (16, 0), (9, 0), (7, 0)]):
        ok &= perm_test(*args, P16[i % 4], seed=i, w=2)
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
