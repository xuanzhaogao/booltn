"""Kernel-level tests of libcount8: tc8 (int8 tensor cores), cc8 (dp4a, split-K), perm8 (with sums) vs numpy."""
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


def gemm_test(kind, bb, mm, nn, kk, prime, seed=0):
    rng = np.random.default_rng(seed)
    Bn, M, N, K = 1 << bb, 1 << mm, 1 << nn, 1 << kk
    A = rng.integers(0, prime, size=(Bn, M, K), dtype=np.int64)
    Bm = rng.integers(0, prime, size=(Bn, N, K), dtype=np.int64)
    want = (np.einsum("bmk,bnk->bmn", A, Bm) % prime).astype(np.uint8)
    pA = dev(A.astype(np.uint8))
    pB = dev(np.transpose(Bm, (0, 2, 1)).astype(np.uint8)) if kind == "cco" else dev(Bm.astype(np.uint8))
    pC = L.c8_malloc(Bn * M * N)
    pp = L.c8_malloc(4)
    pv = np.array([prime], dtype=np.int32); L.c8_h2d(pp, pv.ctypes.data, 4, -1)
    scratch = L.c8_malloc(4 * Bn * M * N) if kind == "cc_split" else None
    plan = L.c8_plan_new()
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
    got = host(pC, Bn * M * N).reshape(Bn, M, N)
    ok = np.array_equal(got, want)
    print(f"{'PASS' if ok else 'FAIL'} {kind:8s} B={bb} M={mm} N={nn} K={kk} p={prime}  ({dt*1e3:.1f} ms)"
          + ("" if ok else f"  mismatches={int((got != want).sum())}"), flush=True)
    for q in (pA, pB, pC, pp) + ((scratch,) if scratch else ()):
        L.c8_free(q)
    L.c8_plan_free(plan)
    return ok


def perm_test(r, nred, prime, seed=0):
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
    want = np.transpose(y, [rem_axes.index(l) for l in newl[::-1]]).astype(np.uint8).reshape(-1)
    pin = dev(x.astype(np.uint8).reshape(-1))
    pout = L.c8_malloc(max(1 << len(newl), 4))
    pp = L.c8_malloc(4); pv = np.array([prime], dtype=np.int32); L.c8_h2d(pp, pv.ctypes.data, 4, -1)
    plan = L.c8_plan_new()
    src = [labs.index(l) for l in newl]
    rc = L.c8_plan_permute(plan, pin, pout, len(newl), CE.IntArr(src), CE.IntArr(red), len(red), pp)
    assert rc == 0
    L.c8_plan_run(plan, -1, 0); L.c8_sync_all()
    got = host(pout, 1 << len(newl))
    ok = np.array_equal(got, want)
    print(f"{'PASS' if ok else 'FAIL'} perm8 r={r} nred={nred} p={prime}", flush=True)
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
    for args in [(0, 3, 4, 2, 251), (2, 0, 6, 0, 241), (1, 4, 5, 4, 97), (3, 2, 4, 3, 7), (0, 5, 10, 1, 251)]:
        ok &= gemm_test("cco", *args)
    for args in [(10, 0, 251), (12, 2, 241), (8, 3, 97), (14, 0, 251), (6, 1, 5), (20, 0, 251), (16, 0, 13), (9, 0, 7)]:
        ok &= perm_test(*args)
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
