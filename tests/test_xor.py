"""booltn.reduce.XorSystem: the C path (kernels/gf2.c, lib/libbtmeasure.so) gives the same consistency, implied
literals and rank as the Python Gauss-Jordan path, on random XOR systems with random partial assignments
(consistent and inconsistent), including systems over more than 64 variables and scattered variable numbers.
CPU only; skipped (exit 0) when the library has no btx_implied."""
import random
import sys

from booltn.reduce import XorSystem


def python_path(xs):
    x = XorSystem(xs)
    x._c = False
    return x


def main():
    c0 = XorSystem([([0, 1], 1)])
    if not c0._cstate():
        print("SKIP: lib/libbtmeasure.so without btx_implied (make -C kernels measure)")
        return
    rng = random.Random(1)
    ok, n_cases, n_unsat, n_implied = True, 0, 0, 0
    for trial in range(400):
        nv = rng.choice([5, 20, 63, 64, 65, 130, 300])
        scale = rng.choice([1, 7])                      # scattered variable numbers (compressed columns)
        nx = rng.randint(1, nv)
        xs = []
        for _ in range(nx):
            k = rng.randint(1, min(6, nv))
            xs.append((sorted(scale * v for v in rng.sample(range(nv), k)), rng.randint(0, 1)))
        a, b = XorSystem(xs), python_path(xs)
        if not a.consistent:
            continue
        allv = sorted({v for vs, _ in xs for v in vs} | {scale * nv + 3})   # one variable outside the system
        for _ in range(10):
            m = rng.randint(0, len(allv))
            lits = [(v + 1) * rng.choice([1, -1]) for v in rng.sample(allv, m)]
            ra, rb = a.implied(lits), b.implied(lits)
            ka, kb = a.rank(lits), b.rank(lits)
            same = ra[0] == rb[0] and sorted(ra[1]) == sorted(rb[1]) and ka == kb
            n_cases += 1
            n_unsat += not rb[0]
            n_implied += len(rb[1])
            if not same:
                ok = False
                print(f"FAIL nv {nv} xors {xs} lits {lits}: C {ra} {ka}  Python {rb} {kb}")
                break
    print(f"{'ok  ' if ok else 'FAIL'} {n_cases} assignments: {n_unsat} inconsistent, {n_implied} implied literals, "
          "C == Python")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
