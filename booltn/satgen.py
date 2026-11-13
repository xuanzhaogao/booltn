"""Random k-SAT instance generation and tensor-network (einsum) construction."""
import random


def random_ksat(n, m, k, seed=0):
    """Return a list of clauses; each clause is a list of nonzero ints (DIMACS literals)."""
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(1, n + 1), k)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return clauses


def sat_einsum(clauses, n):
    """Einsum spec of the bool TN: one tensor per clause, labels = variables (0-based).

    Returns ixs (list of label lists), iy (output labels, empty = scalar), sizes dict.
    """
    ixs = [[abs(l) - 1 for l in c] for c in clauses]
    used = sorted({l for ix in ixs for l in ix})
    sizes = {l: 2 for l in used}
    return ixs, [], sizes


def write_dimacs(path, clauses, n):
    with open(path, "w") as f:
        f.write(f"p cnf {n} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def read_dimacs(path):
    clauses, n = [], 0
    cur = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "c%":
                continue
            if line[0] == "p":
                n = int(line.split()[2])
                continue
            for t in line.split():
                v = int(t)
                if v == 0:
                    clauses.append(cur)
                    cur = []
                else:
                    cur.append(v)
    return clauses, n


def grid_ksat(L, alpha, k=3, radius=1, seed=0):
    """Local k-SAT on an L x L grid: each clause picks k distinct variables within a (2r+1)^2 window."""
    rng = random.Random(seed)
    n = L * L
    m = round(alpha * n)
    clauses = []
    for _ in range(m):
        cx, cy = rng.randrange(L), rng.randrange(L)
        cand = [(x, y) for x in range(max(0, cx - radius), min(L, cx + radius + 1))
                for y in range(max(0, cy - radius), min(L, cy + radius + 1))]
        pts = rng.sample(cand, k)
        clauses.append([(x * L + y + 1) * (1 if rng.random() < 0.5 else -1) for x, y in pts])
    return clauses, n
