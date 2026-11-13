# booltn

Boolean tensor-network contraction on NVIDIA Hopper (H100) tensor cores. It decides SAT and counts
models exactly (#SAT) for CNF formulas of low treewidth.

## Performance

All numbers are measured end to end on one H100 PCIe (80 GB) and checked against minisat, brute force,
closed-form counts or other counters. Details and scripts: [booltn-gpu](https://github.com/xuanzhaogao/booltn-gpu) (`docs/`).

| what | result |
|---|---|
| 1-bit tensor-core GEMM (`wgmma .b1.and.popc`) | **11.66 POPS** = 97.5 % of the measured instruction peak (8× int8, 16× fp16) |
| vs PyTorch fp16 on the same contraction trees | **36–48× faster**, 16–20× less memory |
| Largest tensor | **2^37 bits** (16 GiB) on 80 GB; fp16 would stop at about 2^33 entries |
| Grid 3-SAT, L = 22 / 24 (484 / 576 vars), whole instance | **20 ms / 1.27 s** at 4.6 / **7.96 POPS** |
| Random 3-SAT n = 70, α = 4.26 (UNSAT, 512 slices) | 41.7 s; memory-bound at about 35 TOPS |
| 81 published crafted SAT instances (sgen, chessboard, Tseitin, Urquhart, subset cardinality) | **80 solved in 0.7–16 s**, incl. instances no SAT-competition entrant solved; kissat 38, CaDiCaL 33, CryptoMiniSat 56 (900 s) |
| #SAT on grid-structured instances that sharpSAT-TD and d4v2 could not finish in 30 min | **all counted in 1.3–25 s** (Tseitin, domino tilings, plaquette CSP, grid 3-SAT) |
| #SAT where those CPU counters do finish | **5–700× faster**, identical counts (9/9) |

Limits:
* The cost is set by the treewidth. Wide formulas, such as random 3-SAT with n ≥ 100 for #SAT, are
  easier for CPU solvers.
* Counts are limited to about 10^110 (8-bit primes).
* Each crafted SAT family has a CPU specialist that is faster: Gauss for parity, LP for the
  chessboard, cardinality detection for sgen.

## How it works

* **Decision (SAT / UNSAT).** Each clause is a 0/1 tensor over its variables. Contracting the network
  in the OR-AND semiring gives the answer. Tensors are bit-packed (one bit per entry), and large
  contractions run on the 1-bit tensor cores.
* **Counting (#SAT).** The same networks are contracted over integers mod p (p ≤ 251, one byte per
  entry) on the int8 tensor cores. The exact count is rebuilt by CRT over several primes.
* Whole contraction trees are compiled once into a CUDA graph with a single memory arena. Each slice
  (an assignment of the sliced variables) is then one graph replay.

For better contraction orders, see the sibling package [treesa_gpu](https://github.com/xuanzhaogao/treesa_gpu). `booltn` does not
depend on it; both use the same tree files. The research repository [booltn-gpu](https://github.com/xuanzhaogao/booltn-gpu)
includes this package as a submodule.

## Install

```bash
module load cuda/12.8.0                 # Hopper only: needs sm_90a (wgmma, TMA)
make -C kernels                         # -> booltn/lib/libbooltn.so, libcount8.so
pip install -e .                        # numpy; also python-sat (minisat check) and omeco (ordering)
```

omeco's Python bindings come from [omeco](https://github.com/GiggleLiu/omeco) (`maturin develop` in
`omeco-python/`). To use libraries built elsewhere, set `BOOLTN_LIB_DIR`.

## Command line

```bash
booltn-sat --cnf f.cnf --expect none --sc 36        # SAT / UNSAT of a DIMACS file (omeco order, slicing to 2^36 bits)
booltn-sat --n 60 --seed 3 --sc 32                  # random 3-SAT, answer checked against minisat
booltn-sat --grid 24 --alpha 3 --seed 1 --tree trees/grid_L24_a3_s1_sc33.json --max-slices 8 --timing
booltn-count --cnf f.cnf [--sc 32] [--tree t.json]  # exact model count, one JSON line
```

`booltn-sat` and `booltn-count` are the same as `python -m booltn.run_sat` and `python -m booltn.run_count`.
`--timing` prints the time spent in each kernel kind and the heaviest contractions. `--save-tree` and
`--tree` store and reuse contraction orders.

## Python

```python
from booltn.satgen import read_dimacs, sat_einsum
from booltn.ordering import optimize_order
from booltn import engine, count_engine

clauses, n = read_dimacs("f.cnf")
ixs, iy, sizes = sat_einsum(clauses, n)                         # one tensor per clause, labels = variables
tree, sliced, _ = optimize_order(ixs, sizes, sc_target=36, sc_slice=32)   # omeco TreeSA + slicing
sat, slices_done, info = engine.decide(clauses, tree, sliced)  # compiled plan, all slices
count, cinfo = count_engine.count_models(clauses, n, tree, sliced)  # exact #SAT
```

A tree is an omeco-format dict (`{"tensor_index": i}` or `{"args": [l, r], "eins": {...}}`) over the
labels of `sat_einsum`. Tree files hold `{"tree": ..., "sliced": [...]}` (`ordering.load_tree` and
`ordering.save_tree`). `treesa_gpu.runner_json` writes the same format.

## Modules

| module | content |
|---|---|
| `engine` | Bit-packed tensors (labels l0..l_{r-1}: l0 is the fastest bit). Kernel choice from a fitted cost model (`kernel_costs`, `choose_kind`). `CompiledTree` (plan, arena, CUDA graph), `SatSlicer` (leaf words for each slice), `decide` |
| `count_engine` | `CountTree` and `CountSlicer` (int8 mod-p plan), `count_models` (CRT; `mode="stable"` or `"rigorous"`), `profile_tree` |
| `ordering` | `optimize_order` (omeco TreeSA + TreeSASlicer), `prepare_tree`, `load_tree` / `save_tree` |
| `slicer` | `greedy_slice`: slicing for a tree from any optimizer, with cost tc or modeled GPU time |
| `tree_utils` | Contraction shapes (B, M, N, K), tc/sc, GPU time models (`gpu_time_v2`, fitted on the H100) |
| `satgen` | Random and grid k-SAT, DIMACS input/output, `sat_einsum` |
| `hard_families` | Tseitin grids, mutilated chessboard, plaquette CSP, noisy Tseitin |
| `card_detect` | CNF → OPB with syntactically detected cardinality constraints (`python -m booltn.card_detect in.cnf out.opb`) |

CUDA sources (`kernels/`):

| file | content |
|---|---|
| `booltn.cu` | Bit permutations, CUDA-core / tensor-core / small-batch contraction kernels, plans, CUDA graphs |
| `tc_gemm.cuh`, `bgemm_sm90.cuh` | 1-bit wgmma GEMM: TMA producer warp, 2 consumer warpgroups, split-K with `atomicOr` |
| `count8.cu` | Byte permutations, dp4a / outer-product / int8 wgmma mod-p kernels |
| `wgmma_b1.inc`, `wgmma_u8.inc` | Generated inline-PTX wrappers (`make inc` regenerates them) |
| `bgemm_test.cu` | Standalone 1-bit GEMM test and benchmark (`make bgemm_test`) |

## Tests

```bash
flock /tmp/booltn_gpu.lock tests/run_all.sh     # on an H100
```

The suite covers:
* `test_engine.py`: permutations and every kernel / shape class against numpy.
* `test_count_kernels.py`: the int8 kernels against numpy.
* `test_count.py`: counts against brute force, Tseitin and OEIS domino tilings.
* `test_wide.py`: clauses wider than one word.
* `test_slices.py`: every slice against minisat with assumptions, for the eager path, the plan, and the CUDA graph.
* `test_decide.py`: `decide` against minisat.
