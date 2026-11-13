"""booltn: boolean tensor-network contraction (SAT and exact #SAT) on NVIDIA Hopper tensor cores.

Modules (import what you need; nothing is loaded eagerly):
  engine         bit-packed tensors, kernel choice + cost model, CompiledTree, SatSlicer   (decision SAT)
  count_engine   mod-p int8 contraction (CountTree, CountSlicer) + CRT: count_models    (exact #SAT)
  ordering       omeco TreeSA + slicing, tree JSON files
  slicer         greedy slicer for any tree (tc or modeled GPU time as the cost)
  tree_utils     contraction lists, tc/sc, GPU cost models (h100_time)
  satgen         random / grid k-SAT, DIMACS I/O, clause network (sat_einsum)
  hard_families  Tseitin grids, mutilated chessboard, plaquette CSP
  card_detect    CNF -> OPB with detected cardinality constraints
  run_sat, run_count   command-line entry points (booltn-sat, booltn-count)

The CUDA libraries (libbooltn.so, libcount8.so) are built with `make -C kernels` (from the repository root)
into booltn/lib/, or point $BOOLTN_LIB_DIR at them.
"""
__version__ = "0.1.0"
