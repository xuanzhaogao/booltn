#!/bin/bash
# Run every booltn test on the local GPU (H100: sm_90a). Needs the CUDA libraries (make -C kernels),
# the booltn package installed (pip install -e .) and omeco + python-sat. Wrap in
# `flock /tmp/booltn_gpu.lock` on a shared GPU.
cd "$(dirname "$0")"
fail=0
for t in test_engine test_count_kernels test_count test_wide test_slices test_decide; do
  if python $t.py > /tmp/booltn_$t.$$.log 2>&1; then echo "PASS $t"; else echo "FAIL $t (log: /tmp/booltn_$t.$$.log)"; fail=1; fi
done
exit $fail
