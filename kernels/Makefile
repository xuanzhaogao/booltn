# CUDA kernels of booltn (Hopper only: wgmma / TMA need sm_90a).
#   module load cuda/12.8.0
#   make -C booltn/kernels                 -> booltn/booltn/lib/libbooltn.so, libcount8.so
#   make -C booltn/kernels bgemm_test      -> standalone 1-bit GEMM test/benchmark (./bgemm_test M N K B [check] [reps])
#   make -C booltn/kernels inc             -> regenerate the wgmma inline-PTX wrappers (*.inc, checked in)
NVCC    ?= nvcc
ARCH    ?= sm_90a
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)
LIBDIR  ?= ../booltn/lib

LIBS = $(LIBDIR)/libbooltn.so $(LIBDIR)/libcount8.so

all: $(LIBS)

$(LIBDIR)/libbooltn.so: booltn.cu tc_gemm.cuh bgemm_sm90.cuh wgmma_b1.inc | $(LIBDIR)
	$(NVCC) $(NVFLAGS) -shared -Xcompiler -fPIC -o $@ booltn.cu

$(LIBDIR)/libcount8.so: count8.cu bgemm_sm90.cuh wgmma_u8.inc | $(LIBDIR)
	$(NVCC) $(NVFLAGS) -shared -Xcompiler -fPIC -o $@ count8.cu

bgemm_test: bgemm_test.cu bgemm_sm90.cuh
	$(NVCC) $(NVFLAGS) -o $@ bgemm_test.cu

inc:
	python3 gen_wgmma.py && python3 gen_wgmma_u8.py

$(LIBDIR):
	mkdir -p $@

clean:
	rm -f $(LIBS) bgemm_test

.PHONY: all inc clean
