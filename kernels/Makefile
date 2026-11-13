# CUDA kernels of booltn (Hopper only: wgmma / TMA need sm_90a).
#   module load cuda/12.8.0
#   make -C kernels                 -> booltn/lib/libbooltn.so, libcount8.so, libbtmeasure.so   (from the repository root)
#   make -C kernels measure         -> booltn/lib/libbtmeasure.so only (host code, gcc; no CUDA needed)
#   make -C kernels bgemm_test      -> standalone 1-bit GEMM test/benchmark (./bgemm_test M N K B [check] [reps])
#   make -C kernels inc             -> regenerate the wgmma inline-PTX wrappers (*.inc, checked in)
NVCC    ?= nvcc
ARCH    ?= sm_90a
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)
LIBDIR  ?= ../booltn/lib
CC      ?= gcc

LIBS = $(LIBDIR)/libbooltn.so $(LIBDIR)/libcount8.so $(LIBDIR)/libbtmeasure.so

all: $(LIBS)

$(LIBDIR)/libbooltn.so: booltn.cu tc_gemm.cuh bgemm_sm90.cuh wgmma_b1.inc | $(LIBDIR)
	$(NVCC) $(NVFLAGS) -shared -Xcompiler -fPIC -o $@ booltn.cu

$(LIBDIR)/libcount8.so: count8.cu bgemm_sm90.cuh wgmma_u8.inc | $(LIBDIR)
	$(NVCC) $(NVFLAGS) -shared -Xcompiler -fPIC -o $@ count8.cu

$(LIBDIR)/libbtmeasure.so: measure.c | $(LIBDIR)
	$(CC) -O3 -mpopcnt -fPIC -shared -o $@ measure.c -lm

measure: $(LIBDIR)/libbtmeasure.so

bgemm_test: bgemm_test.cu bgemm_sm90.cuh
	$(NVCC) $(NVFLAGS) -o $@ bgemm_test.cu

inc:
	python3 gen_wgmma.py && python3 gen_wgmma_u8.py

$(LIBDIR):
	mkdir -p $@

clean:
	rm -f $(LIBS) bgemm_test

.PHONY: all inc clean measure
