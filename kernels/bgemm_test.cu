// Correctness test + benchmark for the Hopper boolean GEMM.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bgemm_sm90.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    CK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&fn, cudaEnableDefault, &q));
  }
  return fn;
}

// 3D map over a bit-packed [batch][rows][row_bytes] tensor, box = box_bytes x box_rows x 1
static CUtensorMap make_map(void* ptr, int row_bytes, int rows, int batch, int box_rows, int box_bytes = 128,
                            bool swizzle = true) {
  CUtensorMap m;
  cuuint64_t dims[3] = {(cuuint64_t)row_bytes, (cuuint64_t)rows, (cuuint64_t)batch};
  cuuint64_t strides[2] = {(cuuint64_t)row_bytes, (cuuint64_t)row_bytes * rows};
  cuuint32_t box[3] = {(cuuint32_t)box_bytes, (cuuint32_t)box_rows, 1};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tensor map encode failed %d\n", (int)r); exit(1); }
  return m;
}

// naive reference: one thread per output bit-word (32 columns)
__global__ void ref_kernel(const uint32_t* A, const uint32_t* B, uint32_t* C, int M, int N, int Kw, int ldc, int batch) {
  long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long total = (long long)batch * M * ldc;
  if (idx >= total) return;
  int b = idx / ((long long)M * ldc);
  int r = idx % ((long long)M * ldc);
  int m = r / ldc, q = r % ldc;
  const uint32_t* a = A + ((long long)b * M + m) * Kw;
  uint32_t w = 0;
  for (int t = 0; t < 32; ++t) {
    int n = 32 * q + t;
    if (n >= N) break;
    const uint32_t* bb = B + ((long long)b * N + n) * Kw;
    uint32_t acc = 0;
    for (int k = 0; k < Kw; ++k) acc |= a[k] & bb[k];
    w |= (acc != 0u) << t;
  }
  C[idx] = w;
}
__global__ void ref_counts(const uint32_t* A, const uint32_t* B, int* C, int M, int N, int Kw, int batch) {
  long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (idx >= (long long)batch * M * N) return;
  int b = idx / ((long long)M * N);
  int r = idx % ((long long)M * N);
  int m = r / N, n = r % N;
  const uint32_t* a = A + ((long long)b * M + m) * Kw;
  const uint32_t* bb = B + ((long long)b * N + n) * Kw;
  int c = 0;
  for (int k = 0; k < Kw; ++k) c += __popc(a[k] & bb[k]);
  C[idx] = c;
}

__global__ void fill_random(uint32_t* p, long long n, uint32_t seed, int density_shift) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  auto h = [](uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; };
  uint32_t v = h((uint32_t)i * 2654435761u ^ seed);
  // sparsify: AND together several hashes so the OR-reduction is not trivially all-ones
  for (int s = 0; s < density_shift; ++s) v &= h(v + 0x9e3779b9u * (s + 1));
  p[i] = v;
}

static constexpr int STAGES = 4;

struct Problem {
  int M, N, K, batch;
};

static int g_mode = 0;
static double run(Problem pr, bool check, bool counts, int reps) {
  int Kbytes = ((pr.K + 127) / 128) * 16;
  int Kw = Kbytes / 4;
  size_t aw = (size_t)pr.batch * pr.M * Kw, bw = (size_t)pr.batch * pr.N * Kw;
  int ldc = (pr.N + 31) / 32;
  if (!getenv("BGEMM_NO_TMA_STORE")) ldc = ((pr.N + 127) / 128) * 4;  // pad row pitch to 16 bytes for TMA store
  size_t cw = (size_t)pr.batch * pr.M * ldc;
  uint32_t *dA, *dB, *dC, *dR;
  int *dCnt = nullptr, *dRc = nullptr;
  CK(cudaMalloc(&dA, aw * 4));
  CK(cudaMalloc(&dB, bw * 4));
  CK(cudaMalloc(&dC, cw * 4));
  // density: choose so that P(OR=1) is around 1/2 for the given K: p_a*p_b*K ~ 0.7
  int ds = 0;
  while (ds < 8 && pow(2.0, -2.0 * (ds + 1)) * pr.K > 0.7) ++ds;
  fill_random<<<(aw + 255) / 256, 256>>>(dA, aw, 1234, ds);
  fill_random<<<(bw + 255) / 256, 256>>>(dB, bw, 5678, ds);
  CK(cudaGetLastError());

  CUtensorMap mA = make_map(dA, Kbytes, pr.M, pr.batch, bgemm::BM);
  CUtensorMap mB = make_map(dB, Kbytes, pr.N, pr.batch, bgemm::BN);
  bgemm::Params p;
  p.M = pr.M; p.N = pr.N; p.Kbytes = Kbytes; p.batch = pr.batch;
  p.C = dC; p.counts = nullptr; p.ldc_words = ldc; p.c_batch_stride = (long long)pr.M * ldc;
  if (counts) {
    CK(cudaMalloc(&dCnt, (size_t)pr.batch * pr.M * pr.N * 4));
    p.counts = dCnt; p.c_batch_stride = (long long)pr.M * pr.N;
  }
  p.tiles_m = (pr.M + bgemm::BM - 1) / bgemm::BM;
  p.tiles_n = (pr.N + bgemm::BN - 1) / bgemm::BN;
  p.group_m = getenv("BGEMM_GROUP_M") ? atoi(getenv("BGEMM_GROUP_M")) : 8;
  p.use_tma_store = (ldc % 4 == 0) && !counts && !getenv("BGEMM_DIRECT_STORE");
  p.trace = nullptr;
  long long* dTrace = nullptr;
  if (getenv("BGEMM_TRACE")) { CK(cudaMalloc(&dTrace, 3 * 64 * 4 * 8)); CK(cudaMemset(dTrace, 0, 3 * 64 * 4 * 8)); p.trace = dTrace; }
  CUtensorMap mC = make_map(dC, ldc * 4, pr.M, pr.batch, 64, 32, false);
  int ntiles = p.tiles_m * p.tiles_n * pr.batch;
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  int grid = ntiles < nsm ? ntiles : nsm;
  size_t smem = sizeof(bgemm::SharedStorage<STAGES>) + 1024;
  bool stagger = getenv("BGEMM_STAGGER") && atoi(getenv("BGEMM_STAGGER"));
  auto kern = stagger ? (g_mode == 0 ? bgemm::bgemm_kernel<STAGES, 0, true> : bgemm::bgemm_kernel<STAGES, 3, true>) : g_mode == 0 ? bgemm::bgemm_kernel<STAGES, 0> : g_mode == 1 ? bgemm::bgemm_kernel<STAGES, 1> : g_mode == 2 ? bgemm::bgemm_kernel<STAGES, 2> : g_mode == 3 ? bgemm::bgemm_kernel<STAGES, 3> : g_mode == 4 ? bgemm::bgemm_kernel<STAGES, 4> : bgemm::bgemm_kernel<STAGES, 5>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  kern<<<grid, 384, smem>>>(mA, mB, mC, p);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  if (check) {
    bool ok = true;
    if (counts) {
      CK(cudaMalloc(&dRc, (size_t)pr.batch * pr.M * pr.N * 4));
      long long tot = (long long)pr.batch * pr.M * pr.N;
      ref_counts<<<(tot + 255) / 256, 256>>>(dA, dB, dRc, pr.M, pr.N, Kw, pr.batch);
      std::vector<int> h1(tot), h2(tot);
      CK(cudaMemcpy(h1.data(), dCnt, tot * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(h2.data(), dRc, tot * 4, cudaMemcpyDeviceToHost));
      long long bad = 0, firstbad = -1;
      for (long long i = 0; i < tot; ++i) if (h1[i] != h2[i]) { if (firstbad < 0) firstbad = i; ++bad; }
      if (bad) { ok = false; printf("  count mismatches: %lld / %lld (first at %lld: got %d want %d)\n", bad, tot, firstbad, h1[firstbad], h2[firstbad]); }
      cudaFree(dRc);
    } else {
      CK(cudaMalloc(&dR, cw * 4));
      ref_kernel<<<(cw + 255) / 256, 256>>>(dA, dB, dR, pr.M, pr.N, Kw, ldc, pr.batch);
      std::vector<uint32_t> h1(cw), h2(cw);
      CK(cudaMemcpy(h1.data(), dC, cw * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(h2.data(), dR, cw * 4, cudaMemcpyDeviceToHost));
      long long bad = 0, ones = 0;
      for (size_t i = 0; i < cw; ++i) { if (h1[i] != h2[i]) ++bad; ones += __builtin_popcount(h2[i]); }
      if (bad) { ok = false; printf("  bit-word mismatches: %lld / %zu\n", bad, cw); }
      printf("  density of ones in C: %.3f\n", (double)ones / ((double)pr.batch * pr.M * pr.N));
      cudaFree(dR);
    }
    printf("  check %s\n", ok ? "PASSED" : "FAILED");
  }

  if (dTrace) {
    std::vector<long long> h(3 * 64 * 4);
    CK(cudaMemcpy(h.data(), dTrace, h.size() * 8, cudaMemcpyDeviceToHost));
    long long t0 = h[0];
    printf("tile | WG0: start full mma_done epi_done | WG1: start full mma_done epi_done | producer issue\n");
    for (int i = 0; i < 12; ++i) {
      printf("%3d |", i);
      for (int w = 0; w < 2; ++w) for (int e = 0; e < 4; ++e) printf(" %7lld", h[(w * 64 + i) * 4 + e] - t0);
      printf(" | %7lld\n", h[2 * 64 * 4 + i] - t0);
    }
    p.trace = nullptr;
  }
  double tops = 0;
  if (reps > 0) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) kern<<<grid, 384, smem>>>(mA, mB, mC, p);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double ops = 2.0 * pr.M * (double)pr.N * pr.K * pr.batch;
    tops = ops * reps / (ms * 1e-3) / 1e12;
    printf("  M=%d N=%d K=%d batch=%d: %.3f ms/iter, %.1f TOPS\n", pr.M, pr.N, pr.K, pr.batch, ms / reps, tops);
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC); if (dCnt) cudaFree(dCnt);
  return tops;
}

int main(int argc, char** argv) {
  if (getenv("BGEMM_MODE")) g_mode = atoi(getenv("BGEMM_MODE"));
  if (argc >= 5) {
    Problem pr{atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    bool check = argc >= 6 ? atoi(argv[5]) : 1;
    int reps = argc >= 7 ? atoi(argv[6]) : 20;
    run(pr, check, false, reps);
    return 0;
  }
  printf("== correctness (counts) ==\n");
  run({128, 256, 1024, 1}, true, true, 0);
  run({256, 512, 4096, 2}, true, true, 0);
  run({200, 300, 1000, 3}, true, true, 0);
  printf("== correctness (bits) ==\n");
  run({128, 256, 1024, 1}, true, false, 0);
  run({1024, 1024, 8192, 2}, true, false, 0);
  run({200, 300, 1000, 3}, true, false, 0);
  run({64, 40, 256, 5}, true, false, 0);
  printf("== performance ==\n");
  int sizes[][4] = {{4096, 4096, 4096, 1}, {8192, 8192, 8192, 1}, {16384, 16384, 16384, 1},
                    {16384, 16384, 65536, 1}, {32768, 32768, 32768, 1}};
  for (auto& s : sizes) run({s[0], s[1], s[2], s[3]}, false, false, 10);
  return 0;
}
