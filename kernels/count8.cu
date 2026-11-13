// count8: exact model counting by tensor-network contraction, one residue class mod a small prime p per pass.
//
// Tensors hold residues mod p (p <= 251) as one byte per entry; a tensor with r binary labels has 2^r bytes,
// label at position q contributes 2^q to the linear element index (position 0 = fastest). Exact counts are
// reconstructed on the host by CRT over several primes. The prime is read from device memory (*pp), so one
// compiled plan / CUDA graph serves every prime.
//
// Kernels:
//   perm8        : label permutation with optional sum (mod p) over dropped labels
//   tc8          : batched GEMM on int8 tensor cores (wgmma u8.u8 -> s32), epilogue mod p -> u8
//   cc8          : batched GEMM on CUDA cores (dp4a), optional split-K via int32 atomics + finalize
//   small8_batch : many tiny contractions (arbitrary layouts), one block each, one launch per tree level
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bgemm_sm90.cuh"
#include "wgmma_u8.inc"

#define C8_CHECK(x)                                                                              \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "count8 CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      return -1;                                                                                 \
    }                                                                                            \
  } while (0)

namespace c8 {
using namespace bgemm;

static cudaStream_t g_stream = 0;
static std::vector<cudaStream_t> g_streams;
static int g_nsm = 0;

// x mod p for 0 <= x < 2^31 using a Barrett multiplier m = floor(2^32 / p)
__device__ __forceinline__ uint32_t modp(uint32_t x, uint32_t p, uint32_t m) {
  uint32_t q = __umulhi(x, m);
  uint32_t r = x - q * p;
  return r >= p ? r - p : r;
}

// ------------------------------- index maps (element index -> source offset) ------------------
struct IdxMap8 {
  long long T[5][256];  // chunks of 8 index bits (rank <= 40)
};
static void build_map8(IdxMap8& m, const int* src_pos, int rout) {
  memset(&m, 0, sizeof(m));
  for (int c = 0; c < 5; ++c)
    for (int v = 0; v < 256; ++v) {
      long long x = 0;
      for (int i = 0; i < 8; ++i) {
        int q = 8 * c + i;
        if (q < rout && ((v >> i) & 1) && src_pos[q] >= 0) x += 1ll << src_pos[q];
      }
      m.T[c][v] = x;
    }
}
__device__ __forceinline__ long long map8(const IdxMap8& m, long long y) {
  return m.T[0][y & 255] + m.T[1][(y >> 8) & 255] + m.T[2][(y >> 16) & 255] + m.T[3][(y >> 24) & 255] +
         m.T[4][(y >> 32) & 255];
}

// ------------------------------------------ permutation ----------------------------------------
struct Perm8Params {
  IdxMap8 m;
  long long roff[256];
  int ncomb;
  long long n;  // output elements
};
__global__ void __launch_bounds__(256) perm8_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                                    const int* __restrict__ pp, const __grid_constant__ Perm8Params P) {
  __shared__ IdxMap8 sm;
  __shared__ long long sr[256];
  for (int i = threadIdx.x; i < (int)(sizeof(IdxMap8) / 8); i += blockDim.x)
    reinterpret_cast<long long*>(&sm)[i] = reinterpret_cast<const long long*>(&P.m)[i];
  for (int i = threadIdx.x; i < P.ncomb; i += blockDim.x) sr[i] = P.roff[i];
  __syncthreads();
  const uint32_t p = (uint32_t)*pp, mb = 0xffffffffu / p;
  for (long long y = blockIdx.x * (long long)blockDim.x + threadIdx.x; y < P.n; y += (long long)gridDim.x * blockDim.x) {
    const long long x = map8(sm, y);
    if (P.ncomb == 1) {
      out[y] = in[x];
    } else {
      uint32_t s = 0;
      for (int j = 0; j < P.ncomb; ++j) s += in[x + sr[j]];
      out[y] = (uint8_t)modp(s, p, mb);
    }
  }
}


// ------------------------------ tiled byte permutation (no sums) -------------------------------
// tile = 2^ts elements spanned by the a lowest input positions and the b lowest output positions;
// coalesced loads (input-contiguous) -> smem -> coalesced stores (output-contiguous)
struct Perm8TParams {
  int ts;
  int nouter_bits;
  long long nouter;
  long long out_in[40], out_out[40];   // outer bit -> input / output element offset
  long long ld[2][64];                 // tile element (input order) -> input offset (chunks of 6 bits)
  long long st[2][64];                 // tile element (output order) -> output offset
  int mp[2][64];                       // output-order tile index -> input-order tile index
};

__global__ void __launch_bounds__(256) perm8t_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                                     const __grid_constant__ Perm8TParams P) {
  __shared__ uint8_t tile[4096];
  __shared__ long long s_ld[2][64], s_st[2][64];
  __shared__ int s_mp[2][64];
  for (int i = threadIdx.x; i < 128; i += blockDim.x) {
    (&s_ld[0][0])[i] = (&P.ld[0][0])[i];
    (&s_st[0][0])[i] = (&P.st[0][0])[i];
    (&s_mp[0][0])[i] = (&P.mp[0][0])[i];
  }
  __syncthreads();
  const int nt = 1 << P.ts;
  const int lane = threadIdx.x & 31;
  long long my_in = 0, my_out = 0;
  long long my_in2 = 0, my_out2 = 0;
  if (lane < P.nouter_bits) { my_in = P.out_in[lane]; my_out = P.out_out[lane]; }
  if (lane + 32 < P.nouter_bits) { my_in2 = P.out_in[lane + 32]; my_out2 = P.out_out[lane + 32]; }
  for (long long o = blockIdx.x; o < P.nouter; o += gridDim.x) {
    long long bi = ((o >> lane) & 1) ? my_in : 0, bo = ((o >> lane) & 1) ? my_out : 0;
    if (P.nouter_bits > 32) {
      if ((o >> (lane + 32)) & 1) { bi += my_in2; bo += my_out2; }
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
      bi += __shfl_xor_sync(0xffffffff, bi, d);
      bo += __shfl_xor_sync(0xffffffff, bo, d);
    }
    for (int t = threadIdx.x; t < nt; t += blockDim.x) tile[t] = in[bi + s_ld[0][t & 63] + s_ld[1][t >> 6]];
    __syncthreads();
    for (int t = threadIdx.x; t < nt; t += blockDim.x)
      out[bo + s_st[0][t & 63] + s_st[1][t >> 6]] = tile[s_mp[0][t & 63] + s_mp[1][t >> 6]];
    __syncthreads();
  }
}

// ---------------- CUDA-core "outer" GEMM for small K: B in [b][k][n] layout, 16 outputs/thread -----
// A [b][m][k] (K = 2^kk <= 16 bytes), B [b][k][n] (n fastest, N = 2^nn >= 16), C [b][m][n]
__global__ void __launch_bounds__(256) cco8_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                   uint8_t* __restrict__ C, const int* __restrict__ pp,
                                                   long long batch, int mm, int nn, int kk) {
  const uint32_t p = (uint32_t)*pp, mb = 0xffffffffu / p;
  const int K = 1 << kk;
  const long long N16 = 1ll << (nn - 4);
  const long long total = batch << (mm + nn - 4);
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const long long n16 = idx & (N16 - 1);
    const long long t = idx >> (nn - 4);
    const long long b = t >> mm;
    const uint8_t* arow = A + (t << kk);                       // t = b*M + m
    const uint4* bcol = reinterpret_cast<const uint4*>(B + ((b << kk) << nn)) + n16;
    uint32_t acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0;
    for (int k = 0; k < K; ++k) {
      const uint32_t a = arow[k];
      const uint4 v = __ldg(bcol + ((long long)k << (nn - 4)));
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc[4 * q + 0] += a * (w[q] & 0xff);
        acc[4 * q + 1] += a * ((w[q] >> 8) & 0xff);
        acc[4 * q + 2] += a * ((w[q] >> 16) & 0xff);
        acc[4 * q + 3] += a * (w[q] >> 24);
      }
    }
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      o[q] = modp(acc[4 * q], p, mb) | (modp(acc[4 * q + 1], p, mb) << 8) | (modp(acc[4 * q + 2], p, mb) << 16) |
             (modp(acc[4 * q + 3], p, mb) << 24);
    reinterpret_cast<uint4*>(C + (t << nn))[n16] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// ------------------------------- CUDA-core GEMM (dp4a), split-K --------------------------------
// A [b][m][k], B [b][n][k] (k fastest, K = 2^kk bytes), C [b][m][n]. Each thread: one output element.
// lks > 0: blockIdx.y = K chunk; partial sums are atomically added to the int32 scratch `acc` (finalize later).
__global__ void __launch_bounds__(256) cc8_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                  uint8_t* __restrict__ C, int* __restrict__ acc,
                                                  const int* __restrict__ pp, long long batch, int mm, int nn,
                                                  int kk, int lks) {
  const uint32_t p = (uint32_t)*pp, mb = 0xffffffffu / p;
  const long long M = 1ll << mm, N = 1ll << nn, K = 1ll << kk;
  const long long total = batch << (mm + nn);
  const long long kc = K >> lks;
  const long long kbeg = (long long)blockIdx.y * kc, kend = kbeg + kc;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const long long n = idx & (N - 1);
    const long long t = idx >> nn;
    const long long m = t & (M - 1);
    const long long b = t >> mm;
    const uint8_t* a = A + (((b << mm) + m) << kk);
    const uint8_t* bb = B + (((b << nn) + n) << kk);
    uint32_t s = 0;
    if (K >= 4) {
      const uint32_t* aw = reinterpret_cast<const uint32_t*>(a);
      const uint32_t* bw = reinterpret_cast<const uint32_t*>(bb);
      long long k = kbeg;
      while (k < kend) {
        const long long kstop = min(kend, k + 16384);  // keep the int32 sum < 2^31 (16384 * 250^2 < 2^31)
        for (; k < kstop; k += 4) s = __dp4a(__ldg(aw + (k >> 2)), __ldg(bw + (k >> 2)), s);
        if (k < kend) s = modp(s, p, mb);
      }
    } else {
      for (long long k = kbeg; k < kend; ++k) s += (uint32_t)a[k] * (uint32_t)bb[k];
    }
    if (lks == 0) {
      C[idx] = (uint8_t)modp(s, p, mb);
    } else {
      s = modp(s, p, mb);
      if (s) atomicAdd(acc + idx, (int)s);
    }
  }
}

__global__ void finalize8_kernel(const int* __restrict__ acc, uint8_t* __restrict__ C, const int* __restrict__ pp,
                                 long long n) {
  const uint32_t p = (uint32_t)*pp, mb = 0xffffffffu / p;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    C[i] = (uint8_t)modp((uint32_t)acc[i], p, mb);
}

// --------------------------------- tensor-core GEMM (wgmma u8) ---------------------------------
struct Tc8Params {
  int M, N, Kbytes, batch;
  uint8_t* C;
  long long c_batch_stride;  // bytes
  int tiles_m, tiles_n, group_m;
  const int* pp;
};

template <int BN, int NWG, int STAGES>
struct Tc8Smem {
  alignas(1024) uint8_t A[STAGES][64 * NWG * 128];
  alignas(1024) uint8_t B[STAGES][BN * 128];
  int4 coord[STAGES];
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
};

__device__ __forceinline__ void tc8_tile_coords(const Tc8Params& p, int tile, int& b, int& tm, int& tn) {
  int per_b = p.tiles_m * p.tiles_n;
  b = tile / per_b;
  int r = tile - b * per_b;
  int gsz = p.group_m * p.tiles_n;
  int g = r / gsz;
  int first = g * p.group_m;
  int gm = min(p.tiles_m - first, p.group_m);
  int rr = r - g * gsz;
  tm = first + rr % gm;
  tn = rr / gm;
}

template <int BN, int NWG, int STAGES, bool SMR, int MINB>
__global__ void __launch_bounds__(128 * (NWG + 1), MINB)
    tc8_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, Tc8Params p) {
  constexpr int BM = 64 * NWG;
  constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128;
  constexpr int RED_EVERY = 128;  // stages between mod-p reductions: 128*128*250^2 < 2^31
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  Tc8Smem<BN, NWG, STAGES>& S = *reinterpret_cast<Tc8Smem<BN, NWG, STAGES>*>(base);
  const int wg = threadIdx.x / 128;
  const int KT = (p.Kbytes + 127) / 128;
  const int ntiles = p.tiles_m * p.tiles_n * p.batch;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&S.full[s], 1);
      mbar_init(&S.empty[s], 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (wg == 0) {
    if constexpr (SMR) setmaxnreg_dec<40>();
    if (threadIdx.x == 0) {
      int it = 0;
      for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int b, tm, tn;
        tc8_tile_coords(p, tile, b, tm, tn);
        for (int kt = 0; kt < KT; ++kt, ++it) {
          int s = it % STAGES;
          mbar_wait(&S.empty[s], ((it / STAGES) & 1) ^ 1);
          if (kt == 0) S.coord[s] = make_int4(b, tm, tn, 0);
          mbar_expect_tx(&S.full[s], A_BYTES + B_BYTES);
          tma_load_3d(S.A[s], &tmA, &S.full[s], kt * 128, tm * BM, b);
          tma_load_3d(S.B[s], &tmB, &S.full[s], kt * 128, tn * BN, b);
        }
      }
    }
  } else {
    if constexpr (SMR) setmaxnreg_inc<232>();
    const uint32_t pr = (uint32_t)*p.pp, mbar = 0xffffffffu / pr;
    const int cw = wg - 1;
    int it = 0;
    int d[BN / 2];
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) d[i] = 0;
    const int my_tiles = ntiles > (int)blockIdx.x ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    for (int lt = 0; lt < my_tiles; ++lt) {
      int b = 0, tm = 0, tn = 0;
      bool first = true;
      fence_operands(d);
      for (int kt = 0; kt < KT; ++kt, ++it) {
        int s = it % STAGES;
        mbar_wait(&S.full[s], (it / STAGES) & 1);
        if (kt == 0) {
          int4 c = S.coord[s];
          b = c.x; tm = c.y; tn = c.z;
        }
        wg_fence();
        fence_operands(d);
        const uint8_t* a = S.A[s] + cw * 64 * 128;
        const uint8_t* bb = S.B[s];
        const int nks = min(4, (p.Kbytes - kt * 128 + 31) / 32);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if (k < nks) {
            WgmmaU8<BN>::run(d, desc_sw128(a + 32 * k), desc_sw128(bb + 32 * k), first ? 0 : 1);
            first = false;
          }
        wg_commit();
        fence_operands(d);
        wg_wait<1>();
        if (kt > 0 && (threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
        if (kt % RED_EVERY == RED_EVERY - 1 && kt + 1 < KT) {  // periodic reduction for very long K
          wg_wait<0>();
          fence_operands(d);
#pragma unroll
          for (int i = 0; i < BN / 2; ++i) d[i] = (int)modp((uint32_t)d[i], pr, mbar);
          fence_operands(d);
        }
      }
      wg_wait<0>();
      fence_operands(d);
      if ((threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
      // epilogue: d[4j+e] -> row r0 (+8 for e>=2), col 8j + 2*(lane&3) + (e&1); store 2-byte pairs
      const int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) & 3;
      const int r0 = tm * BM + cw * 64 + 16 * warp + lane / 4;
      uint8_t* Cb = p.C + (long long)b * p.c_batch_stride;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int col = tn * BN + 8 * j + 2 * (lane & 3);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = r0 + 8 * h;
          if (row >= p.M) continue;
          const uint32_t v0 = modp((uint32_t)d[4 * j + 2 * h], pr, mbar);
          const uint32_t v1 = modp((uint32_t)d[4 * j + 2 * h + 1], pr, mbar);
          uint8_t* dst = Cb + (long long)row * p.N + col;
          if (col + 1 < p.N) {
            *reinterpret_cast<uint16_t*>(dst) = (uint16_t)(v0 | (v1 << 8));
          } else if (col < p.N) {
            dst[0] = (uint8_t)v0;
          }
        }
      }
    }
  }
}

// ------------------------------- batched small contractions ------------------------------------
struct SmallOp8 {
  const uint8_t* a;
  const uint8_t* b;
  uint8_t* c;
  int rc, nk;
  signed char a_pos[24], b_pos[24];
  signed char ka[8], kb[8];
};
__global__ void __launch_bounds__(256) small8_kernel(const SmallOp8* __restrict__ ops, const int* __restrict__ pp) {
  __shared__ SmallOp8 s;
  __shared__ long long koa[256], kob[256];
  if (threadIdx.x == 0) s = ops[blockIdx.x];
  __syncthreads();
  const int ncomb = 1 << s.nk;
  for (int j = threadIdx.x; j < ncomb; j += blockDim.x) {
    long long xa = 0, xb = 0;
    for (int i = 0; i < s.nk; ++i)
      if ((j >> i) & 1) {
        if (s.ka[i] >= 0) xa += 1ll << s.ka[i];
        if (s.kb[i] >= 0) xb += 1ll << s.kb[i];
      }
    koa[j] = xa;
    kob[j] = xb;
  }
  __syncthreads();
  const uint32_t p = (uint32_t)*pp, mb = 0xffffffffu / p;
  const long long n = 1ll << s.rc;
  for (long long y = threadIdx.x; y < n; y += blockDim.x) {
    long long xa = 0, xb = 0;
    for (int q = 0; q < s.rc; ++q)
      if ((y >> q) & 1) {
        if (s.a_pos[q] >= 0) xa += 1ll << s.a_pos[q];
        if (s.b_pos[q] >= 0) xb += 1ll << s.b_pos[q];
      }
    uint32_t acc = 0;
    for (int j = 0; j < ncomb; ++j) acc += (uint32_t)s.a[xa + koa[j]] * (uint32_t)s.b[xb + kob[j]];  // <= 256*250^2
    s.c[y] = (uint8_t)modp(acc, p, mb);
  }
}

// accumulate a scalar (rank-0 root, one byte) into an int32 running sum mod p (sum over slices)
__global__ void add_accum8_kernel(int* acc, const uint8_t* root, const int* pp) {
  if (threadIdx.x == 0) acc[0] = (int)(((uint32_t)acc[0] + root[0]) % (uint32_t)*pp);
}

// ------------------------------------- plans ---------------------------------------------------
typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PFN_encodeTiled g_encode = nullptr;
static int make_map(CUtensorMap* m, const void* ptr, long long row_bytes, long long rows, long long batch, int box_rows) {
  cuuint64_t dims[3] = {(cuuint64_t)row_bytes, (cuuint64_t)rows, (cuuint64_t)batch};
  cuuint64_t strides[2] = {(cuuint64_t)row_bytes, (cuuint64_t)(row_bytes * rows)};
  cuuint32_t box[3] = {128, (cuuint32_t)box_rows, 1};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult r = g_encode(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(ptr), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : -1;
}

typedef void (*Tc8KernelPtr)(CUtensorMap, CUtensorMap, Tc8Params);
struct Tc8Launch {
  Tc8KernelPtr kern;
  CUtensorMap mA, mB;
  Tc8Params p;
  int grid, threads;
  size_t smem;
};

template <int BN, int NWG, int STAGES, bool SMR, int MINB>
static int prepare_tc8(Tc8Launch& L, const uint8_t* A, const uint8_t* B, uint8_t* C, const int* pp, int bb, int mm,
                       int nn, int kk) {
  constexpr int BM = 64 * NWG;
  long long Kbytes = 1ll << kk;
  if (make_map(&L.mA, A, Kbytes, 1ll << mm, 1ll << bb, BM)) return -1;
  if (make_map(&L.mB, B, Kbytes, 1ll << nn, 1ll << bb, BN)) return -1;
  Tc8Params& p = L.p;
  p.M = 1 << mm;
  p.N = 1 << nn;
  p.Kbytes = (int)Kbytes;
  p.batch = 1 << bb;
  p.C = C;
  p.c_batch_stride = (long long)p.M * p.N;
  p.tiles_m = (p.M + BM - 1) / BM;
  p.tiles_n = (p.N + BN - 1) / BN;
  p.group_m = 8;
  p.pp = pp;
  long long ntiles = (long long)p.tiles_m * p.tiles_n * p.batch;
  if (ntiles >= (1ll << 31)) return -2;
  auto kern = tc8_kernel<BN, NWG, STAGES, SMR, MINB>;
  L.kern = (Tc8KernelPtr)kern;
  L.smem = sizeof(Tc8Smem<BN, NWG, STAGES>) + 1024;
  L.threads = 128 * (NWG + 1);
  static bool configured = false;
  static int occ = 1;
  if (!configured) {
    C8_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smem));
    C8_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, L.threads, L.smem));
    if (occ < 1) occ = 1;
    configured = true;
  }
  L.grid = (int)std::min<long long>(ntiles, (long long)g_nsm * occ);
  return 0;
}


static bool plan_perm8t(Perm8TParams& P, const int* src, int r) {
  if (r < 8 || r > 40) return false;
  std::vector<int> inv(r);
  for (int q = 0; q < r; ++q) inv[src[q]] = q;
  int a = 0, b = 0;
  std::vector<int> S;
  auto build = [&]() {
    std::vector<char> in_s(r, 0);
    for (int q = 0; q < a; ++q) in_s[q] = 1;
    for (int q = 0; q < b; ++q) in_s[src[q]] = 1;
    S.clear();
    for (int q = 0; q < r; ++q) if (in_s[q]) S.push_back(q);
  };
  build();
  for (;;) {
    bool grown = false;
    for (int side = 0; side < 2; ++side) {
      int& v = (side == 0) == (a <= b) ? a : b;
      if (v >= r) continue;
      ++v;
      build();
      if ((int)S.size() > 12) { --v; build(); } else grown = true;
    }
    if (!grown) break;
  }
  std::vector<int> Tl;  // tile input order: positions 0..a-1 then rest of S
  for (int q = 0; q < a; ++q) Tl.push_back(q);
  for (int q : S) if (q >= a) Tl.push_back(q);
  const int ts = (int)Tl.size();
  P.ts = ts;
  std::vector<int> To;  // tile output order: src[0..b-1] then rest by output position
  for (int q = 0; q < b; ++q) To.push_back(src[q]);
  {
    std::vector<int> rest;
    for (int q : Tl) if (inv[q] >= b) rest.push_back(q);
    std::sort(rest.begin(), rest.end(), [&](int x, int y) { return inv[x] < inv[y]; });
    for (int q : rest) To.push_back(q);
  }
  std::vector<int> tl_of(r, -1);
  for (int k = 0; k < ts; ++k) tl_of[Tl[k]] = k;
  std::vector<int> Ol;
  for (int q = 0; q < r; ++q) if (tl_of[q] < 0) Ol.push_back(q);
  P.nouter_bits = (int)Ol.size();
  if (P.nouter_bits > 40) return false;
  P.nouter = 1ll << P.nouter_bits;
  memset(P.out_in, 0, sizeof(P.out_in)); memset(P.out_out, 0, sizeof(P.out_out));
  for (int j = 0; j < P.nouter_bits; ++j) { P.out_in[j] = 1ll << Ol[j]; P.out_out[j] = 1ll << inv[Ol[j]]; }
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 64; ++v) {
      long long xi = 0, xo = 0;
      int mpv = 0;
      for (int t = 0; t < 6; ++t) {
        int k = 6 * c + t;
        if (k < ts && ((v >> t) & 1)) {
          xi += 1ll << Tl[k];
          xo += 1ll << inv[To[k]];
          mpv += 1 << tl_of[To[k]];
        }
      }
      P.ld[c][v] = xi; P.st[c][v] = xo; P.mp[c][v] = mpv;
    }
  return true;
}

static long long grid_for(long long work, int per_block = 256) {
  long long g = (work + per_block - 1) / per_block;
  return std::max<long long>(1, std::min<long long>(g, (long long)g_nsm * 32));
}

struct Op {
  int kind = 0;  // 1 perm8, 2 cc8, 3 tc8, 4 small batch, 5 add_accum
  const uint8_t* a = nullptr;
  const uint8_t* b = nullptr;
  uint8_t* c = nullptr;
  int* acc = nullptr;
  const int* pp = nullptr;
  std::unique_ptr<Perm8Params> perm;
  std::unique_ptr<Perm8TParams> permt;
  Tc8Launch tc;
  long long grid = 1;
  int bb = 0, mm = 0, nn = 0, kk = 0, lks = 0;
  long long nout = 0;
  int launch(cudaStream_t st) const {
    switch (kind) {
      case 1: perm8_kernel<<<(int)grid, 256, 0, st>>>(a, c, pp, *perm); break;
      case 2: {
        if (lks) C8_CHECK(cudaMemsetAsync(acc, 0, nout * 4, st));
        dim3 g((unsigned)grid, 1u << lks);
        cc8_kernel<<<g, 256, 0, st>>>(a, b, c, acc, pp, 1ll << bb, mm, nn, kk, lks);
        if (lks) finalize8_kernel<<<(int)grid_for(nout), 256, 0, st>>>(acc, c, pp, nout);
        break;
      }
      case 3: tc.kern<<<tc.grid, tc.threads, tc.smem, st>>>(tc.mA, tc.mB, tc.p); break;
      case 4: small8_kernel<<<(int)grid, 256, 0, st>>>((const SmallOp8*)a, pp); break;
      case 5: add_accum8_kernel<<<1, 32, 0, st>>>(acc, a, pp); break;
      case 6: perm8t_kernel<<<(int)grid, 256, 0, st>>>(a, c, *permt); break;
      case 7: cco8_kernel<<<(int)grid, 256, 0, st>>>(a, b, c, pp, 1ll << bb, mm, nn, kk); break;
      default: return -9;
    }
    C8_CHECK(cudaGetLastError());
    return 0;
  }
};

struct Plan {
  std::vector<Op> ops;
  std::vector<void*> dev_bufs;
  cudaGraphExec_t exec = nullptr;
};

static cudaStream_t pick_stream(int sidx) { return sidx < 0 || sidx >= (int)g_streams.size() ? g_stream : g_streams[sidx]; }

}  // namespace c8

using namespace c8;
extern "C" {

int c8_init() {
  C8_CHECK(cudaFree(0));
  C8_CHECK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDriverEntryPointQueryResult q;
  C8_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_encode, cudaEnableDefault, &q));
  C8_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  return 0;
}
int c8_make_streams(int n) {
  while ((int)g_streams.size() < n) {
    cudaStream_t st;
    C8_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    g_streams.push_back(st);
  }
  return (int)g_streams.size();
}
void* c8_malloc(size_t bytes) {
  void* p = nullptr;
  bytes = (bytes + 255) & ~(size_t)255;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  return p;
}
int c8_free(void* p) {
  C8_CHECK(cudaFree(p));
  return 0;
}
int c8_h2d(void* dst, const void* src, size_t bytes, int sidx) {
  C8_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, pick_stream(sidx)));
  return 0;
}
int c8_d2h(void* dst, const void* src, size_t bytes, int sidx) {
  cudaStream_t st = pick_stream(sidx);
  C8_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st));
  C8_CHECK(cudaStreamSynchronize(st));
  return 0;
}
int c8_memset(void* p, int v, size_t bytes, int sidx) {
  C8_CHECK(cudaMemsetAsync(p, v, bytes, pick_stream(sidx)));
  return 0;
}
int c8_sync_all() {
  C8_CHECK(cudaDeviceSynchronize());
  return 0;
}
void* c8_event_create() {
  cudaEvent_t e;
  if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
  return (void*)e;
}
int c8_event_record(void* e, int sidx) {
  C8_CHECK(cudaEventRecord((cudaEvent_t)e, pick_stream(sidx)));
  return 0;
}
float c8_event_elapsed(void* e0, void* e1) {
  float ms = -1;
  cudaEventSynchronize((cudaEvent_t)e1);
  cudaEventElapsedTime(&ms, (cudaEvent_t)e0, (cudaEvent_t)e1);
  return ms;
}

void* c8_plan_new() { return new Plan(); }
int c8_plan_free(void* h) {
  Plan* P = (Plan*)h;
  for (void* q : P->dev_bufs) cudaFree(q);
  if (P->exec) cudaGraphExecDestroy(P->exec);
  delete P;
  return 0;
}
// out[y] = sum over dropped-label combos of in[x(y)] (mod p); src_pos[q] = input position of output label q
int c8_plan_permute(void* h, const uint8_t* in, uint8_t* out, int rout, const int* src_pos, const int* red_pos, int nred,
                    const int* pp) {
  if (nred > 8 || rout > 40) return -3;
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.a = in;
  op.c = out;
  op.pp = pp;
  if (nred == 0) {
    op.permt.reset(new Perm8TParams);
    if (plan_perm8t(*op.permt, src_pos, rout)) {
      op.kind = 6;
      op.grid = std::min<long long>(op.permt->nouter, (long long)g_nsm * 16);
      return 0;
    }
    op.permt.reset();
  }
  op.kind = 1;
  op.perm.reset(new Perm8Params);
  build_map8(op.perm->m, src_pos, rout);
  op.perm->ncomb = 1 << nred;
  for (int j = 0; j < op.perm->ncomb; ++j) {
    long long x = 0;
    for (int i = 0; i < nred; ++i)
      if ((j >> i) & 1) x += 1ll << red_pos[i];
    op.perm->roff[j] = x;
  }
  op.perm->n = 1ll << rout;
  op.grid = grid_for(op.perm->n);
  return 0;
}
// C[b][m][n] = sum_k A[b][m][k] B[b][n][k] mod p on CUDA cores; scratch (int32, >= C elements) for split-K
int c8_plan_cc(void* h, const uint8_t* A, const uint8_t* B, uint8_t* C, int* scratch, int bb, int mm, int nn, int kk,
               const int* pp) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 2;
  op.a = A; op.b = B; op.c = C; op.acc = scratch; op.pp = pp;
  op.bb = bb; op.mm = mm; op.nn = nn; op.kk = kk;
  op.nout = 1ll << (bb + mm + nn);
  op.lks = 0;
  if (scratch)
    while ((op.nout << op.lks) < (1ll << 17) && kk - op.lks > 8 && op.lks < 12) ++op.lks;
  op.grid = grid_for(op.nout);
  return 0;
}
// tensor cores: requires kk >= 4 (16-byte rows); A [b][m][k], B [b][n][k], C [b][m][n]
int c8_plan_tc(void* h, const uint8_t* A, const uint8_t* B, uint8_t* C, int bb, int mm, int nn, int kk, const int* pp) {
  if (kk < 4 || nn < 1 || bb > 30) return -3;
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 3;
  int bn = nn >= 8 ? 256 : (nn >= 7 ? 128 : 64);
  bool two = mm >= 7;
  int rc;
  if (bn == 256) rc = two ? prepare_tc8<256, 2, 4, true, 1>(op.tc, A, B, C, pp, bb, mm, nn, kk)
                          : prepare_tc8<256, 1, 4, true, 1>(op.tc, A, B, C, pp, bb, mm, nn, kk);
  else if (bn == 128) rc = two ? prepare_tc8<128, 2, 6, true, 1>(op.tc, A, B, C, pp, bb, mm, nn, kk)
                               : prepare_tc8<128, 1, 4, false, 2>(op.tc, A, B, C, pp, bb, mm, nn, kk);
  else rc = prepare_tc8<64, 1, 4, false, 3>(op.tc, A, B, C, pp, bb, mm, nn, kk);
  if (rc) P->ops.pop_back();
  return rc;
}
// n small contractions in one launch; flat arrays: ptrs[3n], rc[n], nk[n], apos[24n], bpos[24n], ka[8n], kb[8n]
int c8_plan_small_batch(void* h, int n, void** ptrs, const int* rc, const int* nk, const int* apos, const int* bpos,
                        const int* ka, const int* kb, const int* pp) {
  Plan* P = (Plan*)h;
  std::vector<SmallOp8> v(n);
  for (int i = 0; i < n; ++i) {
    SmallOp8& o = v[i];
    o.a = (const uint8_t*)ptrs[3 * i];
    o.b = (const uint8_t*)ptrs[3 * i + 1];
    o.c = (uint8_t*)ptrs[3 * i + 2];
    o.rc = rc[i];
    o.nk = nk[i];
    if (o.rc > 24 || o.nk > 8) return -3;
    for (int q = 0; q < 24; ++q) { o.a_pos[q] = (signed char)apos[24 * i + q]; o.b_pos[q] = (signed char)bpos[24 * i + q]; }
    for (int q = 0; q < 8; ++q) { o.ka[q] = (signed char)ka[8 * i + q]; o.kb[q] = (signed char)kb[8 * i + q]; }
  }
  void* d = nullptr;
  C8_CHECK(cudaMalloc(&d, sizeof(SmallOp8) * n));
  C8_CHECK(cudaMemcpy(d, v.data(), sizeof(SmallOp8) * n, cudaMemcpyHostToDevice));
  P->dev_bufs.push_back(d);
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 4;
  op.a = (const uint8_t*)d;
  op.pp = pp;
  op.grid = n;
  return 0;
}
// small-K CUDA-core GEMM: A [b][m][k], B [b][k][n] (n fastest), kk <= 4, nn >= 4
int c8_plan_cco(void* h, const uint8_t* A, const uint8_t* B, uint8_t* C, int bb, int mm, int nn, int kk, const int* pp) {
  if (kk > 4 || nn < 4) return -3;
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 7;
  op.a = A; op.b = B; op.c = C; op.pp = pp;
  op.bb = bb; op.mm = mm; op.nn = nn; op.kk = kk;
  op.grid = grid_for((1ll << (bb + mm + nn - 4)));
  return 0;
}

int c8_plan_add_accum(void* h, int* acc, const uint8_t* root, const int* pp) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 5;
  op.acc = acc;
  op.a = root;
  op.pp = pp;
  return 0;
}
int c8_plan_size(void* h) { return (int)((Plan*)h)->ops.size(); }
int c8_plan_run_range(void* h, int i0, int i1, int sidx) {
  Plan* P = (Plan*)h;
  for (int i = i0; i < i1 && i < (int)P->ops.size(); ++i) {
    int rc = P->ops[i].launch(pick_stream(sidx));
    if (rc) return rc;
  }
  return 0;
}
int c8_plan_run(void* h, int sidx, int use_graph) {
  Plan* P = (Plan*)h;
  if (!use_graph) return c8_plan_run_range(h, 0, (int)P->ops.size(), sidx);
  if (!P->exec) {
    cudaGraph_t g;
    C8_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    for (auto& op : P->ops) {
      int rc = op.launch(g_stream);
      if (rc) { cudaStreamEndCapture(g_stream, &g); return rc; }
    }
    C8_CHECK(cudaStreamEndCapture(g_stream, &g));
    C8_CHECK(cudaGraphInstantiate(&P->exec, g, 0));
    cudaGraphDestroy(g);
  }
  C8_CHECK(cudaGraphLaunch(P->exec, pick_stream(sidx)));
  return 0;
}

}  // extern "C"
