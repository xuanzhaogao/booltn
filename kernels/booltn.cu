// booltn: boolean (OR-AND semiring) tensor-network contraction kernels for Hopper.
//
// Tensors are bit-packed: a tensor with r binary labels is a 2^r-bit array stored in uint32 words,
// label at position p contributes 2^p to the linear bit index (position 0 = fastest).
//
// Kernels (C API at the bottom, called from Python via ctypes):
//   bt_permute         : label permutation with optional OR-reduction of dropped labels
//   bt_gemm_tc         : batched GEMM on 1-bit tensor cores (wgmma b1 and.popc), K >= 128 bits
//   bt_gemm_cc         : batched GEMM on CUDA cores, masked-OR formulation, any K, N >= 32 bits
//   bt_contract_generic: fallback, arbitrary layouts, slow (for tiny contractions)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "tc_gemm.cuh"

#define BT_CHECK(x)                                                                  \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_ != cudaSuccess) {                                                         \
      fprintf(stderr, "booltn CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      return -1;                                                                     \
    }                                                                                \
  } while (0)

namespace bt {

static cudaStream_t g_stream = 0;
static std::vector<cudaStream_t> g_streams;  // extra streams for concurrent plan replays
static int g_nsm = 0;

// --------------------------------------------------------------------------------------------
// index maps: output bit index y -> source bit index x (64-bit), x = sum_c T[c][byte_c(y>>5)] + off[y&31]
// --------------------------------------------------------------------------------------------
struct IdxMap {
  long long T[4][256];
  long long off[32];
};

static void build_map(IdxMap& m, const int* src_pos, int rout) {
  memset(&m, 0, sizeof(m));
  for (int t = 0; t < 32; ++t) {
    long long x = 0;
    for (int p = 0; p < 5 && p < rout; ++p)
      if ((t >> p) & 1 && src_pos[p] >= 0) x += 1ll << src_pos[p];
    m.off[t] = x;
  }
  for (int c = 0; c < 4; ++c)
    for (int v = 0; v < 256; ++v) {
      long long x = 0;
      for (int i = 0; i < 8; ++i) {
        int p = 5 + 8 * c + i;
        if (p < rout && ((v >> i) & 1) && src_pos[p] >= 0) x += 1ll << src_pos[p];
      }
      m.T[c][v] = x;
    }
}

__device__ __forceinline__ long long map_base(const IdxMap& m, long long w) {
  return m.T[0][w & 255] + m.T[1][(w >> 8) & 255] + m.T[2][(w >> 16) & 255] + m.T[3][(w >> 24) & 255];
}
__device__ __forceinline__ uint32_t getbit(const uint32_t* a, long long x) {
  return (__ldg(a + (x >> 5)) >> (x & 31)) & 1u;
}

// ------------------------------------- permutation ------------------------------------------
struct PermParams {
  IdxMap m;
  long long roff[256];
  int ncomb;  // 2^nred
  int rout;
  int mode;   // 0 = word copy, 1 = single-word bit permutation, 2 = general gather
  long long nwords;
};

__global__ void __launch_bounds__(256) perm_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                                   const __grid_constant__ PermParams P) {
  __shared__ IdxMap sm;
  __shared__ long long sroff[256];
  for (int i = threadIdx.x; i < (int)(sizeof(IdxMap) / 8); i += blockDim.x)
    reinterpret_cast<long long*>(&sm)[i] = reinterpret_cast<const long long*>(&P.m)[i];
  for (int i = threadIdx.x; i < P.ncomb; i += blockDim.x) sroff[i] = P.roff[i];
  __syncthreads();
  const int nb = P.rout >= 5 ? 32 : (1 << P.rout);
  for (long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x; w < P.nwords;
       w += (long long)gridDim.x * blockDim.x) {
    const long long base = map_base(sm, w);
    uint32_t r = 0;
    if (P.mode == 0) {
      r = __ldg(in + (base >> 5));
    } else if (P.mode == 1) {
      const uint32_t x = __ldg(in + (base >> 5));
      for (int t = 0; t < nb; ++t) r |= ((x >> (int)sm.off[t]) & 1u) << t;
    } else {
      for (int t = 0; t < nb; ++t) {
        const long long x = base + sm.off[t];
        uint32_t bit = 0;
        for (int j = 0; j < P.ncomb && !bit; ++j) bit = getbit(in, x + sroff[j]);
        r |= bit << t;
      }
    }
    out[w] = r;
  }
}


// ------------------------- fast tiled bit permutation (no reduction) -------------------------
// Tile = 2^ts bits spanned by the a lowest input positions and the b lowest output positions.
// Per tile: coalesced load -> register bit-exchange of e in-word/word labels -> in-word delta swaps
// -> coalesced store. See bt_permute for the host-side planning.
struct Perm2Params {
  int ts;                 // tile bits (<= 16)
  int e;                  // number of exchanged label pairs
  int ein[5];             // in-word positions exchanged (round j uses ein[j])
  int nswap;              // in-word delta swaps
  int sw_d[5];
  uint32_t sw_m[5];
  int nouter_bits;
  long long nouter;
  long long out_in_w[32];   // outer bit j -> input word offset
  long long out_out_w[32];  // outer bit j -> output word offset
  long long ld[2][64];      // tile word j -> input word offset (chunks of 6 bits)
  long long st[2][64];      // output tile word j' -> output word offset
  int gb[2][64];            // group g -> smem_in word (chunks of 6 bits)
  int ex[32];               // exchange index i -> smem_in word offset
  int og[2][64];            // group g -> smem_out word
  int oe[32];               // exchange index i -> smem_out word offset
};

__device__ __forceinline__ int padw(int w) { return w + (w >> 5); }

template <int E>
__global__ void __launch_bounds__(256) perm2_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                                    const __grid_constant__ Perm2Params P) {
  __shared__ uint32_t sin_[2048 + 64], sout[2048 + 64];
  __shared__ long long s_ld[2][64], s_st[2][64];
  __shared__ int s_gb[2][64], s_og[2][64], s_ex[32], s_oe[32];
  for (int i = threadIdx.x; i < 128; i += blockDim.x) {
    (&s_ld[0][0])[i] = (&P.ld[0][0])[i];
    (&s_st[0][0])[i] = (&P.st[0][0])[i];
    (&s_gb[0][0])[i] = (&P.gb[0][0])[i];
    (&s_og[0][0])[i] = (&P.og[0][0])[i];
  }
  if (threadIdx.x < 32) { s_ex[threadIdx.x] = P.ex[threadIdx.x]; s_oe[threadIdx.x] = P.oe[threadIdx.x]; }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int ntw = 1 << (P.ts - 5);
  constexpr int ne = 1 << E;
  // per-lane outer-bit contributions (outer bits <= 32)
  const long long my_in = lane < P.nouter_bits ? P.out_in_w[lane] : 0;
  const long long my_out = lane < P.nouter_bits ? P.out_out_w[lane] : 0;
  const int nswap = P.nswap;
  for (long long o = blockIdx.x; o < P.nouter; o += gridDim.x) {
    // outer base offsets: warp-parallel sum over set bits of o
    long long bi = ((o >> lane) & 1) ? my_in : 0, bo = ((o >> lane) & 1) ? my_out : 0;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
      bi += __shfl_xor_sync(0xffffffff, bi, d);
      bo += __shfl_xor_sync(0xffffffff, bo, d);
    }
    for (int j = threadIdx.x; j < ntw; j += blockDim.x)
      sin_[padw(j)] = __ldg(in + bi + s_ld[0][j & 63] + s_ld[1][j >> 6]);
    __syncthreads();
    // one word per thread; the 2^E words of an exchange group live in 2^E consecutive lanes
    for (int q = threadIdx.x; q < ntw; q += blockDim.x) {
      const int i = q & (ne - 1);
      const int g = q >> E;
      uint32_t x = sin_[padw(s_gb[0][g & 63] + s_gb[1][g >> 6] + s_ex[i])];
#pragma unroll
      for (int j = 0; j < E; ++j) {
        const int c = P.ein[j];
        const int sh = 1 << c;
        const uint32_t m = c == 0 ? 0x55555555u : c == 1 ? 0x33333333u : c == 2 ? 0x0F0F0F0Fu : c == 3 ? 0x00FF00FFu : 0x0000FFFFu;
        const uint32_t other = __shfl_xor_sync(0xffffffff, x, 1 << j);
        if ((i >> j) & 1) {
          x ^= ((other >> sh) ^ x) & m;
        } else {
          x ^= (((x >> sh) ^ other) & m) << sh;
        }
      }
      for (int qq = 0; qq < nswap; ++qq) {
        uint32_t t = ((x >> P.sw_d[qq]) ^ x) & P.sw_m[qq];
        x ^= t ^ (t << P.sw_d[qq]);
      }
      sout[padw(s_og[0][g & 63] + s_og[1][g >> 6] + s_oe[i])] = x;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ntw; j += blockDim.x)
      out[bo + s_st[0][j & 63] + s_st[1][j >> 6]] = sout[padw(j)];
  }
}

__global__ void fill_random_kernel(uint32_t* p, long long n, uint32_t seed, int and_rounds) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    auto h = [](uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; };
    uint32_t v = h((uint32_t)i * 2654435761u ^ seed ^ (uint32_t)(i >> 32) * 0x9e3779b9u);
    for (int s = 0; s < and_rounds; ++s) v &= h(v + 0x9e3779b9u * (s + 1) + seed);
    p[i] = v;
  }
}

// zero-pad rows of 2^kk bits (kk <= 6) to 128-bit rows so the TMA/wgmma path can be used for small K
__global__ void pad_rows_kernel(const uint32_t* __restrict__ in, uint4* __restrict__ out, long long nrows, int kk) {
  for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r < nrows; r += (long long)gridDim.x * blockDim.x) {
    uint4 v = make_uint4(0, 0, 0, 0);
    if (kk == 6) {
      v.x = __ldg(in + 2 * r);
      v.y = __ldg(in + 2 * r + 1);
    } else {
      const long long x = r << kk;
      const uint32_t m = kk == 5 ? 0xffffffffu : ((1u << (1 << kk)) - 1u);
      v.x = (__ldg(in + (x >> 5)) >> (x & 31)) & m;
    }
    out[r] = v;
  }
}

__global__ void or_accum_kernel(uint32_t* flag, const uint32_t* src) {
  if (threadIdx.x == 0 && (src[0] & 1u)) flag[0] = 1u;
}

// --------------------------- CUDA-core masked-OR batched GEMM --------------------------------
// A: [b][m][k] (row = 2^kk bits), B: [b][k][n] (row = 2^nn bits, nn >= 5), C: [b][m][n]
// C[b,m,:] = OR_k (A[b,m,k] ? B[b,k,:] : 0); each thread produces RM rows x 1 word.
template <int RM>
__global__ void __launch_bounds__(256) cc_kernel(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                                 uint32_t* __restrict__ C, long long batch, int mm, int nn, int kk,
                                                 int lks) {
  // all extents are powers of two: decompose indices with shifts (no 64-bit division)
  const int lnw = nn - 5;                       // log2 words per C/B row
  const int lmb = mm >= 2 && RM == 4 ? mm - 2 : mm;  // log2 row-blocks
  const long long NW = 1ll << lnw;
  const long long M = 1ll << mm, K = 1ll << kk;
  const long long total = batch << (lmb + lnw);
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const long long w = idx & (NW - 1);
    const long long t = idx >> lnw;
    const long long mb = t & ((1ll << lmb) - 1);
    const long long b = t >> lmb;
    uint32_t acc[RM];
#pragma unroll
    for (int r = 0; r < RM; ++r) acc[r] = 0;
    const uint32_t* Bb = B + ((b << kk) << lnw) + w;
    const long long row0 = (b << mm) + mb * RM;  // global row index of first row
    const long long kc = K >> lks;               // split-K: this block handles [ky*kc, (ky+1)*kc)
    const long long kbeg = (long long)blockIdx.y * kc, kend = kbeg + kc;
    for (long long k0 = kbeg; k0 < kend; k0 += 32) {
      const int kn = (int)min(32ll, kend - k0);
      uint32_t aw[RM];
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        const long long x = ((row0 + r) << kk) + k0;
        aw[r] = (mb * RM + r < M) ? (__ldg(A + (x >> 5)) >> (x & 31)) : 0u;
      }
      const uint32_t* Bk = Bb + (k0 << lnw);
      for (int k = 0; k < kn; ++k) {
        const uint32_t bw = __ldg(Bk + ((long long)k << lnw));
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          int mask;
          asm("bfe.s32 %0, %1, %2, 1;" : "=r"(mask) : "r"(aw[r]), "r"(k));
          acc[r] |= (uint32_t)mask & bw;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RM; ++r)
      if (mb * RM + r < M) {
        if (lks == 0) C[((row0 + r) << lnw) + w] = acc[r];
        else if (acc[r]) atomicOr(&C[((row0 + r) << lnw) + w], acc[r]);
      }
  }
}

// CUDA-core masked-OR GEMM for small N (nn < 5): each output word packs R = 2^(5-nn) consecutive global rows
// (row g = b*M + m). If M < R a word spans RB = R/M batch elements, each with its own B row.
// Layouts as cc_kernel: A [b][m][k], B [b][k][n] (sub-word rows), C [b][m][n]. Works for any mm >= 0.
template <int NN>
__global__ void __launch_bounds__(256) ccs_kernel(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                                  uint32_t* __restrict__ C, long long batch, int mm, int kk, int lks) {
  constexpr int N = 1 << NN;
  constexpr int R = 32 / N;
  constexpr uint32_t NMASK = N == 32 ? 0xffffffffu : ((1u << N) - 1u);
  const long long M = 1ll << mm, K = 1ll << kk;
  const int Meff = (int)min((long long)R, M);
  const int RB = R / Meff;
  uint32_t REPM = 0;
  for (int r = 0; r < Meff; ++r) REPM |= 1u << (r * N);
  const long long total_rows = batch << mm;
  const long long nwords = (total_rows + R - 1) / R;
  for (long long wi = blockIdx.x * (long long)blockDim.x + threadIdx.x; wi < nwords;
       wi += (long long)gridDim.x * blockDim.x) {
    const long long g0 = wi * R;
    const long long b0 = g0 >> mm;
    uint32_t acc = 0;
    const long long kc = K >> lks;
    const long long kbeg = (long long)blockIdx.y * kc, kend = kbeg + kc;
    for (long long k0 = kbeg; k0 < kend; k0 += 32) {
      const int kn = (int)min(32ll, kend - k0);
      uint32_t aw[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const long long x = ((g0 + r) << kk) + k0;
        aw[r] = (g0 + r < total_rows) ? (__ldg(A + (x >> 5)) >> (x & 31)) : 0u;
      }
      for (int k = 0; k < kn; ++k) {
        uint32_t brep = 0;
#pragma unroll
        for (int q = 0; q < R; ++q) {
          if (q < RB && b0 + q < batch) {
            const long long xb = (((b0 + q) << kk) + k0 + k) << NN;
            const uint32_t bb = (__ldg(B + (xb >> 5)) >> (xb & 31)) & NMASK;
            brep |= (bb * REPM) << (q * Meff * N);
          }
        }
        uint32_t mask = 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          int m1;
          asm("bfe.s32 %0, %1, %2, 1;" : "=r"(m1) : "r"(aw[r]), "r"(k));
          mask |= (uint32_t)m1 & (NMASK << (r * N));
        }
        acc |= brep & mask;
      }
    }
    if (lks == 0) C[wi] = acc;
    else if (acc) atomicOr(&C[wi], acc);
  }
}

// ------------------------ batched small contractions (one block per op) -----------------------
// Arbitrary layouts, no permutes. Used for the many tiny contractions near the leaves, launched
// level-by-level so a whole tree level is one kernel launch.
struct SmallOp {
  const uint32_t* a;
  const uint32_t* b;
  uint32_t* c;
  int rc, nk;
  signed char a_pos[24], b_pos[24];  // position in A/B of output label p (-1 if absent)
  signed char ka[8], kb[8];          // positions in A/B of summed label i (-1 if absent)
};

__global__ void __launch_bounds__(256) small_batch_kernel(const SmallOp* __restrict__ ops) {
  __shared__ SmallOp s;
  __shared__ long long offa[32], offb[32], koa[256], kob[256];
  if (threadIdx.x == 0) s = ops[blockIdx.x];
  __syncthreads();
  const int rc = s.rc, ncomb = 1 << s.nk;
  if (threadIdx.x < 32) {
    long long xa = 0, xb = 0;
    for (int p = 0; p < 5 && p < rc; ++p)
      if ((threadIdx.x >> p) & 1) {
        if (s.a_pos[p] >= 0) xa += 1ll << s.a_pos[p];
        if (s.b_pos[p] >= 0) xb += 1ll << s.b_pos[p];
      }
    offa[threadIdx.x] = xa;
    offb[threadIdx.x] = xb;
  }
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
  const long long nbits = 1ll << rc;
  const long long nb_pad = (nbits + 31) & ~31ll;
  const int lane = threadIdx.x & 31;
  for (long long y = threadIdx.x; y < nb_pad; y += blockDim.x) {
    uint32_t acc = 0;
    if (y < nbits) {
      const long long w = y >> 5;
      long long xa = offa[lane], xb = offb[lane];
      for (int p = 5; p < rc; ++p)
        if ((w >> (p - 5)) & 1) {
          if (s.a_pos[p] >= 0) xa += 1ll << s.a_pos[p];
          if (s.b_pos[p] >= 0) xb += 1ll << s.b_pos[p];
        }
      for (int j = 0; j < ncomb && !acc; ++j) acc = getbit(s.a, xa + koa[j]) & getbit(s.b, xb + kob[j]);
    }
    const uint32_t word = __ballot_sync(0xffffffffu, acc);
    if (lane == 0 && y < nbits) s.c[y >> 5] = word;
  }
}

// ------------------------------------ generic contraction ------------------------------------
struct GenParams {
  IdxMap ma, mb;
  long long ka[256], kb[256];
  int ncomb, rc;
  long long nwords;
};

__global__ void __launch_bounds__(256) gen_kernel(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                                  uint32_t* __restrict__ C, const __grid_constant__ GenParams P) {
  __shared__ IdxMap sa, sb;
  __shared__ long long ska[256], skb[256];
  for (int i = threadIdx.x; i < (int)(sizeof(IdxMap) / 8); i += blockDim.x) {
    reinterpret_cast<long long*>(&sa)[i] = reinterpret_cast<const long long*>(&P.ma)[i];
    reinterpret_cast<long long*>(&sb)[i] = reinterpret_cast<const long long*>(&P.mb)[i];
  }
  for (int i = threadIdx.x; i < P.ncomb; i += blockDim.x) {
    ska[i] = P.ka[i];
    skb[i] = P.kb[i];
  }
  __syncthreads();
  const int nb = P.rc >= 5 ? 32 : (1 << P.rc);
  for (long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x; w < P.nwords;
       w += (long long)gridDim.x * blockDim.x) {
    const long long ba = map_base(sa, w), bb = map_base(sb, w);
    uint32_t r = 0;
    for (int t = 0; t < nb; ++t) {
      const long long xa = ba + sa.off[t], xb = bb + sb.off[t];
      uint32_t acc = 0;
      for (int j = 0; j < P.ncomb && !acc; ++j) acc = getbit(A, xa + ska[j]) & getbit(B, xb + skb[j]);
      r |= acc << t;
    }
    C[w] = r;
  }
}

// ------------------------------------ tensor-core GEMM ---------------------------------------
typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PFN_encodeTiled g_encode = nullptr;

static int make_map(CUtensorMap* m, const void* ptr, long long row_bytes, long long rows, long long batch,
                    int box_rows) {
  cuuint64_t dims[3] = {(cuuint64_t)row_bytes, (cuuint64_t)rows, (cuuint64_t)batch};
  cuuint64_t strides[2] = {(cuuint64_t)row_bytes, (cuuint64_t)(row_bytes * rows)};
  cuuint32_t box[3] = {128, (cuuint32_t)box_rows, 1};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult r = g_encode(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(ptr), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "booltn: cuTensorMapEncodeTiled failed (%d) rows=%lld row_bytes=%lld batch=%lld\n", (int)r, rows,
            row_bytes, batch);
    return -1;
  }
  return 0;
}

typedef void (*TcKernelPtr)(CUtensorMap, CUtensorMap, tcg::TcParams);

struct TcLaunch {
  TcKernelPtr kern;
  CUtensorMap mA, mB;
  tcg::TcParams p;
  int grid, threads;
  size_t smem;
};

template <int BN, int NWG, int STAGES, bool SMR, int MINB>
static int prepare_tc(TcLaunch& L, const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk) {
  constexpr int BM = 64 * NWG;
  long long Kbytes = 1ll << (kk - 3);
  if (make_map(&L.mA, A, Kbytes, 1ll << mm, 1ll << bb, BM)) return -1;
  if (make_map(&L.mB, B, Kbytes, 1ll << nn, 1ll << bb, BN)) return -1;
  tcg::TcParams& p = L.p;
  p.M = 1 << mm;
  p.N = 1 << nn;
  p.Kbits = (int)std::min<long long>(1ll << kk, 1 << 30);
  p.Kbytes = (int)Kbytes;
  p.batch = 1 << bb;
  p.C = C;
  p.ldc_words = 1 << (nn - 5);
  p.c_batch_stride = (long long)p.M * p.ldc_words;
  p.tiles_m = (p.M + BM - 1) / BM;
  p.tiles_n = (p.N + BN - 1) / BN;
  p.group_m = 8;
  long long ntiles = (long long)p.tiles_m * p.tiles_n * p.batch;
  if (ntiles >= (1ll << 31)) return -2;
  // split-K when there are few output tiles but many K stages (bool OR of partial results via atomicOr)
  const int KT = (int)((Kbytes + 127) / 128);
  p.ksplit = 1;
  while (ntiles * p.ksplit < 2ll * g_nsm && KT / (p.ksplit * 2) >= 4) p.ksplit *= 2;
  p.kt_per_split = (KT + p.ksplit - 1) / p.ksplit;
  ntiles *= p.ksplit;
  auto kern = tcg::tc_kernel<BN, NWG, STAGES, SMR, MINB>;
  L.kern = (TcKernelPtr)kern;
  L.smem = sizeof(tcg::TcSmem<BN, NWG, STAGES>) + 1024;
  L.threads = 128 * (NWG + 1);
  static bool configured = false;
  static int occ = 1;
  if (!configured) {
    BT_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smem));
    BT_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, L.threads, L.smem));
    if (occ < 1) occ = 1;
    configured = true;
  }
  L.grid = (int)std::min<long long>(ntiles, (long long)g_nsm * occ);
  return 0;
}

// host planning for perm2 (rin == rout == r >= 10, no reduction). Returns false if not applicable.
static bool plan_perm2(Perm2Params& P, const int* src, int r) {
  if (r < 10 || r > 62) return false;
  std::vector<int> inv(r);
  for (int p = 0; p < r; ++p) inv[src[p]] = p;
  int a = 5, b = 5;
  std::vector<int> S;
  auto build = [&]() {
    std::vector<char> in_s(r, 0);
    for (int q = 0; q < a; ++q) in_s[q] = 1;
    for (int p = 0; p < b; ++p) in_s[src[p]] = 1;
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
      if ((int)S.size() > 16) { --v; build(); } else grown = true;
    }
    if (!grown) break;
  }
  // tile-local ordering: input positions 0..a-1, then the rest of S ascending
  std::vector<int> Tl;
  for (int q = 0; q < a; ++q) Tl.push_back(q);
  for (int q : S) if (q >= a) Tl.push_back(q);
  const int ts = (int)Tl.size();
  P.ts = ts;
  std::vector<int> tl_of_in(r, -1);
  for (int k = 0; k < ts; ++k) tl_of_in[Tl[k]] = k;
  // outer labels
  std::vector<int> Ol;
  for (int q = 0; q < r; ++q) if (tl_of_in[q] < 0) Ol.push_back(q);
  P.nouter_bits = (int)Ol.size();
  if (P.nouter_bits > 32) return false;
  P.nouter = 1ll << P.nouter_bits;
  for (int j = 0; j < P.nouter_bits; ++j) {
    P.out_in_w[j] = 1ll << (Ol[j] - 5);
    P.out_out_w[j] = 1ll << (inv[Ol[j]] - 5);
  }
  // load table: tile word bit (k-5) -> input word offset
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 64; ++v) {
      long long x = 0;
      for (int t = 0; t < 6; ++t) {
        int k = 5 + 6 * c + t;
        if (k < ts && ((v >> t) & 1)) x += 1ll << (Tl[k] - 5);
      }
      P.ld[c][v] = x;
    }
  // exchange planning
  std::vector<int> Eout, Ein;  // Eout: tile bits (>=5) whose label goes to output in-word; Ein: in-word positions leaving
  std::vector<char> out_inword(r, 0);
  for (int p = 0; p < 5; ++p) out_inword[src[p]] = 1;
  for (int k = 5; k < ts; ++k) if (out_inword[Tl[k]]) Eout.push_back(k);
  for (int q = 0; q < 5; ++q) if (!out_inword[q]) Ein.push_back(q);
  const int e = (int)Eout.size();
  P.e = e;
  for (int j = 0; j < e; ++j) P.ein[j] = Ein[j];
  // group bits: tile word bits (k>=5) excluding Eout
  std::vector<int> Gk;
  for (int k = 5; k < ts; ++k) if (std::find(Eout.begin(), Eout.end(), k) == Eout.end()) Gk.push_back(k);
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 64; ++v) {
      int x = 0;
      for (int t = 0; t < 6; ++t) {
        int gi = 6 * c + t;
        if (gi < (int)Gk.size() && ((v >> t) & 1)) x += 1 << (Gk[gi] - 5);
      }
      P.gb[c][v] = x;
    }
  for (int i = 0; i < 32; ++i) {
    int x = 0;
    for (int j = 0; j < e; ++j) if ((i >> j) & 1) x += 1 << (Eout[j] - 5);
    P.ex[i] = x;
  }
  // in-word arrangement after exchange: arr[x] = input position of label at in-word position x
  int arr[5];
  for (int x = 0; x < 5; ++x) arr[x] = x;
  for (int j = 0; j < e; ++j) arr[Ein[j]] = Tl[Eout[j]];
  P.nswap = 0;
  for (int p = 0; p < 5; ++p) {
    if (arr[p] == src[p]) continue;
    int x = p + 1;
    while (x < 5 && arr[x] != src[p]) ++x;
    if (x >= 5) return false;
    // swap index bits p < x
    int d = (1 << x) - (1 << p);
    uint32_t m = 0;
    for (int t = 0; t < 32; ++t) if (((t >> p) & 1) && !((t >> x) & 1)) m |= 1u << t;
    P.sw_d[P.nswap] = d;
    P.sw_m[P.nswap] = m;
    ++P.nswap;
    std::swap(arr[p], arr[x]);
  }
  // output tile ordering: To = [src[0..b-1]] + remaining tile labels sorted by output position
  std::vector<int> To;
  for (int p = 0; p < b; ++p) To.push_back(src[p]);
  {
    std::vector<int> rest;
    for (int k = 0; k < ts; ++k) if (inv[Tl[k]] >= b) rest.push_back(Tl[k]);
    std::sort(rest.begin(), rest.end(), [&](int x, int y) { return inv[x] < inv[y]; });
    for (int q : rest) To.push_back(q);
  }
  if ((int)To.size() != ts) return false;
  std::vector<int> ol_of_in(r, -1);
  for (int k = 0; k < ts; ++k) ol_of_in[To[k]] = k;
  // group g -> output smem word; exchange idx i (Ein labels) -> output smem word offset
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 64; ++v) {
      int x = 0;
      for (int t = 0; t < 6; ++t) {
        int gi = 6 * c + t;
        if (gi < (int)Gk.size() && ((v >> t) & 1)) {
          int k2 = ol_of_in[Tl[Gk[gi]]];
          if (k2 < 5) return false;
          x += 1 << (k2 - 5);
        }
      }
      P.og[c][v] = x;
    }
  for (int i = 0; i < 32; ++i) {
    int x = 0;
    for (int j = 0; j < e; ++j)
      if ((i >> j) & 1) {
        int k2 = ol_of_in[Ein[j]];
        if (k2 < 5) return false;
        x += 1 << (k2 - 5);
      }
    P.oe[i] = x;
  }
  for (int c = 0; c < 2; ++c)
    for (int v = 0; v < 64; ++v) {
      long long x = 0;
      for (int t = 0; t < 6; ++t) {
        int k = 5 + 6 * c + t;
        if (k < ts && ((v >> t) & 1)) x += 1ll << (inv[To[k]] - 5);
      }
      P.st[c][v] = x;
    }
  return true;
}

// (namespace continues)
static long long grid_for(long long work, int per_block = 256) {
  long long g = (work + per_block - 1) / per_block;
  return std::max<long long>(1, std::min<long long>(g, (long long)g_nsm * 32));
}

static int g_perm_impl = 2;  // 2 = use tiled perm2 when applicable, 1 = legacy gather only

// One prepared kernel launch (all host-side planning done at construction).
struct Op {
  int kind = 0;  // 1 perm (gather), 2 perm2 (tiled), 3 cc, 4 tc, 5 generic
  const uint32_t* a = nullptr;
  const uint32_t* b = nullptr;
  uint32_t* c = nullptr;
  std::unique_ptr<PermParams> pp;
  std::unique_ptr<Perm2Params> p2;
  std::unique_ptr<GenParams> gp;
  TcLaunch tc;
  long long grid = 1;
  int bb = 0, mm = 0, nn = 0, kk = 0;
  int lks = 0;              // log2 split-K factor (cc/ccs/tc); output pre-zeroed when > 0
  size_t c_bytes = 0;

  int launch(cudaStream_t st) const {
    switch (kind) {
      case 1: perm_kernel<<<(int)grid, 256, 0, st>>>(a, c, *pp); break;
      case 2:
        switch (p2->e) {
          case 0: perm2_kernel<0><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
          case 1: perm2_kernel<1><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
          case 2: perm2_kernel<2><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
          case 3: perm2_kernel<3><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
          case 4: perm2_kernel<4><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
          default: perm2_kernel<5><<<(int)grid, 256, 0, st>>>(a, c, *p2); break;
        }
        break;
      case 3: {
        if (lks) BT_CHECK(cudaMemsetAsync(c, 0, c_bytes, st));
        dim3 g((unsigned)grid, 1u << lks);
        if (mm >= 2)
          cc_kernel<4><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, nn, kk, lks);
        else
          cc_kernel<1><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, nn, kk, lks);
        break;
      }
      case 4:
        if (tc.p.ksplit > 1) BT_CHECK(cudaMemsetAsync(tc.p.C, 0, c_bytes, st));
        tc.kern<<<tc.grid, tc.threads, tc.smem, st>>>(tc.mA, tc.mB, tc.p);
        break;
      case 5: gen_kernel<<<(int)grid, 256, 0, st>>>(a, b, c, *gp); break;
      case 6: or_accum_kernel<<<1, 32, 0, st>>>(c, a); break;
      case 8: small_batch_kernel<<<(int)grid, 256, 0, st>>>((const SmallOp*)a); break;
      case 9: pad_rows_kernel<<<(int)grid, 256, 0, st>>>(a, (uint4*)c, 1ll << bb, kk); break;
      case 7: {
        if (lks) BT_CHECK(cudaMemsetAsync(c, 0, c_bytes, st));
        dim3 g((unsigned)grid, 1u << lks);
        switch (nn) {
          case 0: ccs_kernel<0><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, kk, lks); break;
          case 1: ccs_kernel<1><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, kk, lks); break;
          case 2: ccs_kernel<2><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, kk, lks); break;
          case 3: ccs_kernel<3><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, kk, lks); break;
          default: ccs_kernel<4><<<g, 256, 0, st>>>(a, b, c, 1ll << bb, mm, kk, lks); break;
        }
        break;
      }
      default: return -9;
    }
    BT_CHECK(cudaGetLastError());
    return 0;
  }
};

static int make_permute(Op& op, const uint32_t* in, int rin, uint32_t* out, int rout, const int* src_pos,
                        const int* red_pos, int nred) {
  if (nred > 8) return -3;
  op.a = in;
  op.c = out;
  if (g_perm_impl == 2 && nred == 0 && rin == rout) {
    op.p2.reset(new Perm2Params);
    if (plan_perm2(*op.p2, src_pos, rin)) {
      op.kind = 2;
      op.grid = std::min<long long>(op.p2->nouter, (long long)g_nsm * 8);
      return 0;
    }
    op.p2.reset();
  }
  op.pp.reset(new PermParams);
  PermParams& P = *op.pp;
  build_map(P.m, src_pos, rout);
  P.ncomb = 1 << nred;
  for (int j = 0; j < P.ncomb; ++j) {
    long long x = 0;
    for (int i = 0; i < nred; ++i)
      if ((j >> i) & 1) x += 1ll << red_pos[i];
    P.roff[j] = x;
  }
  P.rout = rout;
  bool word_copy = rout >= 5 && nred == 0 && rin >= 5;
  bool in_word = nred == 0 && rin >= 5;
  for (int p = 0; p < 5 && p < rout; ++p) {
    if (src_pos[p] != p) word_copy = false;
    if (src_pos[p] >= 5) in_word = false;
  }
  P.mode = word_copy ? 0 : (in_word ? 1 : 2);
  P.nwords = rout >= 5 ? (1ll << (rout - 5)) : 1;
  op.kind = 1;
  op.grid = grid_for(P.nwords);
  return 0;
}

static int choose_lks(long long threads, int kk) {
  int lks = 0;
  while ((threads << lks) < (1ll << 17) && kk - lks > 6 && lks < 12) ++lks;
  return lks;
}

static int make_cc(Op& op, const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk) {
  op.a = A; op.b = B; op.c = C;
  op.bb = bb; op.mm = mm; op.nn = nn; op.kk = kk;
  long long nbits_c = 1ll << (bb + mm + nn);
  op.c_bytes = (size_t)std::max<long long>(4, (nbits_c + 31) / 32 * 4);
  if (nn < 5) {
    op.kind = 7;
    long long rows = 1ll << (bb + mm);
    long long R = 1ll << (5 - nn);
    long long threads = (rows + R - 1) / R;
    op.lks = choose_lks(threads, kk);
    op.grid = grid_for(threads);
    return 0;
  }
  op.kind = 3;
  long long rows = mm >= 2 ? ((1ll << mm) + 3) / 4 : (1ll << mm);
  long long threads = (1ll << bb) * rows * (1ll << (nn - 5));
  op.lks = choose_lks(threads, kk);
  op.grid = grid_for(threads);
  return 0;
}

static int make_tc(Op& op, const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk, int cfg) {
  if (kk < 7 || nn < 5 || bb > 30) return -3;
  if (cfg < 0) {
    int bn = nn >= 8 ? 256 : (nn >= 7 ? 128 : 64);
    bool two = mm >= 7;
    if (bn == 256) cfg = two ? 0 : 4;
    else if (bn == 128) cfg = two ? 1 : 3;
    else cfg = 2;
  }
  op.kind = 4;
  op.c_bytes = (size_t)(1ll << (bb + mm + nn - 3));
  switch (cfg) {
    case 0: return prepare_tc<256, 2, 4, true, 1>(op.tc, A, B, C, bb, mm, nn, kk);
    case 1: return prepare_tc<128, 2, 6, true, 1>(op.tc, A, B, C, bb, mm, nn, kk);
    case 2: return prepare_tc<64, 1, 4, false, 3>(op.tc, A, B, C, bb, mm, nn, kk);
    case 3: return prepare_tc<128, 1, 4, false, 2>(op.tc, A, B, C, bb, mm, nn, kk);
    case 4: return prepare_tc<256, 1, 4, true, 1>(op.tc, A, B, C, bb, mm, nn, kk);
  }
  return -4;
}

static int make_generic(Op& op, const uint32_t* A, const uint32_t* B, uint32_t* C, int rc, const int* a_pos,
                        const int* b_pos, int nk, const int* k_a, const int* k_b) {
  if (nk > 8) return -3;
  op.kind = 5;
  op.a = A; op.b = B; op.c = C;
  op.gp.reset(new GenParams);
  GenParams& P = *op.gp;
  build_map(P.ma, a_pos, rc);
  build_map(P.mb, b_pos, rc);
  P.ncomb = 1 << nk;
  for (int j = 0; j < P.ncomb; ++j) {
    long long xa = 0, xb = 0;
    for (int i = 0; i < nk; ++i)
      if ((j >> i) & 1) {
        if (k_a[i] >= 0) xa += 1ll << k_a[i];
        if (k_b[i] >= 0) xb += 1ll << k_b[i];
      }
    P.ka[j] = xa;
    P.kb[j] = xb;
  }
  P.rc = rc;
  P.nwords = rc >= 5 ? (1ll << (rc - 5)) : 1;
  op.grid = grid_for(P.nwords);
  return 0;
}

struct Plan {
  std::vector<Op> ops;
  std::vector<void*> dev_bufs;  // device op tables owned by the plan
  cudaGraphExec_t exec = nullptr;
  cudaStream_t cap = nullptr;
};

}  // namespace bt

// ============================================ C API ============================================
using namespace bt;
extern "C" {

int bt_init() {
  BT_CHECK(cudaFree(0));
  BT_CHECK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
  cudaDriverEntryPointQueryResult q;
  BT_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_encode, cudaEnableDefault, &q));
  cudaMemPool_t pool;
  BT_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t thr = UINT64_MAX;
  BT_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  BT_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  return 0;
}

void* bt_malloc(size_t bytes) {
  void* p = nullptr;
  if (bytes == 0) bytes = 4;
  bytes = (bytes + 255) & ~(size_t)255;
  if (cudaMallocAsync(&p, bytes, g_stream) != cudaSuccess) return nullptr;
  return p;
}
int bt_free(void* p) {
  BT_CHECK(cudaFreeAsync(p, g_stream));
  return 0;
}
int bt_h2d(void* dst, const void* src, size_t bytes) {
  BT_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_stream));
  return 0;
}
int bt_d2h(void* dst, const void* src, size_t bytes) {
  BT_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_stream));
  BT_CHECK(cudaStreamSynchronize(g_stream));
  return 0;
}
int bt_memset(void* p, int v, size_t bytes) {
  BT_CHECK(cudaMemsetAsync(p, v, bytes, g_stream));
  return 0;
}
int bt_sync() {
  BT_CHECK(cudaStreamSynchronize(g_stream));
  return 0;
}
void* bt_event_create() {
  cudaEvent_t e;
  if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
  return (void*)e;
}
int bt_event_record(void* e) {
  BT_CHECK(cudaEventRecord((cudaEvent_t)e, g_stream));
  return 0;
}
float bt_event_elapsed(void* e0, void* e1) {
  float ms = -1;
  cudaEventSynchronize((cudaEvent_t)e1);
  cudaEventElapsedTime(&ms, (cudaEvent_t)e0, (cudaEvent_t)e1);
  return ms;
}
size_t bt_mem_used() {
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);
  uint64_t v = 0;
  cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &v);
  return (size_t)v;
}

int bt_fill_random(uint32_t* p, long long nwords, unsigned seed, int and_rounds) {
  fill_random_kernel<<<(int)grid_for(nwords), 256, 0, g_stream>>>(p, nwords, seed, and_rounds);
  BT_CHECK(cudaGetLastError());
  return 0;
}

int bt_set_perm_impl(int v) { g_perm_impl = v; return 0; }

int bt_pad_rows(const uint32_t* in, uint32_t* out, int lrows, int kk) {
  if (kk > 6) return -3;
  pad_rows_kernel<<<(int)grid_for(1ll << lrows), 256, 0, g_stream>>>(in, (uint4*)out, 1ll << lrows, kk);
  BT_CHECK(cudaGetLastError());
  return 0;
}

// ---- immediate-mode kernels ----
int bt_permute(const uint32_t* in, int rin, uint32_t* out, int rout, const int* src_pos, const int* red_pos,
               int nred) {
  Op op;
  int rc = make_permute(op, in, rin, out, rout, src_pos, red_pos, nred);
  if (rc) return rc;
  rc = op.launch(g_stream);
  return rc ? rc : op.kind;
}
int bt_gemm_cc(const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk) {
  Op op;
  int rc = make_cc(op, A, B, C, bb, mm, nn, kk);
  return rc ? rc : op.launch(g_stream);
}
int bt_gemm_tc(const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk, int cfg) {
  Op op;
  int rc = make_tc(op, A, B, C, bb, mm, nn, kk, cfg);
  return rc ? rc : op.launch(g_stream);
}
int bt_contract_generic(const uint32_t* A, const uint32_t* B, uint32_t* C, int rc_, const int* a_pos,
                        const int* b_pos, int nk, const int* k_a, const int* k_b) {
  Op op;
  int rc = make_generic(op, A, B, C, rc_, a_pos, b_pos, nk, k_a, k_b);
  return rc ? rc : op.launch(g_stream);
}

// ---- compiled plans: record ops once, replay many times (optionally as a CUDA graph) ----
void* bt_plan_new() { return new Plan(); }
int bt_plan_free(void* h) {
  Plan* P = (Plan*)h;
  for (void* p : P->dev_bufs) cudaFree(p);
  if (P->exec) cudaGraphExecDestroy(P->exec);
  delete P;
  return 0;
}
int bt_plan_size(void* h) { return (int)((Plan*)h)->ops.size(); }
int bt_plan_permute(void* h, const uint32_t* in, int rin, uint32_t* out, int rout, const int* src_pos,
                    const int* red_pos, int nred) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  int rc = make_permute(P->ops.back(), in, rin, out, rout, src_pos, red_pos, nred);
  if (rc) P->ops.pop_back();
  return rc;
}
int bt_plan_cc(void* h, const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  int rc = make_cc(P->ops.back(), A, B, C, bb, mm, nn, kk);
  if (rc) P->ops.pop_back();
  return rc;
}
int bt_plan_tc(void* h, const uint32_t* A, const uint32_t* B, uint32_t* C, int bb, int mm, int nn, int kk, int cfg) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  int rc = make_tc(P->ops.back(), A, B, C, bb, mm, nn, kk, cfg);
  if (rc) P->ops.pop_back();
  return rc;
}
int bt_plan_generic(void* h, const uint32_t* A, const uint32_t* B, uint32_t* C, int rc_, const int* a_pos,
                    const int* b_pos, int nk, const int* k_a, const int* k_b) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  int rc = make_generic(P->ops.back(), A, B, C, rc_, a_pos, b_pos, nk, k_a, k_b);
  if (rc) P->ops.pop_back();
  return rc;
}
// A batch of independent small contractions executed by one launch (one block each).
// Host array of n ops described by flat arrays to keep the ctypes interface simple:
//   ptrs[3n] (a,b,c), rc[n], nk[n], apos[24n], bpos[24n], ka[8n], kb[8n]
int bt_plan_small_batch(void* h, int n, void** ptrs, const int* rc, const int* nk, const int* apos, const int* bpos,
                        const int* ka, const int* kb) {
  Plan* P = (Plan*)h;
  std::vector<SmallOp> v(n);
  for (int i = 0; i < n; ++i) {
    SmallOp& o = v[i];
    o.a = (const uint32_t*)ptrs[3 * i];
    o.b = (const uint32_t*)ptrs[3 * i + 1];
    o.c = (uint32_t*)ptrs[3 * i + 2];
    o.rc = rc[i];
    o.nk = nk[i];
    if (o.rc > 24 || o.nk > 8) return -3;
    for (int p = 0; p < 24; ++p) { o.a_pos[p] = (signed char)apos[24 * i + p]; o.b_pos[p] = (signed char)bpos[24 * i + p]; }
    for (int p = 0; p < 8; ++p) { o.ka[p] = (signed char)ka[8 * i + p]; o.kb[p] = (signed char)kb[8 * i + p]; }
  }
  void* d = nullptr;
  BT_CHECK(cudaMalloc(&d, sizeof(SmallOp) * n));
  BT_CHECK(cudaMemcpy(d, v.data(), sizeof(SmallOp) * n, cudaMemcpyHostToDevice));
  P->dev_bufs.push_back(d);
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 8;
  op.a = (const uint32_t*)d;
  op.grid = n;
  return 0;
}

// pad rows of 2^kk bits (kk <= 6) to 128 bits; nrows = 2^lrows
int bt_plan_pad_rows(void* h, const uint32_t* in, uint32_t* out, int lrows, int kk) {
  if (kk > 6) return -3;
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 9;
  op.a = in;
  op.c = out;
  op.bb = lrows;  // reuse field: log2 rows
  op.kk = kk;
  op.grid = grid_for(1ll << lrows);
  return 0;
}

// flag[0] |= src[0] & 1 (accumulate the boolean result over slices on the device)
int bt_plan_or_accum(void* h, uint32_t* flag, const uint32_t* src) {
  Plan* P = (Plan*)h;
  P->ops.emplace_back();
  Op& op = P->ops.back();
  op.kind = 6;
  op.a = src;
  op.c = flag;
  return 0;
}
// Launch ops [i0, i1) directly (for per-op timing / debugging).
int bt_plan_run_range(void* h, int i0, int i1) {
  Plan* P = (Plan*)h;
  for (int i = i0; i < i1 && i < (int)P->ops.size(); ++i) {
    int rc = P->ops[i].launch(g_stream);
    if (rc) return rc;
  }
  return 0;
}
// Create n extra streams (for running several plans/slices concurrently).
int bt_make_streams(int n) {
  while ((int)g_streams.size() < n) {
    cudaStream_t st;
    BT_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    g_streams.push_back(st);
  }
  return (int)g_streams.size();
}
static cudaStream_t pick_stream(int sidx) { return sidx < 0 || sidx >= (int)g_streams.size() ? g_stream : g_streams[sidx]; }
int bt_h2d_stream(void* dst, const void* src, size_t bytes, int sidx) {
  BT_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, pick_stream(sidx)));
  return 0;
}
int bt_d2h_stream(void* dst, const void* src, size_t bytes, int sidx) {
  cudaStream_t st = pick_stream(sidx);
  BT_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st));
  BT_CHECK(cudaStreamSynchronize(st));
  return 0;
}
int bt_sync_all() {
  BT_CHECK(cudaDeviceSynchronize());
  return 0;
}
// Replay a plan on stream sidx (graph captured on the default plan stream on first use).
int bt_plan_run_stream(void* h, int sidx) {
  Plan* P = (Plan*)h;
  if (!P->exec) {
    cudaGraph_t g;
    BT_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    for (auto& op : P->ops) {
      int rc = op.launch(g_stream);
      if (rc) { cudaStreamEndCapture(g_stream, &g); return rc; }
    }
    BT_CHECK(cudaStreamEndCapture(g_stream, &g));
    BT_CHECK(cudaGraphInstantiate(&P->exec, g, 0));
    cudaGraphDestroy(g);
  }
  BT_CHECK(cudaGraphLaunch(P->exec, pick_stream(sidx)));
  return 0;
}

// Replay the whole plan; use_graph=1 captures it into a CUDA graph on first use.
int bt_plan_run(void* h, int use_graph) {
  Plan* P = (Plan*)h;
  if (!use_graph) return bt_plan_run_range(h, 0, (int)P->ops.size());
  if (!P->exec) {
    cudaGraph_t g;
    BT_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    for (auto& op : P->ops) {
      int rc = op.launch(g_stream);
      if (rc) { cudaStreamEndCapture(g_stream, &g); return rc; }
    }
    BT_CHECK(cudaStreamEndCapture(g_stream, &g));
    BT_CHECK(cudaGraphInstantiate(&P->exec, g, 0));
    cudaGraphDestroy(g);
  }
  BT_CHECK(cudaGraphLaunch(P->exec, g_stream));
  return 0;
}

}  // extern "C"
