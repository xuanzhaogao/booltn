// Templated Hopper 1-bit tensor-core boolean GEMM (generalization of bgemm_sm90.cuh).
//   C[b][m][n] = OR_k A[b][m][k] & B[b][n][k]     (A, B K-major bit rows; C N-major bit rows)
// Tile: BM = 64*NWG rows (NWG consumer warpgroups), BN in {64,128,256} cols, 1024-bit K stages.
#pragma once
#include "bgemm_sm90.cuh"
#include "wgmma_b1.inc"

namespace tcg {
using namespace bgemm;

struct TcParams {
  int M, N, Kbits, Kbytes, batch;
  uint32_t* C;
  int ldc_words;
  long long c_batch_stride;  // words
  int tiles_m, tiles_n, group_m;
  int ksplit = 1;        // split-K factor (results OR-ed with atomicOr; C pre-zeroed)
  int kt_per_split = 0;  // 1024-bit K stages per split
};

template <int BN, int NWG, int STAGES>
struct TcSmem {
  alignas(1024) uint8_t A[STAGES][64 * NWG * 128];
  alignas(1024) uint8_t B[STAGES][BN * 128];
  int4 coord[STAGES];
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
};

__device__ __forceinline__ void tc_tile_coords(const TcParams& p, int tile, int& b, int& tm, int& tn) {
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

// threshold + pack + store one warpgroup's 64 x BN accumulator tile
template <int BN>
__device__ __forceinline__ void pack_store(const int (&d)[BN / 2], const TcParams& p, int b, int row0, int col0) {
  constexpr int WPR = BN / 32;  // words per row
  constexpr int H1 = WPR / 2;
  const int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) & 3, q4 = lane & 3;
  const bool hi = q4 & 2, odd = q4 & 1;
  uint32_t v[2][WPR];
#pragma unroll
  for (int q = 0; q < WPR; ++q) {
    uint32_t a0 = 0, a1 = 0;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = 4 * q + jj;
      a0 += (uint32_t)min(d[4 * j + 0], 1) << (8 * jj);
      a0 += (uint32_t)min(d[4 * j + 1], 1) << (8 * jj + 1);
      a1 += (uint32_t)min(d[4 * j + 2], 1) << (8 * jj);
      a1 += (uint32_t)min(d[4 * j + 3], 1) << (8 * jj + 1);
    }
    v[0][q] = a0 << (2 * q4);
    v[1][q] = a1 << (2 * q4);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < H1; ++i) {
      uint32_t send = hi ? v[h][i] : v[h][H1 + i];
      uint32_t keep = hi ? v[h][H1 + i] : v[h][i];
      v[h][i] = keep | __shfl_xor_sync(0xffffffff, send, 2);
    }
  uint32_t* Cb = p.C + (long long)b * p.c_batch_stride;
  const int r0 = row0 + 16 * warp + lane / 4;
  if constexpr (H1 >= 2) {
    constexpr int H2 = H1 / 2;
    uint32_t w[2][H2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < H2; ++i) {
        uint32_t send = odd ? v[h][i] : v[h][H2 + i];
        uint32_t keep = odd ? v[h][H2 + i] : v[h][i];
        w[h][i] = keep | __shfl_xor_sync(0xffffffff, send, 1);
      }
    const int wc = col0 / 32 + (hi ? H1 : 0) + (odd ? H2 : 0);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = r0 + 8 * h;
      if (row >= p.M) continue;
      uint32_t* dst = Cb + (long long)row * p.ldc_words + wc;
      if (p.ksplit > 1) {
#pragma unroll
        for (int i = 0; i < H2; ++i)
          if (wc + i < p.ldc_words && w[h][i]) atomicOr(dst + i, w[h][i]);
        continue;
      }
      if constexpr (H2 == 2) {
        if (wc + 1 < p.ldc_words && (p.ldc_words & 1) == 0) {
          *reinterpret_cast<uint2*>(dst) = make_uint2(w[h][0], w[h][1]);
          continue;
        }
      }
#pragma unroll
      for (int i = 0; i < H2; ++i)
        if (wc + i < p.ldc_words) dst[i] = w[h][i];
    }
  } else {
#pragma unroll
    for (int h = 0; h < 2; ++h) v[h][0] |= __shfl_xor_sync(0xffffffff, v[h][0], 1);
    const int wc = col0 / 32 + (hi ? 1 : 0);
    if (!odd && wc < p.ldc_words) {
      if (p.ksplit > 1) {
        if (r0 < p.M && v[0][0]) atomicOr(Cb + (long long)r0 * p.ldc_words + wc, v[0][0]);
        if (r0 + 8 < p.M && v[1][0]) atomicOr(Cb + (long long)(r0 + 8) * p.ldc_words + wc, v[1][0]);
      } else {
        if (r0 < p.M) Cb[(long long)r0 * p.ldc_words + wc] = v[0][0];
        if (r0 + 8 < p.M) Cb[(long long)(r0 + 8) * p.ldc_words + wc] = v[1][0];
      }
    }
  }
}

template <int BN, int NWG, int STAGES, bool SETMAXNREG, int MINB>
__global__ void __launch_bounds__(128 * (NWG + 1), MINB)
    tc_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, TcParams p) {
  constexpr int BM = 64 * NWG;
  constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  TcSmem<BN, NWG, STAGES>& S = *reinterpret_cast<TcSmem<BN, NWG, STAGES>*>(base);

  const int wg = threadIdx.x / 128;
  const int KT = (p.Kbytes + 127) / 128;
  const int ntiles = p.tiles_m * p.tiles_n * p.batch * p.ksplit;
  const int KTS = p.ksplit > 1 ? p.kt_per_split : KT;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&S.full[s], 1);
      mbar_init(&S.empty[s], 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    if constexpr (SETMAXNREG) setmaxnreg_dec<40>();
    if (threadIdx.x == 0) {
      int it = 0;
      for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int b, tm, tn;
        const int ks = tile % p.ksplit;
        tc_tile_coords(p, tile / p.ksplit, b, tm, tn);
        const int kt0 = ks * KTS, kt1 = min(KT, kt0 + KTS);
        for (int kt = kt0; kt < kt1; ++kt, ++it) {
          int s = it % STAGES;
          mbar_wait(&S.empty[s], ((it / STAGES) & 1) ^ 1);
          if (kt == kt0) S.coord[s] = make_int4(b, tm, tn, ks);
          mbar_expect_tx(&S.full[s], A_BYTES + B_BYTES);
          tma_load_3d(S.A[s], &tmA, &S.full[s], kt * 128, tm * BM, b);
          tma_load_3d(S.B[s], &tmB, &S.full[s], kt * 128, tn * BN, b);
        }
      }
    }
  } else {
    if constexpr (SETMAXNREG) setmaxnreg_inc<232>();
    const int cw = wg - 1;
    int it = 0;
    int d[BN / 2];
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) d[i] = 0;
    const int my_tiles = ntiles > (int)blockIdx.x ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    for (int lt = 0; lt < my_tiles; ++lt) {
      int b = 0, tm = 0, tn = 0, ks = 0;
      fence_operands(d);
      // the number of K stages of this tile depends on its split index, read from the first stage
      int nkt = 1;
      for (int j = 0; j < nkt; ++j, ++it) {
        int s = it % STAGES;
        mbar_wait(&S.full[s], (it / STAGES) & 1);
        if (j == 0) {
          int4 c = S.coord[s];
          b = c.x; tm = c.y; tn = c.z; ks = c.w;
          nkt = min(KT, ks * KTS + KTS) - ks * KTS;
        }
        const int kt = ks * KTS + j;
        wg_fence();
        fence_operands(d);
        const uint8_t* a = S.A[s] + cw * 64 * 128;
        const uint8_t* bb = S.B[s];
        const int nks = min(4, (p.Kbits - kt * 1024 + 255) / 256);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if (k < nks) Wgmma<BN>::run(d, desc_sw128(a + 32 * k), desc_sw128(bb + 32 * k), (j | k) != 0);
        wg_commit();
        fence_operands(d);
        wg_wait<1>();
        if (j > 0 && (threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
      }
      wg_wait<0>();
      fence_operands(d);
      if ((threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
      pack_store<BN>(d, p, b, tm * BM + cw * 64, tn * BN);
    }
  }
}

}  // namespace tcg
