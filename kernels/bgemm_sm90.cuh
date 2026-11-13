// Boolean (OR-AND semiring) batched GEMM on Hopper 1-bit tensor cores.
//
//   C[b][m][n] = OR_k ( A[b][m][k] AND B[b][n][k] )
//
// Layouts (all bit-packed, little-endian within bytes/words):
//   A : [batch][M][K bits]  (K-major, row pitch = lda bytes, multiple of 16)
//   B : [batch][N][K bits]  (K-major, row pitch = ldb bytes, multiple of 16)
//   C : [batch][M][N bits]  (N-major, row pitch = ldc 32-bit words)  -- or int32 counts
//
// Kernel: warp-specialized, TMA (128B swizzle) -> SMEM ring of STAGES, two consumer
// warpgroups issuing wgmma.m64n256k256.s32.b1.b1.and.popc, epilogue thresholds
// popc counts (>0) and packs bits with warp shuffles.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace bgemm {

// ------------------------------- PTX helpers -------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t cta) {
  // arrive on the barrier at the same smem offset in CTA `cta` of the cluster
  asm volatile(
      "{\n.reg .b32 remAddr32;\n"
      "mapa.shared::cluster.u32 remAddr32, %0, %1;\n"
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n}\n" ::"r"(smem_u32(bar)),
      "r"(cta)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE_%=;\n"
      "bra WAIT_%=;\n"
      "DONE_%=:\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2,
                                               uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4, %5}], [%2], %6;\n" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map, const void* src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];\n" ::"l"(map),
               "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
template <int N>
__device__ __forceinline__ void fence_operands(int (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i])::"memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
template <uint32_t R>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R));
}
template <uint32_t R>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R));
}

// wgmma smem matrix descriptor, K-major operand stored with 128B swizzle:
// rows of 128 bytes, 8-row swizzle atoms of 1024 bytes stacked along M/N.
__device__ __forceinline__ uint64_t desc_sw128(const void* p) {
  uint64_t addr = smem_u32(p);
  uint64_t d = 0;
  d |= (addr & 0x3FFFF) >> 4;            // start address
  d |= (uint64_t)1 << 16;                // leading byte offset (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;      // stride byte offset: 8 rows * 128B
  d |= (uint64_t)1 << 62;                // 128B swizzle
  return d;
}

// D(64x256,s32) (+)= popc(A(64x256b) & B(256x256b)^T)
__device__ __forceinline__ void wgmma_b1_n256(int (&d)[128], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k256.s32.b1.b1.and.popc "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
      "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
      "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
      " %128, %129, p;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
        "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
        "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
        "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]),
        "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]),
        "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]),
        "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]),
        "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]),
        "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]),
        "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]),
        "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]),
        "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]),
        "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), "+r"(d[104]),
        "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111]),
        "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]), "+r"(d[118]),
        "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]),
        "+r"(d[126]), "+r"(d[127])
      : "l"(da), "l"(db), "r"(scale_d));
}

// ------------------------------- kernel ------------------------------------
constexpr int BM = 128;          // 2 consumer warpgroups x 64 rows
constexpr int BN = 256;
constexpr int BK_BYTES = 128;    // 1024 bits per stage (one 128B swizzle row)
constexpr int A_STAGE_BYTES = BM * BK_BYTES;   // 16 KB
constexpr int B_STAGE_BYTES = BN * BK_BYTES;   // 32 KB

struct Params {
  int M, N, Kbytes, batch;   // Kbytes = ceil(K/8) rounded to 16
  uint32_t* C;               // bit output (if counts == nullptr)
  int* counts;               // optional int32 count output [batch][M][N]
  int ldc_words;             // C row pitch in 32-bit words
  long long c_batch_stride;  // in words (bits mode) or ints (count mode)
  int tiles_m, tiles_n;      // tile counts
  int group_m;               // raster: tiles grouped in bands of group_m tile-rows for L2 reuse
  int use_tma_store;         // 1 if C row pitch is a multiple of 16 bytes and tmC is valid
  long long* trace;          // optional timeline buffer (CTA 0): [who][tile][event]
};

// map linear tile id -> (batch, tm, tn) with grouped rasterization
__device__ __forceinline__ void tile_coords(const Params& p, int tile, int& b, int& tm, int& tn) {
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

// Threshold (count>0) + pack the 64x256 accumulator tile into bits. Each lane returns
// out[0..1] = words (2*q4, 2*q4+1) of row r0 = 16*warp + lane/4, out[2..3] = same for row r0+8.
__device__ __forceinline__ void pack_bits(const int (&d)[128], uint32_t (&out)[4]) {
  const int q4 = threadIdx.x & 3;
  uint32_t v[2][8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
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
  // reduce-scatter across the 4 lanes of a quad
  const bool hi = q4 & 2, odd = q4 & 1;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t send = hi ? v[h][i] : v[h][4 + i];
      uint32_t keep = hi ? v[h][4 + i] : v[h][i];
      v[h][i] = keep | __shfl_xor_sync(0xffffffff, send, 2);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t send = odd ? v[h][i] : v[h][2 + i];
      uint32_t keep = odd ? v[h][2 + i] : v[h][i];
      out[2 * h + i] = keep | __shfl_xor_sync(0xffffffff, send, 1);
    }
  }
}

template <int STAGES>
struct SharedStorage {
  alignas(1024) uint8_t A[STAGES][A_STAGE_BYTES];
  alignas(1024) uint8_t B[STAGES][B_STAGE_BYTES];
  alignas(128) uint8_t Cs[2][2][64 * 32];
  int4 coord[STAGES];  // (b, tm, tn) of the tile this stage belongs to, written by the producer
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
  alignas(8) uint64_t go;
};

// Pack the thresholded 64x256 accumulator tile of one warpgroup into bits and store.
// Accumulator layout (wgmma m64nN, 32-bit): warp w, lane l, j = 0..N/8-1:
//   d[4j+0], d[4j+1] -> row 16w + l/4,     cols 8j + 2(l%4) + {0,1}
//   d[4j+2], d[4j+3] -> row 16w + l/4 + 8, cols 8j + 2(l%4) + {0,1}
__device__ __forceinline__ void store_bits(const int (&d)[128], const Params& p, int b, int row0, int col0) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x / 32) & 3;
  const int q4 = lane & 3;
  const int r0 = row0 + 16 * warp + lane / 4;
  uint32_t* Cb = p.C + (long long)b * p.c_batch_stride;
#pragma unroll
  for (int q = 0; q < 8; ++q) {  // 8 output words of 32 cols each
    uint32_t w0 = 0, w1 = 0;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = 4 * q + jj;
      const int sh = 8 * jj + 2 * q4;
      w0 |= (uint32_t)(d[4 * j + 0] > 0) << sh;
      w0 |= (uint32_t)(d[4 * j + 1] > 0) << (sh + 1);
      w1 |= (uint32_t)(d[4 * j + 2] > 0) << sh;
      w1 |= (uint32_t)(d[4 * j + 3] > 0) << (sh + 1);
    }
    w0 |= __shfl_xor_sync(0xffffffff, w0, 1);
    w0 |= __shfl_xor_sync(0xffffffff, w0, 2);
    w1 |= __shfl_xor_sync(0xffffffff, w1, 1);
    w1 |= __shfl_xor_sync(0xffffffff, w1, 2);
    const int wc = col0 / 32 + q;
    if ((q & 3) == q4 && wc < p.ldc_words) {
      // mask partial last word
      int nb = p.N - (col0 + 32 * q);
      uint32_t m = nb >= 32 ? 0xffffffffu : ((1u << nb) - 1u);
      if (r0 < p.M) Cb[(long long)r0 * p.ldc_words + wc] = w0 & m;
      if (r0 + 8 < p.M) Cb[(long long)(r0 + 8) * p.ldc_words + wc] = w1 & m;
    }
  }
}

__device__ __forceinline__ void store_counts(const int (&d)[128], const Params& p, int b, int row0, int col0) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x / 32) & 3;
  const int r0 = row0 + 16 * warp + lane / 4;
  int* Cb = p.counts + (long long)b * p.c_batch_stride;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    int c = col0 + 8 * j + 2 * (lane & 3);
    if (r0 < p.M) {
      if (c < p.N) Cb[(long long)r0 * p.N + c] = d[4 * j + 0];
      if (c + 1 < p.N) Cb[(long long)r0 * p.N + c + 1] = d[4 * j + 1];
    }
    if (r0 + 8 < p.M) {
      if (c < p.N) Cb[(long long)(r0 + 8) * p.N + c] = d[4 * j + 2];
      if (c + 1 < p.N) Cb[(long long)(r0 + 8) * p.N + c + 1] = d[4 * j + 3];
    }
  }
}

// Persistent kernel: each CTA loops over output tiles (tile = (batch, tm, tn)).
template <int STAGES, int MODE = 0, bool STAGGER = false>
__global__ void __launch_bounds__(384, 1)
    bgemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                 const __grid_constant__ CUtensorMap tmC, Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // align to 1024 for 128B swizzle atoms
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  SharedStorage<STAGES>& S = *reinterpret_cast<SharedStorage<STAGES>*>(base);

  const int wg = threadIdx.x / 128;  // 0 = producer, 1..2 = consumers
  const int KT = (p.Kbytes + BK_BYTES - 1) / BK_BYTES;
  const int ntiles = p.tiles_m * p.tiles_n * p.batch;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&S.full[s], 1);
      mbar_init(&S.empty[s], 8);  // one arrive per consumer warp
    }
    mbar_init(&S.go, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (threadIdx.x == 0) {
      int it = 0;  // global k-iteration counter across tiles
      for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int b, tm, tn;
        tile_coords(p, tile, b, tm, tn);
        for (int kt = 0; kt < KT; ++kt, ++it) {
          int s = it % STAGES;
          mbar_wait(&S.empty[s], ((it / STAGES) & 1) ^ 1);
          if (kt == 0) S.coord[s] = make_int4(b, tm, tn, 0);
          if (p.trace && blockIdx.x == 0 && it < 64) p.trace[2 * 64 * 4 + it] = clock64();
          if (MODE == 1 || MODE >= 3) {
            mbar_arrive(&S.full[s]);
          } else {
            mbar_expect_tx(&S.full[s], A_STAGE_BYTES + B_STAGE_BYTES);
            tma_load_3d(S.A[s], &tmA, &S.full[s], kt * BK_BYTES, tm * BM, b);
            tma_load_3d(S.B[s], &tmB, &S.full[s], kt * BK_BYTES, tn * BN, b);
          }
        }
      }
    }
  } else {
    setmaxnreg_inc<232>();
    const int cw = wg - 1;  // consumer index: rows cw*64 .. cw*64+63 of the tile
    int it = 0, ntile_local = 0;
    int d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0;
    if (STAGGER && cw == 1) mbar_wait(&S.go, 0);  // start one tile behind consumer 0
    const int my_tiles = ntiles > (int)blockIdx.x ? (ntiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    for (; ntile_local < my_tiles; ++ntile_local) {
      int b = 0, tm = 0, tn = 0;
      fence_operands(d);
      const bool tr = p.trace && blockIdx.x == 0 && (threadIdx.x & 127) == 0 && ntile_local < 64;
      long long* T = tr ? p.trace + (cw * 64 + ntile_local) * 4 : nullptr;
      if (tr) T[0] = clock64();
      for (int kt = 0; kt < KT; ++kt, ++it) {
        int s = it % STAGES;
        mbar_wait(&S.full[s], (it / STAGES) & 1);
        if (kt == 0) {
          int4 c = S.coord[s];
          b = c.x; tm = c.y; tn = c.z;
        }
        if (tr && kt == 0) T[1] = clock64();
        wg_fence();
        fence_operands(d);
        const uint8_t* a = S.A[s] + cw * 64 * BK_BYTES;
        const uint8_t* bb = S.B[s];
        if (MODE != 2) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            wgmma_b1_n256(d, desc_sw128(a + 32 * k), desc_sw128(bb + 32 * k), (kt | k) != 0);
          }
        }
        wg_commit();
        fence_operands(d);
        wg_wait<1>();
        if (kt > 0 && (threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
        if (STAGGER && cw == 0 && it == 1 && (threadIdx.x & 127) == 0) mbar_arrive(&S.go);
      }
      wg_wait<0>();
      fence_operands(d);
      if (tr) T[2] = clock64();
      if ((threadIdx.x & 31) == 0) mbar_arrive(&S.empty[(it - 1) % STAGES]);
      if (STAGGER && cw == 0 && it == 1 && (threadIdx.x & 127) == 0) mbar_arrive(&S.go);  // KT == 1 case
      if (MODE == 3) {
        if (d[0] == 0x7fffffff) p.C[0] = d[1];  // keep accumulators alive, skip epilogue
      } else if (p.counts) {
        store_counts(d, p, b, tm * BM + cw * 64, tn * BN);
      } else if (MODE == 4) {  // pack only
        uint32_t w[4];
        pack_bits(d, w);
        if ((w[0] ^ w[1] ^ w[2] ^ w[3]) == 0x12345679u) p.C[0] = 1;
      } else if (p.use_tma_store) {
        uint32_t w[4];
        pack_bits(d, w);
        const int buf = ntile_local & 1;
        uint8_t* cs = S.Cs[cw][buf];
        const bool leader = (threadIdx.x & 127) == 0;
        if (leader) bulk_wait_read<1>();  // the store issued from this buffer 2 tiles ago has read smem
        named_bar_sync(1 + cw, 128);
        const int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) & 3;
        const int r0 = 16 * warp + lane / 4;
        uint2* p0 = reinterpret_cast<uint2*>(cs + r0 * 32 + 8 * (lane & 3));
        uint2* p1 = reinterpret_cast<uint2*>(cs + (r0 + 8) * 32 + 8 * (lane & 3));
        *p0 = make_uint2(w[0], w[1]);
        *p1 = make_uint2(w[2], w[3]);
        fence_proxy_async();
        named_bar_sync(1 + cw, 128);
        if (leader && MODE != 5) {
          tma_store_3d(&tmC, cs, tn * (BN / 8), tm * BM + cw * 64, b);
          bulk_commit();
        }
      } else {
        // direct vectorized stores: each lane writes 2 adjacent words for 2 rows; a warp store
        // instruction covers 8 rows x 32 contiguous bytes = 8 full sectors.
        uint32_t w[4];
        pack_bits(d, w);
        const int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) & 3;
        const int r0 = tm * BM + cw * 64 + 16 * warp + lane / 4;
        const int wc = tn * (BN / 32) + 2 * (lane & 3);
        uint32_t* Cb = p.C + (long long)b * p.c_batch_stride;
        if (wc + 1 < p.ldc_words) {
          if (r0 < p.M) *reinterpret_cast<uint2*>(Cb + (long long)r0 * p.ldc_words + wc) = make_uint2(w[0], w[1]);
          if (r0 + 8 < p.M) *reinterpret_cast<uint2*>(Cb + (long long)(r0 + 8) * p.ldc_words + wc) = make_uint2(w[2], w[3]);
        } else if (wc < p.ldc_words) {
          if (r0 < p.M) Cb[(long long)r0 * p.ldc_words + wc] = w[0];
          if (r0 + 8 < p.M) Cb[(long long)(r0 + 8) * p.ldc_words + wc] = w[2];
        }
      }
      if (tr) T[3] = clock64();
    }
    if ((threadIdx.x & 127) == 0) bulk_wait<0>();
  }
}

}  // namespace bgemm
