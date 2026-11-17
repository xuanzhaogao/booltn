// perm4: tiled bit permutation of bit-packed tensors (shared by libbooltn and libcount8).
//
// A tensor of rank r is a 2^r-bit array in uint32 words, label position p contributing 2^p to the bit index
// (positions 0..4 are the bit within a word). count8 byte tensors use it as bit tensors with three more labels
// (the bit within the byte) that stay in place.
//
// Tile = labels {input positions < a} + {labels of output positions < b}, ts <= 18 labels (2^18-bit = 32 KB tiles).
// Tiles start at 2^16 bits and grow until both the loads and the stores move whole 128-byte lines (a, b >= 10):
// on the H100, runs of 32 bytes reach 0.6-0.8 TB/s, 64 bytes 1.1 TB/s, >= 128 bytes 1.6-1.7 TB/s, and lines
// shared by 4 blocks (even running at the same time) 1.0-1.2 TB/s. The algorithm is perm2's (booltn.cu): coalesced
// load -> smem -> in-word bit exchange of e labels (shuffles) + in-word delta swaps -> smem -> coalesced store.
// What makes it 1.4-1.65 TB/s instead of 0.6-0.9:
//  * few instructions per word: every tile map is linear in the bits of the tile word index j = tid + NT k, so
//    each splits into a per-thread base (summed once from per-bit tables) and a per-k part (constant-bank operand);
//    one exchange round is a shuffle, a funnel rotate and a LOP3 with per-thread constants;
//  * the next tile is loaded into registers while the current one is exchanged and stored;
//  * the outer (tile index) bits that complete the 128-byte lines of the input or the output vary fastest, so
//    blocks sharing a line run at the same time.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace p4 {

__device__ __forceinline__ int padw(int w) { return w + (w >> 5); }

struct Params {
  int ts, e, nt, wpt, padn;
  int ein[5];
  int nswap;
  int sw_d[5];
  uint32_t sw_m[5];
  int nouter_bits;
  long long nouter;
  long long out_in_w[32];   // outer bit -> input word offset (fastest outer bits complete 128-byte lines)
  long long out_out_w[32];  // outer bit -> output word offset
  uint32_t ldq[13], stq[13];  // tile word bit -> input / output word offset (within the tile's span)
  int sxq[13], dxq[13];       // exchange index bit -> smem word of the source / destination (unpadded)
  uint32_t kld[8], kst[8];    // tile word j = NT k: offsets
  int ksx[8], kdx[8];
};

template <int E, int WPT, int NT>
__global__ void __launch_bounds__(NT, 1024 / NT) kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                                   const __grid_constant__ Params P) {
  extern __shared__ uint32_t smem4[];
  uint32_t* sin_ = smem4;
  uint32_t* sout = smem4 + P.padn;
  const int tid = threadIdx.x, lane = tid & 31;
  const int ntb = 31 - __clz((int)blockDim.x);
  uint32_t ld0 = 0, st0 = 0;
  int sx0 = 0, dx0 = 0;
  for (int b = 0; b < ntb; ++b)
    if ((tid >> b) & 1) { ld0 += P.ldq[b]; st0 += P.stq[b]; sx0 += P.sxq[b]; dx0 += P.dxq[b]; }
  constexpr int ne = 1 << E;
  const int i = tid & (ne - 1);
  uint32_t rs[E > 0 ? E : 1], rm[E > 0 ? E : 1];
#pragma unroll
  for (int j = 0; j < E; ++j) {
    const int c = P.ein[j];
    const int sh = 1 << c;
    const uint32_t m = c == 0 ? 0x55555555u : c == 1 ? 0x33333333u : c == 2 ? 0x0F0F0F0Fu : c == 3 ? 0x00FF00FFu : 0x0000FFFFu;
    const bool upper = (i >> j) & 1;
    rs[j] = upper ? 32 - sh : sh;
    rm[j] = upper ? ~m : m;
  }
  const long long my_in = lane < P.nouter_bits ? P.out_in_w[lane] : 0;
  const long long my_out = lane < P.nouter_bits ? P.out_out_w[lane] : 0;
  auto outer = [&](long long o, long long mine) {
    long long v = ((o >> lane) & 1) ? mine : 0;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
  };
  uint32_t R[WPT];
  long long o = blockIdx.x;
  if (o < P.nouter) {
    const uint32_t* src = in + outer(o, my_in);
#pragma unroll
    for (int k = 0; k < WPT; ++k) R[k] = __ldg(src + (ld0 + P.kld[k]));
  }
  for (; o < P.nouter; o += gridDim.x) {
    // keep the per-thread bases opaque so the compiler does not hoist 4 x WPT sums into registers
    asm volatile("" : "+r"(ld0), "+r"(st0), "+r"(sx0), "+r"(dx0));
    uint32_t* dst = out + outer(o, my_out);
#pragma unroll
    for (int k = 0; k < WPT; ++k) sin_[padw(tid + NT * k)] = R[k];
    const long long o2 = o + gridDim.x;
    if (o2 < P.nouter) {
      const uint32_t* src = in + outer(o2, my_in);
#pragma unroll
      for (int k = 0; k < WPT; ++k) R[k] = __ldg(src + (ld0 + P.kld[k]));
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < WPT; ++k) {
      uint32_t x = sin_[padw(sx0 + P.ksx[k])];
#pragma unroll
      for (int j = 0; j < E; ++j) {
        const uint32_t other = __shfl_xor_sync(0xffffffff, x, 1 << j);
        const uint32_t rot = __funnelshift_l(other, other, rs[j]);
        x = (x & rm[j]) | (rot & ~rm[j]);
      }
#pragma unroll
      for (int q = 0; q < 5; ++q)
        if (q < P.nswap) {
          const uint32_t t = ((x >> P.sw_d[q]) ^ x) & P.sw_m[q];
          x ^= t ^ (t << P.sw_d[q]);
        }
      sout[padw(dx0 + P.kdx[k])] = x;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < WPT; ++k) dst[st0 + P.kst[k]] = sout[padw(tid + NT * k)];
  }
}

// host planning for perm4 (rin == rout == r, 10 <= r <= 37, no reduction). Returns false if not applicable.
static bool plan(Params& P, const int* src, int r, int maxts = 18) {
  if (r < 10 || r > 37) return false;
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
  // grow the input / output fast sets alternately; stop at 2^16-bit tiles once both reach 128-byte runs,
  // else grow up to 2^maxts bits
  for (int lim = 16; lim <= maxts; ++lim) {
    for (;;) {
      bool grown = false;
      for (int side = 0; side < 2; ++side) {
        int& v = (side == 0) == (a <= b) ? a : b;
        if (v >= r) continue;
        ++v;
        build();
        if ((int)S.size() > lim) { --v; build(); } else grown = true;
      }
      if (!grown) break;
    }
    if (std::min(a, b) >= 10) break;
  }
  build();
  std::vector<int> Tl;
  for (int q = 0; q < a; ++q) Tl.push_back(q);
  for (int q : S) if (q >= a) Tl.push_back(q);
  const int ts = (int)Tl.size();
  if (ts > 18) return false;
  P.ts = ts;
  std::vector<int> tl_of_in(r, -1);
  for (int k = 0; k < ts; ++k) tl_of_in[Tl[k]] = k;
  std::vector<int> Ol;
  for (int q = 0; q < r; ++q) if (tl_of_in[q] < 0) Ol.push_back(q);
  std::stable_sort(Ol.begin(), Ol.end(), [&](int x, int y) {
    const int kx = std::min(x, inv[x]) < 10 ? std::min(x, inv[x]) : 64 + x;
    const int ky = std::min(y, inv[y]) < 10 ? std::min(y, inv[y]) : 64 + y;
    return kx < ky;
  });
  P.nouter_bits = (int)Ol.size();
  if (P.nouter_bits > 32) return false;
  P.nouter = 1ll << P.nouter_bits;
  memset(P.out_in_w, 0, sizeof(P.out_in_w));
  memset(P.out_out_w, 0, sizeof(P.out_out_w));
  for (int j = 0; j < P.nouter_bits; ++j) {
    P.out_in_w[j] = 1ll << (Ol[j] - 5);
    P.out_out_w[j] = 1ll << (inv[Ol[j]] - 5);
  }
  memset(P.ldq, 0, sizeof(P.ldq)); memset(P.stq, 0, sizeof(P.stq));
  memset(P.sxq, 0, sizeof(P.sxq)); memset(P.dxq, 0, sizeof(P.dxq));
  for (int t = 0; t + 5 < ts; ++t) P.ldq[t] = 1u << (Tl[5 + t] - 5);
  std::vector<int> Eout, Ein;
  std::vector<char> out_inword(r, 0);
  for (int p = 0; p < 5; ++p) out_inword[src[p]] = 1;
  for (int k = 5; k < ts; ++k) if (out_inword[Tl[k]]) Eout.push_back(k);
  for (int q = 0; q < 5; ++q) if (!out_inword[q]) Ein.push_back(q);
  const int e = (int)Eout.size();
  if (e != (int)Ein.size()) return false;
  P.e = e;
  for (int j = 0; j < e; ++j) P.ein[j] = Ein[j];
  std::vector<int> Gk;
  for (int k = 5; k < ts; ++k) if (std::find(Eout.begin(), Eout.end(), k) == Eout.end()) Gk.push_back(k);
  for (int bq = 0; bq + 5 < ts; ++bq) P.sxq[bq] = bq < e ? 1 << (Eout[bq] - 5) : 1 << (Gk[bq - e] - 5);
  int arr[5];
  for (int x = 0; x < 5; ++x) arr[x] = x;
  for (int j = 0; j < e; ++j) arr[Ein[j]] = Tl[Eout[j]];
  P.nswap = 0;
  for (int p = 0; p < 5; ++p) {
    if (arr[p] == src[p]) continue;
    int x = p + 1;
    while (x < 5 && arr[x] != src[p]) ++x;
    if (x >= 5) return false;
    int d = (1 << x) - (1 << p);
    uint32_t m = 0;
    for (int t = 0; t < 32; ++t) if (((t >> p) & 1) && !((t >> x) & 1)) m |= 1u << t;
    P.sw_d[P.nswap] = d;
    P.sw_m[P.nswap] = m;
    ++P.nswap;
    std::swap(arr[p], arr[x]);
  }
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
  for (int bq = 0; bq + 5 < ts; ++bq) {
    const int lab = bq < e ? Ein[bq] : Tl[Gk[bq - e]];
    const int k2 = ol_of_in[lab];
    if (k2 < 5) return false;
    P.dxq[bq] = 1 << (k2 - 5);
  }
  for (int t = 0; t + 5 < ts; ++t) P.stq[t] = 1u << (inv[To[5 + t]] - 5);
  const int ntw = 1 << (ts - 5);
  if (ntw <= 256) { P.nt = ntw; P.wpt = 1; }
  else if (ntw <= 2048) { P.nt = 256; P.wpt = ntw / 256; }
  else { P.nt = ntw / 8; P.wpt = 8; }
  P.padn = ntw + ntw / 32;
  const int ntb = 31 - __builtin_clz(P.nt);
  for (int k = 0; k < 8; ++k) {
    uint32_t xl = 0, xs = 0;
    int xi = 0, xo = 0;
    for (int t = ntb; t + 5 < ts; ++t)
      if (((k << ntb) >> t) & 1) { xl += P.ldq[t]; xs += P.stq[t]; xi += P.sxq[t]; xo += P.dxq[t]; }
    P.kld[k] = xl; P.kst[k] = xs; P.ksx[k] = xi; P.kdx[k] = xo;
  }
  return true;
}

typedef void (*Kern)(const uint32_t*, uint32_t*, Params);
template <int E>
static Kern pick(int nt, int wpt) {
  if (nt <= 256) {
    switch (wpt) {
      case 8: return kernel<E, 8, 256>;
      case 4: return kernel<E, 4, 256>;
      case 2: return kernel<E, 2, 256>;
      default: return kernel<E, 1, 256>;
    }
  }
  return nt == 512 ? kernel<E, 8, 512> : kernel<E, 8, 1024>;
}
static Kern kern_of(const Params& P) {
  switch (P.e) {
    case 0: return pick<0>(P.nt, P.wpt);
    case 1: return pick<1>(P.nt, P.wpt);
    case 2: return pick<2>(P.nt, P.wpt);
    case 3: return pick<3>(P.nt, P.wpt);
    case 4: return pick<4>(P.nt, P.wpt);
    default: return pick<5>(P.nt, P.wpt);
  }
}

// Plan + kernel choice + launch geometry. Returns 1 if planned, 0 if not applicable, < 0 on a CUDA error.
struct Launch {
  Params P;
  Kern k = nullptr;
  size_t smem = 0;
  long long grid = 1;
  cudaError_t launch(const uint32_t* in, uint32_t* out, cudaStream_t st) const {
    k<<<(int)grid, P.nt, smem, st>>>(in, out, P);
    return cudaGetLastError();
  }
};
static int prepare(Launch& L, const int* src, int r, int nsm, int maxts = 18, int gridmul = 1) {
  if (!plan(L.P, src, r, maxts)) return 0;
  L.k = kern_of(L.P);
  L.smem = (size_t)2 * L.P.padn * 4;
  if (L.smem > 48 * 1024 && cudaFuncSetAttribute(L.k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smem) != cudaSuccess)
    return -1;
  int occ = 1;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, L.k, L.P.nt, L.smem) != cudaSuccess) return -1;
  L.grid = std::min<long long>(L.P.nouter, (long long)nsm * std::max(1, occ) * gridmul);
  return 1;
}

}  // namespace p4
