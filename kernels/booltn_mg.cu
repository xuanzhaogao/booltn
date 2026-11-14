// Multi-GPU additions to libbooltn: the same kernels plus device selection and an in-place exchange of
// buffer halves between two GPUs (redistribution of a tensor that is partitioned across GPUs).
//
// libbooltn keeps one stream per loaded library, so booltn.distributed loads one copy of this library per
// device (the file is copied under a different name) and calls bt_mg_init(dev) on each copy. The exchange
// functions take explicit device ids and may be called through any copy.
#include "booltn.cu"

namespace btmg {

static int g_dev = 0;

struct DevRes {
  cudaStream_t peer = nullptr, local = nullptr;
  void* stage[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};  // [direction][parity]
  size_t cap = 0;
};
static DevRes g_res[64];

static int ensure(int dev, size_t chunk) {
  DevRes& R = g_res[dev];
  BT_CHECK(cudaSetDevice(dev));
  if (!R.peer) {
    BT_CHECK(cudaStreamCreateWithFlags(&R.peer, cudaStreamNonBlocking));
    BT_CHECK(cudaStreamCreateWithFlags(&R.local, cudaStreamNonBlocking));
  }
  if (R.cap < chunk) {
    for (int d = 0; d < 2; ++d)
      for (int i = 0; i < 2; ++i) {
        if (R.stage[d][i]) BT_CHECK(cudaFree(R.stage[d][i]));
        BT_CHECK(cudaMalloc(&R.stage[d][i], chunk));
      }
    R.cap = chunk;
  }
  return 0;
}

__global__ void or_halves_kernel(uint32_t* p, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    p[i] |= p[i + n];
}

}  // namespace btmg

extern "C" {

// Bind this copy of the library to device `dev` (for the calling thread) and initialize it.
int bt_mg_init(int dev) {
  BT_CHECK(cudaSetDevice(dev));
  if (bt_init()) return -1;
  BT_CHECK(cudaDeviceGetAttribute(&bt::g_nsm, cudaDevAttrMultiProcessorCount, dev));
  cudaMemPool_t pool;
  BT_CHECK(cudaDeviceGetDefaultMemPool(&pool, dev));
  uint64_t thr = UINT64_MAX;
  BT_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  btmg::g_dev = dev;
  return 0;
}

int bt_mg_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return -1;
  return n;
}

// Direct peer-to-peer copies between devices a and b (NVLink / PCIe). Returns 1 if enabled, 0 if the
// devices cannot access each other directly (copies then go through the host), < 0 on error.
int bt_mg_enable_peer(int a, int b) {
  int ab = 0, ba = 0;
  BT_CHECK(cudaDeviceCanAccessPeer(&ab, a, b));
  BT_CHECK(cudaDeviceCanAccessPeer(&ba, b, a));
  if (!ab || !ba) return 0;
  for (int k = 0; k < 2; ++k) {
    BT_CHECK(cudaSetDevice(k ? b : a));
    cudaError_t e = cudaDeviceEnablePeerAccess(k ? a : b, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) BT_CHECK(e);
    cudaGetLastError();
  }
  BT_CHECK(cudaSetDevice(btmg::g_dev));
  return 1;
}

// Swap [a, a + bytes) on device da with [b, b + bytes) on device db, in place. Both directions run at the
// same time; chunks pass through two staging buffers per direction so that the local write-back of one
// chunk overlaps the transfer of the next. Synchronous; the caller must have synchronized the work that
// produces the buffers.
int bt_mg_swap(void* a, int da, void* b, int db, size_t bytes, size_t chunk) {
  using namespace btmg;
  if (chunk == 0) chunk = (size_t)256 << 20;
  if (chunk > bytes) chunk = bytes;
  if (bytes == 0) return 0;
  if (ensure(da, chunk)) return -1;
  if (ensure(db, chunk)) return -1;
  DevRes& A = g_res[da];
  DevRes& B = g_res[db];
  const size_t n = (bytes + chunk - 1) / chunk;
  char* pa = (char*)a;
  char* pb = (char*)b;
  auto len = [&](size_t i) { return std::min(chunk, bytes - i * chunk); };
  for (size_t i = 0; i <= n; ++i) {
    if (i > 0) {  // transfers of chunk i-1 have arrived
      BT_CHECK(cudaSetDevice(da));
      BT_CHECK(cudaStreamSynchronize(A.peer));
      BT_CHECK(cudaSetDevice(db));
      BT_CHECK(cudaStreamSynchronize(B.peer));
    }
    if (i > 1) {  // write-backs of chunk i-2 are done: its staging buffers are free again
      BT_CHECK(cudaSetDevice(da));
      BT_CHECK(cudaStreamSynchronize(A.local));
      BT_CHECK(cudaSetDevice(db));
      BT_CHECK(cudaStreamSynchronize(B.local));
    }
    if (i < n) {  // send chunk i both ways into the staging buffers of parity i
      const size_t off = i * chunk, m = len(i);
      BT_CHECK(cudaSetDevice(da));
      if (da == db)
        BT_CHECK(cudaMemcpyAsync(B.stage[0][i & 1], pa + off, m, cudaMemcpyDeviceToDevice, A.peer));
      else
        BT_CHECK(cudaMemcpyPeerAsync(B.stage[0][i & 1], db, pa + off, da, m, A.peer));
      BT_CHECK(cudaSetDevice(db));
      if (da == db)
        BT_CHECK(cudaMemcpyAsync(A.stage[1][i & 1], pb + off, m, cudaMemcpyDeviceToDevice, A.local));
      else
        BT_CHECK(cudaMemcpyPeerAsync(A.stage[1][i & 1], da, pb + off, db, m, B.peer));
    }
    if (i > 0) {  // write chunk i-1 back from staging
      const size_t off = (i - 1) * chunk, m = len(i - 1);
      if (da == db) {  // one device: everything in order on two streams, wait before reusing them
        BT_CHECK(cudaStreamSynchronize(A.peer));
        BT_CHECK(cudaStreamSynchronize(A.local));
        BT_CHECK(cudaMemcpyAsync(pa + off, A.stage[1][(i - 1) & 1], m, cudaMemcpyDeviceToDevice, A.peer));
        BT_CHECK(cudaMemcpyAsync(pb + off, B.stage[0][(i - 1) & 1], m, cudaMemcpyDeviceToDevice, A.local));
        BT_CHECK(cudaStreamSynchronize(A.peer));
        BT_CHECK(cudaStreamSynchronize(A.local));
      } else {
        BT_CHECK(cudaSetDevice(da));
        BT_CHECK(cudaMemcpyAsync(pa + off, A.stage[1][(i - 1) & 1], m, cudaMemcpyDeviceToDevice, A.local));
        BT_CHECK(cudaSetDevice(db));
        BT_CHECK(cudaMemcpyAsync(pb + off, B.stage[0][(i - 1) & 1], m, cudaMemcpyDeviceToDevice, B.local));
      }
    }
  }
  BT_CHECK(cudaSetDevice(da));
  BT_CHECK(cudaStreamSynchronize(A.local));
  BT_CHECK(cudaSetDevice(db));
  BT_CHECK(cudaStreamSynchronize(B.local));
  BT_CHECK(cudaSetDevice(g_dev));
  return 0;
}

// p[i] |= p[i + n] for i < n words: merge the partial sums of a GPU pair after an exchange (on this
// library's stream, in order with the plans).
int bt_mg_or_halves(uint32_t* p, long long n) {
  btmg::or_halves_kernel<<<(int)bt::grid_for(n), 256, 0, bt::g_stream>>>(p, n);
  BT_CHECK(cudaGetLastError());
  return 0;
}

int bt_mg_mem_info(size_t* free_b, size_t* total_b) {
  BT_CHECK(cudaMemGetInfo(free_b, total_b));
  return 0;
}

}  // extern "C"
