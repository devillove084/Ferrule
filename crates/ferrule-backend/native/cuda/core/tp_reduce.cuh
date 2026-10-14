#pragma once
#include "core/tp_reduce_abi.h"
#include "core/device.cuh"
#include <cuda.h>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <new>

#if defined(FERRULE_TP_NCCL_HEADER) && defined(__linux__)
#include <nccl.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
static_assert(NCCL_VERSION_CODE == 22705, "TP requires the pinned NCCL 2.27.5 header");
static_assert(sizeof(ncclUniqueId) == sizeof(FerruleTpUniqueId), "NCCL unique ID ABI");
extern char **environ;

struct FerruleTpLibrary {
  std::atomic<unsigned> refs{1};
  void *dso = nullptr;
  decltype(&ncclGetVersion) version = nullptr;
  decltype(&ncclGetUniqueId) unique_id = nullptr;
  decltype(&ncclCommInitRankConfig) init = nullptr;
  decltype(&ncclCommGetAsyncError) async = nullptr;
  decltype(&ncclAllGather) gather = nullptr;
  decltype(&ncclCommAbort) abort = nullptr;
  decltype(&ncclCommDestroy) destroy = nullptr;
};
struct FerruleTpComm {
  FerruleTpLibrary *library = nullptr;
  ncclComm_t comm = nullptr;
  CUcontext context = nullptr;
  cudaStream_t stream = nullptr;
  uint16_t *storage = nullptr;
  uint32_t pending_count = 0;
  bool initializing = true;
  bool unknown = false;
  bool warmed = false;
};
namespace ferrule::cuda::tp {
constexpr uint32_t max_count = 16384;
constexpr size_t storage_bytes = (4 * max_count + 2) * sizeof(uint16_t);
inline uint16_t *gathered(FerruleTpComm *c) { return c->storage + max_count; }
inline uint16_t *output(FerruleTpComm *c) { return c->storage + 3 * max_count; }
inline uint32_t *flag(FerruleTpComm *c) {
  return reinterpret_cast<uint32_t *>(c->storage + 4 * max_count);
}
inline bool owner(FerruleTpComm *c) {
  CUcontext current = nullptr;
  return c && cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == c->context;
}
inline bool pointer_owner(uint64_t address, uint64_t bytes, CUcontext owner_context) {
  CUcontext context = nullptr;
  CUdeviceptr base = 0;
  size_t size = 0;
  return address != 0 && address % 2 == 0 &&
      cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, address) == CUDA_SUCCESS &&
      context == owner_context && cuMemGetAddressRange(&base, &size, address) == CUDA_SUCCESS &&
      address >= base && address - base <= size && bytes <= size - (address - base);
}
inline int32_t unknown(FerruleTpComm *c) {
  c->unknown = true;
  return FERRULE_TP_UNKNOWN;
}
inline int32_t nccl_error(FerruleTpComm *c, ncclResult_t result, const char *operation) {
  auto text = reinterpret_cast<decltype(&ncclGetErrorString)>(dlsym(c->library->dso, "ncclGetErrorString"));
  fprintf(stderr, "TP_NCCL_ERROR operation=%s result=%d message=%s\n", operation,
          static_cast<int>(result), text ? text(result) : "error-string symbol unavailable");
  return unknown(c);
}
inline void release_library(FerruleTpLibrary *library) {
  if (library && library->refs.fetch_sub(1) == 1) {
    if (library->dso) dlclose(library->dso);
    delete library;
  }
}
// Validation precedes the publishing kernel on the SAME compute stream. The
// sticky word is never reset between sites, and rejection leaves output untouched.
__global__ void validate_two(const uint16_t *values, uint32_t count, uint32_t *status) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t a = values[i], b = values[count + i];
  if ((a & 0x7f80u) == 0x7f80u || (b & 0x7f80u) == 0x7f80u) {
    atomicOr(status, 1u);
    return;
  }
  const float sum = __fadd_rn(core::bf16_value(a), core::bf16_value(b));
  if (!isfinite(sum) || (core::bf16_rne(sum) & 0x7f80u) == 0x7f80u) atomicOr(status, 2u);
}
__global__ void ordered_add_two(const uint16_t *values, uint16_t *out,
                               uint32_t count, const uint32_t *status) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && *status == 0) {
    // Exactly FIRST member + SECOND member, not 0 + FIRST + SECOND or NCCL SUM.
    const float sum = __fadd_rn(core::bf16_value(values[i]), core::bf16_value(values[count + i]));
    out[i] = core::bf16_rne(sum);
  }
}
inline int32_t add(FerruleTpComm *c, uint32_t count) {
  const uint32_t blocks = (count + 255) / 256;
  validate_two<<<blocks, 256, 0, c->stream>>>(gathered(c), count, flag(c));
  if (cudaGetLastError() != cudaSuccess) return unknown(c);
  ordered_add_two<<<blocks, 256, 0, c->stream>>>(gathered(c), output(c), count, flag(c));
  if (cudaGetLastError() != cudaSuccess) return unknown(c);
  c->pending_count = 0;
  return FERRULE_TP_OK;
}
}
using namespace ferrule::cuda::tp;
extern "C" size_t ferrule_tp_library_host_bytes(void) { return sizeof(FerruleTpLibrary); }
extern "C" size_t ferrule_tp_comm_host_bytes(void) { return sizeof(FerruleTpComm); }
extern "C" int32_t ferrule_tp_library_open(const char *path, FerruleTpLibrary **out) {
  if (!out) return FERRULE_TP_INVALID;
  *out = nullptr;
  char canonical[PATH_MAX];
  if (!path || path[0] != '/' || !realpath(path, canonical) || strcmp(path, canonical) != 0)
    return FERRULE_TP_INVALID;
  // No process-global environment mutation. Private NCCL defaults/config-file
  // allocations remain explicitly unknown even with these env overrides rejected.
  for (char **entry = environ; entry && *entry; ++entry)
    if (strncmp(*entry, "NCCL_", 5) == 0) return FERRULE_TP_INVALID;
  auto *l = new (std::nothrow) FerruleTpLibrary;
  if (!l) return FERRULE_TP_UNAVAILABLE;
  l->dso = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#define FERRULE_TP_SYMBOL(field, name) \
  l->field = reinterpret_cast<decltype(l->field)>(dlsym(l->dso, #name)); \
  if (!l->field) { release_library(l); return FERRULE_TP_UNAVAILABLE; }
  if (!l->dso) { fprintf(stderr, "TP_NCCL_LOADER path=%s error=%s\n", path, dlerror()); release_library(l); return FERRULE_TP_UNAVAILABLE; }
  FERRULE_TP_SYMBOL(version, ncclGetVersion)
  FERRULE_TP_SYMBOL(unique_id, ncclGetUniqueId)
  FERRULE_TP_SYMBOL(init, ncclCommInitRankConfig)
  FERRULE_TP_SYMBOL(async, ncclCommGetAsyncError)
  FERRULE_TP_SYMBOL(gather, ncclAllGather)
  FERRULE_TP_SYMBOL(abort, ncclCommAbort)
  FERRULE_TP_SYMBOL(destroy, ncclCommDestroy)
#undef FERRULE_TP_SYMBOL
  int version = 0;
  Dl_info info{};
  if (l->version(&version) != ncclSuccess || version != 22705 ||
      dladdr(reinterpret_cast<void *>(l->version), &info) == 0 || !info.dli_fname ||
      !realpath(info.dli_fname, canonical) || strcmp(path, canonical) != 0) {
    release_library(l); return FERRULE_TP_UNAVAILABLE;
  }
  *out = l;
  return FERRULE_TP_OK;
}
extern "C" void ferrule_tp_library_close(FerruleTpLibrary *library) { release_library(library); }
extern "C" int32_t ferrule_tp_unique_id(FerruleTpLibrary *l, FerruleTpUniqueId *out) {
  if (!l || !out) return FERRULE_TP_INVALID;
  ncclUniqueId id{};
  if (l->unique_id(&id) != ncclSuccess) return FERRULE_TP_UNAVAILABLE;
  memcpy(out, &id, sizeof(id));
  return FERRULE_TP_OK;
}
extern "C" int32_t ferrule_tp_comm_create(FerruleTpLibrary *l, const FerruleTpUniqueId *id,
    uint32_t rank, uint64_t stream, uint64_t storage, uint64_t bytes, FerruleTpComm **out) {
  if (!out) return FERRULE_TP_INVALID;
  *out = nullptr;
  CUcontext context = nullptr;
  if (!l || !id || rank > 1 || bytes != storage_bytes || storage % 4 != 0 ||
      cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
      !pointer_owner(storage, bytes, context)) return FERRULE_TP_INVALID;
  CUcontext stream_context = nullptr;
  if (stream == 0 || cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) != CUDA_SUCCESS ||
      stream_context != context) return FERRULE_TP_INVALID;
  auto *c = new (std::nothrow) FerruleTpComm;
  if (!c) return FERRULE_TP_UNAVAILABLE;
  c->library = l; l->refs.fetch_add(1);
  c->context = context; c->stream = reinterpret_cast<cudaStream_t>(stream);
  c->storage = reinterpret_cast<uint16_t *>(storage);
  *out = c; // Custody must survive any partially initialized/error outcome.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0; // No waiting for an absent/failed peer, no watchdog or poll loop.
  config.minCTAs = 2; config.maxCTAs = 2; config.cgaClusterSize = 1;
  config.collnetEnable = 0; config.netName = "Socket";
  ncclUniqueId native_id{}; memcpy(&native_id, id, sizeof(native_id));
  const ncclResult_t result = l->init(&c->comm, 2, native_id, static_cast<int>(rank), &config);
  if (result != ncclSuccess && result != ncclInProgress) return nccl_error(c, result, "ncclCommInitRankConfig");
  c->initializing = result == ncclInProgress;
  return c->initializing ? FERRULE_TP_PENDING : FERRULE_TP_OK;
}
extern "C" int32_t ferrule_tp_comm_async_status(FerruleTpComm *c) {
  if (!owner(c)) return FERRULE_TP_INVALID;
  if (c->unknown || !c->comm) return FERRULE_TP_UNKNOWN;
  ncclResult_t error = ncclSuccess;
  if (c->library->async(c->comm, &error) != ncclSuccess) return unknown(c);
  if (error == ncclInProgress) return FERRULE_TP_PENDING;
  if (error != ncclSuccess) return nccl_error(c, error, "ncclCommGetAsyncError");
  c->initializing = false;
  return FERRULE_TP_OK;
}
extern "C" int32_t ferrule_tp_allgather_bf16_two(FerruleTpComm *c,
    uint64_t input, uint64_t bytes, uint32_t count) {
  if (!owner(c)) return FERRULE_TP_INVALID;
  if (c->unknown) return FERRULE_TP_UNKNOWN;
  if (c->initializing || c->pending_count || !c->warmed || count == 0 || count > max_count ||
      bytes != static_cast<uint64_t>(count) * 2 || !pointer_owner(input, bytes, c->context))
    return FERRULE_TP_INVALID;
  const uint64_t begin = reinterpret_cast<uint64_t>(c->storage);
  if (input < begin + storage_bytes && begin < input + bytes) return FERRULE_TP_INVALID;
  c->pending_count = count;
  const auto result = c->library->gather(reinterpret_cast<const void *>(input), gathered(c),
      count, ncclBfloat16, c->comm, c->stream);
  if (result == ncclInProgress) return FERRULE_TP_PENDING;
  if (result != ncclSuccess) return nccl_error(c, result, "ncclAllGather");
  return add(c, count);
}
extern "C" int32_t ferrule_tp_comm_warmup(FerruleTpComm *c) {
  if (!owner(c)) return FERRULE_TP_INVALID;
  if (c->unknown) return FERRULE_TP_UNKNOWN;
  if (c->initializing || c->pending_count || c->warmed) return FERRULE_TP_INVALID;
  if (cudaMemsetAsync(c->storage, 0, storage_bytes, c->stream) != cudaSuccess) return unknown(c);
  c->warmed = true;
  c->pending_count = max_count;
  const auto result = c->library->gather(c->storage, gathered(c), max_count, ncclBfloat16, c->comm, c->stream);
  if (result == ncclInProgress) return FERRULE_TP_PENDING;
  if (result != ncclSuccess) return nccl_error(c, result, "ncclAllGather");
  return add(c, max_count);
}
extern "C" int32_t ferrule_tp_comm_progress(FerruleTpComm *c) {
  const auto status = ferrule_tp_comm_async_status(c);
  if (status != FERRULE_TP_OK) return status;
  return c->pending_count ? add(c, c->pending_count) : FERRULE_TP_OK;
}
extern "C" int32_t ferrule_tp_comm_read_status(FerruleTpComm *c, uint32_t *out) {
  if (!owner(c) || !out || c->initializing || c->pending_count || !c->warmed) return FERRULE_TP_INVALID;
  if (c->unknown) return FERRULE_TP_UNKNOWN;
  // No payload D2H. Rust has already proved the compute event complete.
  return cudaMemcpy(out, flag(c), sizeof(*out), cudaMemcpyDeviceToHost) == cudaSuccess
      ? FERRULE_TP_OK : unknown(c);
}
extern "C" int32_t ferrule_tp_comm_abort(FerruleTpComm *c) {
  if (!owner(c)) return FERRULE_TP_INVALID;
  c->unknown = true;
  if (c->comm) {
    const auto result = c->library->abort(c->comm);
    if (result != ncclSuccess) return FERRULE_TP_UNKNOWN;
    c->comm = nullptr;
  }
  // Deliberately NOT delete c or release the DSO. CUDA storage/context still live.
  return FERRULE_TP_UNKNOWN;
}
extern "C" int32_t ferrule_tp_comm_destroy(FerruleTpComm *c) {
  if (!owner(c)) return FERRULE_TP_INVALID;
  if (c->unknown || c->initializing || c->pending_count) return FERRULE_TP_UNKNOWN;
  const auto tail = cudaStreamQuery(c->stream);
  if (tail == cudaErrorNotReady) return FERRULE_TP_PENDING;
  if (tail != cudaSuccess) return unknown(c);
  if (ferrule_tp_comm_async_status(c) != FERRULE_TP_OK) return unknown(c);
  if (c->comm && c->library->destroy(c->comm) != ncclSuccess) return unknown(c);
  release_library(c->library); delete c;
  return FERRULE_TP_OK;
}
#else
// CUDA builds without the exact optional local header never acquire a provider.
extern "C" size_t ferrule_tp_library_host_bytes(void) { return 0; }
extern "C" size_t ferrule_tp_comm_host_bytes(void) { return 0; }
extern "C" int32_t ferrule_tp_library_open(const char *, FerruleTpLibrary **out) {
  if (out) *out = nullptr;
  return FERRULE_TP_UNAVAILABLE;
}
extern "C" void ferrule_tp_library_close(FerruleTpLibrary *) {}
extern "C" int32_t ferrule_tp_unique_id(FerruleTpLibrary *, FerruleTpUniqueId *) { return 2; }
extern "C" int32_t ferrule_tp_comm_create(FerruleTpLibrary *, const FerruleTpUniqueId *, uint32_t,
    uint64_t, uint64_t, uint64_t, FerruleTpComm **out) { if (out) *out = nullptr; return 2; }
extern "C" int32_t ferrule_tp_comm_async_status(FerruleTpComm *) { return 2; }
extern "C" int32_t ferrule_tp_comm_warmup(FerruleTpComm *) { return 2; }
extern "C" int32_t ferrule_tp_allgather_bf16_two(FerruleTpComm *, uint64_t, uint64_t, uint32_t) { return 2; }
extern "C" int32_t ferrule_tp_comm_progress(FerruleTpComm *) { return 2; }
extern "C" int32_t ferrule_tp_comm_read_status(FerruleTpComm *, uint32_t *) { return 2; }
extern "C" int32_t ferrule_tp_comm_abort(FerruleTpComm *) { return 2; }
extern "C" int32_t ferrule_tp_comm_destroy(FerruleTpComm *) { return 2; }
#endif
