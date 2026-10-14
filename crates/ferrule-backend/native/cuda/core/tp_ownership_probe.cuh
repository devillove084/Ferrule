#pragma once
// Read-only diagnostic APIs; no reset, extra retain/release, or automatic tracing.
#include <cuda.h>
#include <atomic>
#include <cstring>
#include <stdint.h>

extern "C" int32_t ferrule_tp_probe_primary(uint32_t ordinal, uint32_t *flags, int32_t *active) {
  if (!flags || !active) return 1;
  CUdevice device = 0;
  auto code = cuInit(0);
  if (code != CUDA_SUCCESS) return static_cast<int32_t>(code);
  code = cuDeviceGet(&device, static_cast<int>(ordinal));
  if (code != CUDA_SUCCESS) return static_cast<int32_t>(code);
  // This API exposes flags/active ONLY. There is no refcount result.
  return static_cast<int32_t>(cuDevicePrimaryCtxGetState(device, flags, active));
}
extern "C" int32_t ferrule_tp_probe_core(uint64_t stream, uint64_t storage, uint64_t bytes) {
#if defined(FERRULE_TP_NCCL_HEADER) && defined(__linux__)
  using namespace ferrule::cuda::tp;
  if (bytes != storage_bytes) return 1;
  FerruleTpComm local{};
  local.stream = reinterpret_cast<cudaStream_t>(stream);
  local.storage = reinterpret_cast<uint16_t *>(storage);
  if (cudaMemsetAsync(local.storage, 0, bytes, local.stream) != cudaSuccess) return 9;
  // Same core CUDA module/kernels as NCCL warmup, WITHOUT loading/calling NCCL.
  return add(&local, max_count);
#else
  (void)stream; (void)storage; (void)bytes;
  return 2;
#endif
}
extern "C" int32_t ferrule_tp_probe_core_status(uint64_t storage, uint32_t *status) {
#if defined(FERRULE_TP_NCCL_HEADER) && defined(__linux__)
  auto *ptr = reinterpret_cast<uint16_t *>(storage) + 4 * ferrule::cuda::tp::max_count;
  return static_cast<int32_t>(cudaMemcpy(status, ptr, 4, cudaMemcpyDeviceToHost));
#else
  (void)storage; (void)status;
  return 2;
#endif
}

#if defined(__linux__) && __has_include(<cupti_callbacks.h>)
#include <cupti_callbacks.h>
#include <cupti_version.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
namespace ferrule::cuda::tp_probe {
// Aggregate SUCCESSFUL API callback counts, never a hidden/global refcount.
constexpr unsigned count = 18;
std::atomic<uint64_t> counters[count]{};
void *library = nullptr;
CUpti_SubscriberHandle subscriber{};
bool enabled = false;
inline bool prefix(const char *name, const char *value) {
  return std::strncmp(name, value, std::strlen(value)) == 0;
}
void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *data) {
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    if (id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) ++counters[0];
    if (id == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) ++counters[1];
    return;
  }
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  const auto *d = static_cast<const CUpti_CallbackData *>(data);
  if (d->callbackSite != CUPTI_API_EXIT) return;
  if (!d->functionReturnValue || !d->functionName) { ++counters[17]; return; }
  if (*static_cast<const CUresult *>(d->functionReturnValue) != CUDA_SUCCESS) { ++counters[16]; return; }
  const char *name = d->functionName;
  if (prefix(name, "cuDevicePrimaryCtxRetain")) ++counters[2];
  else if (prefix(name, "cuDevicePrimaryCtxRelease")) ++counters[3];
  else if (prefix(name, "cuCtxCreate")) ++counters[4];
  else if (prefix(name, "cuCtxDestroy")) ++counters[5];
  else if (prefix(name, "cuStreamCreate")) ++counters[6];
  else if (prefix(name, "cuStreamDestroy")) ++counters[7];
  else if ((prefix(name, "cuMemAlloc") && !prefix(name, "cuMemAllocHost"))) ++counters[8];
  else if (prefix(name, "cuMemFree") && !prefix(name, "cuMemFreeHost")) ++counters[9];
  else if (prefix(name, "cuMemCreate")) ++counters[10];
  else if (prefix(name, "cuMemRelease")) ++counters[11];
  else if (prefix(name, "cuMemMap")) ++counters[12];
  else if (prefix(name, "cuMemUnmap")) ++counters[13];
  else if (prefix(name, "cuModuleLoad")) ++counters[14];
  else if (prefix(name, "cuModuleUnload")) ++counters[15];
}
}
extern "C" int32_t ferrule_tp_probe_trace_start(const char *path) {
  using namespace ferrule::cuda::tp_probe;
  char resolved[PATH_MAX];
  if (enabled || !path || path[0] != '/' || !realpath(path, resolved) || strcmp(path, resolved) != 0) return 1;
  library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!library) return 2;
  auto version = reinterpret_cast<decltype(&cuptiGetVersion)>(dlsym(library, "cuptiGetVersion"));
  auto subscribe = reinterpret_cast<decltype(&cuptiSubscribe)>(dlsym(library, "cuptiSubscribe"));
  auto enable = reinterpret_cast<decltype(&cuptiEnableDomain)>(dlsym(library, "cuptiEnableDomain"));
  uint32_t actual = 0;
  Dl_info info{};
  if (!version || !subscribe || !enable || version(&actual) != CUPTI_SUCCESS || actual != CUPTI_API_VERSION ||
      !dladdr(reinterpret_cast<void *>(subscribe), &info) || !info.dli_fname ||
      !realpath(info.dli_fname, resolved) || strcmp(path, resolved) != 0) return 2;
  auto result = subscribe(&subscriber, callback, nullptr);
  if (result != CUPTI_SUCCESS) return 100 + static_cast<int32_t>(result);
  result = enable(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  if (result != CUPTI_SUCCESS) return 100 + static_cast<int32_t>(result);
  result = enable(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE);
  if (result != CUPTI_SUCCESS) return 100 + static_cast<int32_t>(result);
  // Diagnostic process scope: retain collector/DSO until independent process exit.
  enabled = true;
  return 0;
}
extern "C" int32_t ferrule_tp_probe_counts(uint64_t *out, uint32_t capacity) {
  using namespace ferrule::cuda::tp_probe;
  if (!enabled) return 2;
  if (!out || capacity != count) return 1;
  for (unsigned i = 0; i < count; ++i) out[i] = counters[i].load();
  return 0;
}
#else
extern "C" int32_t ferrule_tp_probe_trace_start(const char *) { return 2; }
extern "C" int32_t ferrule_tp_probe_counts(uint64_t *, uint32_t) { return 2; }
#endif
