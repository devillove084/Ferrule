/* Host-only wrapper for the external NVTX3 header API. No CUDA headers or
 * context are used. NVTX3 dispatches to an injected tool, or is a no-op.
 * The external NVIDIA headers carry Apache-2.0 WITH LLVM-exception; they are
 * deliberately not copied into this repository. */
#include <nvtx3/nvToolsExt.h>

void *ferrule_nvtx_domain_create(void) {
  return nvtxDomainCreateA("ferrule_profile");
}

uint64_t ferrule_nvtx_range_start(void *domain, const char *label) {
  nvtxEventAttributes_t attributes = {0};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = label;
  return nvtxDomainRangeStartEx((nvtxDomainHandle_t)domain, &attributes);
}

void ferrule_nvtx_range_end(void *domain, uint64_t range) {
  nvtxDomainRangeEnd((nvtxDomainHandle_t)domain, range);
}
