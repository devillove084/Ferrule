#include "abi.h"

namespace ferrule::cpu {

// Paged causal GQA deliberately remains in the Rust reference provider until a
// native implementation can publish the complete semantic contract.
constexpr bool native_paged_gqa_available() noexcept { return false; }
static_assert(!native_paged_gqa_available());

} // namespace ferrule::cpu
