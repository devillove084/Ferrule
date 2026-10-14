#pragma once
#include <cstdlib>
#include <cstring>
namespace ferrule::cuda::cutlass::bf16_gemmex_admission {
inline bool literal_config() {
  const char *value = std::getenv("CUBLAS_WORKSPACE_CONFIG");
  return value && std::strcmp(value, ":0:0") == 0;
}
// Evaluated during native-library startup, not lazily at first preparation.
// A later setenv cannot turn an invalid process launch into an admitted one.
inline const bool startup_config_valid = literal_config();
inline bool valid() { return startup_config_valid && literal_config(); }
}
extern "C" int32_t ferrule_bf16_gemmex_process_config_valid() {
  return ferrule::cuda::cutlass::bf16_gemmex_admission::valid() ? 1 : 0;
}
