// Host-only entry point. Including the production header checks compilation,
// but main never invokes CUDA, device queries, contexts, or kernel launches.
#include "core/recurrent_bf16_chunk.cuh"
#include <cstring>
#include <iostream>
using namespace ferrule::cuda::core::bf16_chunk;
int main() {
  std::cout << "{\"scales\":[";
  for (uint32_t dk = 1; dk <= 512; ++dk) {
    float scale = query_scale(dk);
    uint32_t bits;
    std::memcpy(&bits, &scale, sizeof(bits));
    if (dk != 1)
      std::cout << ',';
    std::cout << bits;
  }
  std::cout << "],\"windows\":[";
  bool first = true;
  for (uint32_t rows : {2u, 5u, 32u, 63u, 64u, 65u, 128u, 129u, 257u}) {
    for (uint32_t heads : {1u, 2u, 16u, 32u, 64u, 65u, 128u, 32768u, 32769u}) {
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << '[' << rows << ',' << heads << ','
                << prefix_window(rows, heads) << ']';
    }
  }
  std::cout << "]}\n";
}
