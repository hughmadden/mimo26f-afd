// Compile-only instantiation; never launched by this prep cell.
// Numerical mode: E-W4A8-v1. No held FC1/FC2 or MMA code.
#include "staging.cuh"
__global__ void mimo26_b1_staging_compile_probe(m26b1::Pool pool, m26b1::Tile tile,
        const uint8_t* source, uint8_t* sink) {
    __shared__ __align__(16) uint8_t payload[192 * 64];
    __shared__ __align__(16) uint8_t scales[1024];
    const bool valid = m26b1::stage_async(pool, tile, source, payload, scales, threadIdx.x, blockDim.x);
    if (!valid || !tile.active) return; // Uniform predicate.
    m26b1::stage_commit_wait();
    if (threadIdx.x == 0) sink[blockIdx.x] = payload[0] ^ scales[0];
}
