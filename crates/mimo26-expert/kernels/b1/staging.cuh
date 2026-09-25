// Numerical mode: E-W4A8-v1. No decode, container transform or MMA.
// Exact-width addressing adapted from b12x/moe/_shared/kernels/w4a8_staging.py
// at 3882b935ede761d6c73a5d6fd68e690f1e3f5380; SHA256
// ab95db03abdc06152ab47878cfa4e2e7b7bb19f450516d76ad472f04f1e147f0.
// Apache-2.0; see LICENSE.b12x. Changed: handwritten CUDA, MiMo tagged pool,
// checked tails/64-bit extents; caller owns synchronization and shared storage.
#pragma once
#include "prepared.cuh"
namespace m26b1 {
struct Pool {
    PreparedInfo info;
    uint64_t bytes;
    uint32_t slots; // Physical expert slots, not count of active groups.
};
struct Tile {
    uint32_t slot, n_start, k_start, width;
    bool down, gate, active;
};
M26B1_HD inline Status check_tile(const Pool& pool, const Tile& t) {
    // Padded groups must return before examining arbitrary poison metadata.
    if (!t.active) return Status::ok;
    auto status = check_prepared(pool.info);
    if (status != Status::ok) return status;
    if (!pool.slots || pool.slots > 47 * 256 || t.slot >= pool.slots ||
        pool.bytes < uint64_t(pool.slots) * image_bytes) return Status::bounds;
    if (t.width != 64 && t.width != 128 && t.width != 192) return Status::geometry;
    if (t.n_start % 32 || t.k_start % 32 || (!t.down && t.k_start % 128)) return Status::geometry;
    if (t.down) {
        if (t.gate || t.n_start > hidden - 128 || t.k_start > local_i - t.width)
            return Status::bounds;
    } else {
        if (t.k_start > hidden - 128 || t.n_start > local_i - t.width)
            return Status::bounds;
    }
    return Status::ok;
}
M26B1_HD inline uint32_t payload_transfers(const Tile& t) {
    return t.active ? t.width * 4 : 0; // 16-byte copies, width * 128 / 2 bytes.
}
M26B1_HD inline uint32_t staged_scale_bytes(const Tile& t) {
    return !t.active ? 0 : t.down ? ((t.width + 127) / 128) * 128 * 4 : t.width * 4;
}
// Called only after check_tile; returns offsets within the tagged pool, not a
// pointer. Keeping integer offsets makes >4 GiB address tests allocation-free.
M26B1_HD inline uint64_t payload_source(const Tile& t, uint32_t task) {
    const uint64_t lane = task % 32;
    uint64_t n, k, word;
    if (!t.down) {
        const uint64_t chunks = t.width / 32;
        n = t.n_start + (t.gate ? local_i : 0) + ((task / 32) % chunks) * 32;
        k = t.k_start / 128;
        word = ((n / 256) * (hidden / 128) + k) * 4096
             + (task / (32 * chunks)) * 1024 + ((n % 256) / 32) * 128 + lane * 4;
    } else {
        n = t.n_start + ((task / 32) % 4) * 32;
        k = t.k_start / 32 + task / 128;
        word = ((n / 256) * (local_i / 128) + k / 4) * 4096
             + (k % 4) * 1024 + ((n % 256) / 32) * 128 + lane * 4;
    }
    return uint64_t(t.slot) * image_bytes + region_offset(t.down ? Region::w2 : Region::w13) + word * 4;
}
// Logical staged scale byte -> source byte; absent bytes in the last packed
// K32 scale word are ZERO padding. They never generate a global memory load.
M26B1_HD inline bool scale_source(const Tile& t, uint32_t index, uint64_t& offset) {
    uint64_t n, k;
    if (t.down) {
        const uint32_t word = index / 4;
        const uint32_t block = (word / 128) * 4 + index % 4;
        if (block >= t.width / 32) return false;
        n = t.n_start + word % 128;
        k = t.k_start / 32 + block;
    } else {
        n = t.n_start + (t.gate ? local_i : 0) + index / 4;
        k = t.k_start / 32 + index % 4;
    }
    const uint64_t k_tiles = t.down ? local_i / 128 : hidden / 128;
    const uint64_t byte = ((n / 256) * k_tiles + k / 4) * 1024 + (n % 256) * 4 + k % 4;
    offset = uint64_t(t.slot) * image_bytes + region_offset(t.down ? Region::s2 : Region::s13) + byte;
    return true;
}
inline Status stage_host(const Pool& pool, const Tile& t, const uint8_t* source,
        uint8_t* payload, uint64_t payload_bytes, uint8_t* scales, uint64_t scale_bytes) {
    auto status = check_tile(pool, t);
    if (status != Status::ok || !t.active) return status;
    const uint64_t pb = uint64_t(payload_transfers(t)) * 16, sb = staged_scale_bytes(t);
    if (payload_bytes != pb || scale_bytes != sb) return Status::size;
    if (!span_ok(source, pool.bytes, 16) || !span_ok(payload, pb, 16) || !span_ok(scales, sb, 16))
        return Status::pointer;
    if (overlaps(source, pool.bytes, payload, pb) || overlaps(source, pool.bytes, scales, sb) ||
        overlaps(payload, pb, scales, sb)) return Status::overlap;
    for (uint32_t i = 0; i < payload_transfers(t); ++i)
        for (uint32_t j = 0; j < 16; ++j) payload[i * 16 + j] = source[payload_source(t, i) + j];
    for (uint32_t i = 0; i < sb; ++i) {
        uint64_t offset = 0;
        scales[i] = scale_source(t, i, offset) ? source[offset] : 0;
    }
    return Status::ok;
}
#ifdef __CUDACC__
// Collective: tile/pool/active/threads must be block-uniform. Caller provides
// exactly payload_transfers(t)*16 and staged_scale_bytes(t) shared bytes,
// aligned to 16, and source spanning pool.bytes. No pointer is formed inactive.
__device__ inline bool stage_async(const Pool& pool, const Tile& t, const uint8_t* source,
        uint8_t* shared_payload, uint8_t* shared_scales, uint32_t tid, uint32_t threads) {
    if (check_tile(pool, t) != Status::ok || !threads) return false;
    if (!t.active) return true;
    for (uint32_t task = tid; task < payload_transfers(t); task += threads) {
        const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(shared_payload + task * 16));
        const auto* src = source + payload_source(t, task);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    for (uint32_t i = tid; i < staged_scale_bytes(t); i += threads) {
        uint64_t offset = 0;
        shared_scales[i] = scale_source(t, i, offset) ? source[offset] : 0;
    }
    return true;
}
__device__ inline void stage_commit_wait() {
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}
#endif
} // namespace m26b1
