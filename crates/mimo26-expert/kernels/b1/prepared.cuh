// B1 lattice-independent storage ABI. Numerical mode: E-W4A8-v1.
// New MiMo boundary types; byte mapping and overlap checks belong to the
// approved ds41rt prepare unit. Copyright (c) 2026 T.J. Purtell, MIT.
// Source identity and adaptation details are in prepare.cu and README.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include "../include/mimo26_slice_layout.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define M26B1_HD __host__ __device__
#else
#define M26B1_HD
#endif
namespace m26b1 {
constexpr const char* numerical_mode = "E-W4A8-v1";
constexpr uint32_t prepared_tag = 0x42315031; // B1P1; never canonical layout v2.
constexpr uint32_t hidden = M26X_HIDDEN;
constexpr uint32_t local_i = M26X_GATE_ROWS;
constexpr uint64_t image_bytes = M26X_QUARTER_SLICE_BYTES;
static_assert(hidden == 4096 && local_i == 512 && M26X_LAYOUT_VERSION == 2);
enum class Status : uint32_t { ok, version, geometry, size, pointer, overlap, bounds, metadata, nonfinite };
enum class Region : uint32_t { w13, s13, w2, s2 };
struct CanonicalInfo {
    uint32_t version, hidden_size, intermediate_size, rank;
    uint64_t bytes;
};
// Identity/hash checking belongs to the canonical loader. This metadata must
// describe the actual immutable upload, not a guessed version based on size.
struct PreparedInfo {
    uint32_t tag, source_version, hidden_size, intermediate_size, rank;
    uint64_t bytes;
};
M26B1_HD constexpr uint64_t region_bytes(Region r) {
    return r == Region::w13 ? 2097152 : r == Region::s13 ? 131072 :
           r == Region::w2 ? 1048576 : r == Region::s2 ? 65536 : 0;
}
M26B1_HD constexpr uint64_t region_offset(Region r) {
    return r == Region::w13 ? 0 : r == Region::s13 ? 2097152 :
           r == Region::w2 ? 2228224 : r == Region::s2 ? 3276800 : image_bytes;
}
M26B1_HD inline Status check_canonical(const CanonicalInfo& m) {
    if (m.version != M26X_LAYOUT_VERSION) return Status::version;
    if (m.hidden_size != hidden || m.intermediate_size != local_i || m.rank >= 4)
        return Status::geometry;
    return m.bytes == image_bytes ? Status::ok : Status::size;
}
M26B1_HD inline Status check_prepared(const PreparedInfo& m) {
    if (m.tag != prepared_tag || m.source_version != M26X_LAYOUT_VERSION) return Status::version;
    return check_canonical({m.source_version, m.hidden_size, m.intermediate_size, m.rank, m.bytes});
}
inline bool span_ok(const void* p, uint64_t bytes, uint64_t align = 1) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && !(address % align) && bytes <= UINTPTR_MAX - address;
}
inline bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
    if (!an || !bn) return false;
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
// Destination byte -> canonical byte. All arithmetic is uint64_t; no FP decode,
// nibble spreading, quantization or held compute primitive occurs here.
M26B1_HD inline uint64_t canonical_byte(Region region, uint64_t byte) {
    const bool scales = region == Region::s13 || region == Region::s2;
    const bool gated = region == Region::w13 || region == Region::s13;
    const uint64_t k = gated ? hidden : local_i;
    const uint64_t index = scales ? byte : byte / 4;
    const uint64_t tile_size = scales ? 1024 : 4096;
    const uint64_t tile = index / tile_size, lane = index % tile_size;
    uint64_t row, col;
    if (scales) {
        row = (tile / (k / 128)) * 256 + lane / 4;
        col = (tile % (k / 128)) * 4 + lane % 4;
    } else {
        const uint64_t combined = (lane >> 2) & 31;
        row = (tile / (k / 128)) * 256 + ((lane >> 7) & 7) * 32
            + (lane & 3) * 8 + (combined >> 2);
        col = (tile % (k / 128)) * 16 + (lane >> 10) * 4 + (combined & 3);
    }
    uint64_t base;
    if (gated) {
        const bool gate = row >= local_i;
        if (gate) row -= local_i;
        base = scales ? (gate ? M26X_GATE_SCALE_OFF : M26X_UP_SCALE_OFF)
                      : (gate ? M26X_GATE_PAYLOAD_OFF : M26X_UP_PAYLOAD_OFF);
    } else base = scales ? M26X_DOWN_SCALE_OFF : M26X_DOWN_PAYLOAD_OFF;
    return base + (scales ? row * (k / 32) + col : (row * (k / 8) + col) * 4 + byte % 4);
}
Status prepare_host(const CanonicalInfo&, const uint8_t*, uint64_t,
                    uint8_t*, uint64_t, PreparedInfo*);
#ifdef __CUDACC__
// Host metadata is returned only after enqueue succeeds. Caller must wait for
// the stream before consuming bytes or publishing this prepared allocation.
cudaError_t prepare_async(const CanonicalInfo&, const uint8_t*, uint64_t,
                          uint8_t*, uint64_t, PreparedInfo*, cudaStream_t);
#endif
} // namespace m26b1
