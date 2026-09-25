/* Layout-v2 expert CUDA ABI. No Rust dependency in standalone Spark drivers.
 * The old v1 ABI is intentionally removed: callers must pass explicit sizes,
 * version and immutable validated grouping metadata, not untyped old slices.
 */
#pragma once
#include <stdint.h>
#include "mimo26_slice_layout.h"
#include "mimo26_expert_bits.h"
#ifndef M26X_NO_CUDA_RUNTIME
#include <cuda_runtime.h>
typedef cudaStream_t m26x_stream_t;
#else
typedef void* m26x_stream_t;
typedef int cudaError_t;
#endif
#ifdef __cplusplus
extern "C" {
#endif

enum { M26X_PROJ_GATE = 0, M26X_PROJ_UP = 1, M26X_PROJ_DOWN = 2 };
enum { M26X_OUT_F32 = 0, M26X_OUT_BF16 = 1 };

/* Capacity class = source tokens before top-8 routing; total routed rows <=8C.
 * IDs/offsets point to device arrays of n_groups / n_groups+1 entries. Prepare
 * them from m26x_validate_host_plan-approved host arrays and keep them immutable
 * until all work in the stream completes. Device fault bits must be initialized
 * to zero and checked after completion; malformed metadata never reads weights.
 * Padded groups have NO metadata entries and must return before any load.
 * Buffer sizes are bytes, not dtype-dependent element counts.
 */
typedef struct {
    uint32_t layout_version;
    int32_t manifest_arch, manifest_sms, capacity_class;
    int32_t resident_experts, n_groups, padded_groups, total_tokens, max_m;
    uint64_t grouped_bytes, x_bytes, out_bytes, scratch_bytes;
    const int32_t* expert_ids;
    const int32_t* group_offsets;
    uint32_t* fault;
} m26x_plan;

/* Pure host metadata check, no GPU access. Returns zero on success. */
int m26x_validate_host_plan(const m26x_plan*, const int32_t* host_ids,
                          const int32_t* host_offsets);
cudaError_t m26x_check_aot(int32_t arch, int32_t sms, int32_t capacity, uint32_t naive);
cudaError_t m26x_device_identity(int32_t* arch, int32_t* sm_count);
cudaError_t m26x_grouped_gemm_v2(const m26x_plan*, const uint8_t* grouped,
    const float* x, int32_t proj, int32_t out_dtype, uint32_t naive,
    void* out, m26x_stream_t stream);
/* x [T,4096] -> partial [T,4096]; scratch holds TWO [T,512] f32 arrays.
 * Sum rank partials outside this primitive; routing weights are not applied here.
 */
cudaError_t m26x_expert_ffn_v2(const m26x_plan*, const uint8_t* grouped,
    const float* x, int32_t out_dtype, uint32_t naive, float* scratch,
    void* out, m26x_stream_t stream);
/* R18c phase-wise mixed-M decode dispatch (production; kernels/mixed_dispatch.cuh):
 * one kernel per phase over every active expert, decode widths 1..8. The exact-M
 * arithmetic bodies are verbatim from expert_gemm.cu and bitwise-identical to the
 * per-M launch. M>8 tiles remain a separate I5 item (the native-wide prefill
 * adapter), not this kernel.
 */
cudaError_t m26x_expert_ffn_mixed_v2(const m26x_plan*, const uint8_t* grouped,
    const float* x, int32_t out_dtype, uint32_t naive, float* scratch,
    void* out, m26x_stream_t stream);
/* Full-checkpoint decode, independent of TP slices: checked host dimensions,
 * source byte lengths and destination length. Samples must never be skipped.
 */
cudaError_t m26x_unpack_matrix(const uint8_t* payload, uint64_t payload_bytes,
    const uint8_t* scales, uint64_t scale_bytes, int32_t rows, int32_t cols,
    uint32_t naive, float* out, uint64_t out_bytes, m26x_stream_t stream);
#ifdef __cplusplus
}
#endif
