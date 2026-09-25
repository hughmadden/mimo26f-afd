// Stable, caller-owned B1 route plan. Numerical mode: E-W4A8-v1.
#pragma once
#include "prepared.cuh"
namespace m26b1 {
struct Group {
    int32_t expert, rows, route_base;
    int32_t input_rows[16];
};
static_assert(sizeof(Group) == 19 * sizeof(int32_t));
struct PlanInput {
    uint32_t capacity, rows;
    const int32_t* expert_ids; // token-major [rows,8], actual IDs in [0,256).
    const float* weights;     // unmodified FP32 routing weights, no application.
    uint64_t route_elements;
    const uint8_t* resident;  // Exactly 256 bytes; 0/1 presence by actual ID.
    uint64_t resident_bytes;
};
struct PlanStorage {
    int32_t* inverse;         // original route -> grouped route; -1 in tail.
    int32_t* original_routes; // grouped route -> original route; -1 in tail.
    float* grouped_weights;
    Group* groups;            // rows=0 and indices=-1 in all inactive groups.
    uint32_t* group_count;
    uint32_t* fault;
    uint64_t route_capacity, group_capacity;
};
// Buffers and metadata must stay immutable until the last consumer completes.
// Every successful replay overwrites all route/group capacity slots. On invalid
// value metadata only fault is written; consumers MUST check fault first.
Status plan_host(const PlanInput&, const PlanStorage&);
#ifdef __CUDACC__
cudaError_t plan_async(const PlanInput&, const PlanStorage&, cudaStream_t);
#endif
} // namespace m26b1
