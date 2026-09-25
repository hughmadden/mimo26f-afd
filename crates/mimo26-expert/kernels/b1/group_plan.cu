// Stable group/inverse-map algorithm adapted from
// b12x/moe/_shared/kernels/v41_route_plan.py:14-118,
// 3882b935ede761d6c73a5d6fd68e690f1e3f5380; SHA256
// f5ee61e9e0e3c8cbbf0c6aa50af74be57d33d32b88b09f69debe93c996175da4.
// Apache-2.0; see LICENSE.b12x. Numerical mode: E-W4A8-v1.
// Changes: top-8 / E 256, O(routes) bounded storage instead of E*routes scratch,
// fail-loud metadata, no atomics. GPU is a SERIAL correctness scaffold, not a
// performance-qualified planner. Do not include its timings in a B1 claim.
#include "group_plan.cuh"
namespace m26b1 {
static Status validate_plan(const PlanInput& p, const PlanStorage& s) {
    if ((p.capacity != 256 && p.capacity != 2048 && p.capacity != 4096) || p.rows > p.capacity)
        return Status::geometry;
    const uint64_t cap = uint64_t(p.capacity) * 8, routes = uint64_t(p.rows) * 8;
    if (p.route_elements < routes || p.route_elements > cap || p.resident_bytes != 256 ||
        s.route_capacity != cap || s.group_capacity != cap) return Status::size;
    const void* inputs[] = {p.expert_ids, p.weights, p.resident};
    const uint64_t ib[] = {p.route_elements * 4, p.route_elements * 4, 256};
    for (int i = 0; i < 3; ++i)
        if (ib[i] && !span_ok(inputs[i], ib[i], i == 2 ? 1 : 4)) return Status::pointer;
    const void* outputs[] = {s.inverse, s.original_routes, s.grouped_weights, s.groups, s.group_count, s.fault};
    const uint64_t ob[] = {cap * 4, cap * 4, cap * 4, cap * sizeof(Group), 4, 4};
    for (int i = 0; i < 6; ++i) {
        if (!span_ok(outputs[i], ob[i], 4)) return Status::pointer;
        for (int j = 0; j < i; ++j)
            if (overlaps(outputs[i], ob[i], outputs[j], ob[j])) return Status::overlap;
        for (int j = 0; j < 3; ++j)
            if (ib[j] && overlaps(outputs[i], ob[i], inputs[j], ib[j])) return Status::overlap;
    }
    return Status::ok;
}
M26B1_HD static Status plan_values(const PlanInput& p, const PlanStorage& s) {
    uint32_t counts[256] = {}, bases[256], next[256];
    const uint32_t routes = p.rows * 8;
    for (uint32_t e = 0; e < 256; ++e)
        if (p.resident[e] > 1) return Status::metadata;
    // Complete validation precedes writes to plan buffers.
    for (uint32_t i = 0; i < routes; ++i) {
        const int32_t e = p.expert_ids[i];
        if (e < 0 || e >= 256 || !p.resident[e]) return Status::metadata;
        const float w = p.weights[i];
        if (!(w >= 0.0f && w <= 0x1.fffffep127f)) return Status::nonfinite;
        for (uint32_t j = (i / 8) * 8; j < i; ++j)
            if (p.expert_ids[j] == e) return Status::metadata;
        ++counts[e];
    }
    uint32_t total = 0;
    for (uint32_t e = 0; e < 256; ++e) {
        bases[e] = next[e] = total;
        total += counts[e];
    }
    for (uint32_t i = 0; i < s.route_capacity; ++i) {
        s.inverse[i] = s.original_routes[i] = -1;
        s.grouped_weights[i] = 0.0f;
        s.groups[i].expert = -1;
        s.groups[i].rows = 0;
        s.groups[i].route_base = -1;
        for (int j = 0; j < 16; ++j) s.groups[i].input_rows[j] = -1;
    }
    for (uint32_t i = 0; i < routes; ++i) {
        const auto grouped = next[p.expert_ids[i]]++;
        s.inverse[i] = static_cast<int32_t>(grouped);
        s.original_routes[grouped] = static_cast<int32_t>(i);
        s.grouped_weights[grouped] = p.weights[i];
    }
    uint32_t group = 0;
    for (uint32_t e = 0; e < 256; ++e) {
        for (uint32_t offset = 0; offset < counts[e]; offset += 16) {
            Group& g = s.groups[group++];
            g.expert = static_cast<int32_t>(e);
            g.route_base = static_cast<int32_t>(bases[e] + offset);
            g.rows = static_cast<int32_t>(counts[e] - offset < 16 ? counts[e] - offset : 16);
            for (int32_t j = 0; j < g.rows; ++j)
                g.input_rows[j] = s.original_routes[g.route_base + j] / 8;
        }
    }
    *s.group_count = group;
    return Status::ok;
}
Status plan_host(const PlanInput& p, const PlanStorage& s) {
    auto status = validate_plan(p, s);
    if (status != Status::ok) return status;
    status = plan_values(p, s);
    *s.fault = static_cast<uint32_t>(status);
    return status;
}
#ifdef __CUDACC__
__global__ void plan_kernel(PlanInput p, PlanStorage s) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *s.fault = static_cast<uint32_t>(plan_values(p, s));
}
cudaError_t plan_async(const PlanInput& p, const PlanStorage& s, cudaStream_t stream) {
    if (validate_plan(p, s) != Status::ok) return cudaErrorInvalidValue;
    plan_kernel<<<1, 1, 0, stream>>>(p, s);
    return cudaGetLastError();
}
#endif
} // namespace m26b1
