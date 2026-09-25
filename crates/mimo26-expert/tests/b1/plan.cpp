// Host replay proof only, not CUDA graph qualification. E-W4A8-v1.
#include "group_plan.cuh"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace m26b1;
static void check(bool x, const char* text) { if (!x) throw std::runtime_error(text); }
template<class T> struct Guard {
    std::vector<T> v;
    explicit Guard(size_t n) : v(n + 2) { std::memset(v.data(), 0xa5, v.size() * sizeof(T)); }
    T* data() { return v.data() + 1; }
    bool intact() const {
        const auto* bytes = reinterpret_cast<const uint8_t*>(v.data());
        for (size_t i = 0; i < sizeof(T); ++i)
            if (bytes[i] != 0xa5 || bytes[(v.size() - 1) * sizeof(T) + i] != 0xa5) return false;
        return true;
    }
};
struct Storage {
    Guard<int32_t> inverse, original;
    Guard<float> weights;
    Guard<Group> groups;
    Guard<uint32_t> count, fault;
    uint32_t cap;
    explicit Storage(uint32_t c) : inverse(c * 8), original(c * 8), weights(c * 8), groups(c * 8), count(1), fault(1), cap(c) {}
    PlanStorage view() { return {inverse.data(), original.data(), weights.data(), groups.data(), count.data(), fault.data(), cap * 8ull, cap * 8ull}; }
    bool guards() const { return inverse.intact() && original.intact() && weights.intact() && groups.intact() && count.intact() && fault.intact(); }
};
static bool equivalent(const PlanInput& p, const PlanStorage& s) {
    std::vector<int32_t> routes(p.rows * 8);
    std::iota(routes.begin(), routes.end(), 0);
    std::stable_sort(routes.begin(), routes.end(), [&](int a, int b){ return p.expert_ids[a] < p.expert_ids[b]; });
    uint32_t groups = 0;
    for (uint32_t begin = 0; begin < routes.size();) {
        uint32_t end = begin;
        while (end < routes.size() && p.expert_ids[routes[end]] == p.expert_ids[routes[begin]] && end - begin < 16) ++end;
        const auto& g = s.groups[groups++];
        if (g.expert != p.expert_ids[routes[begin]] || g.rows != int(end - begin) || g.route_base != int(begin)) return false;
        for (uint32_t i = 0; i < 16; ++i)
            if (g.input_rows[i] != (begin + i < end ? routes[begin + i] / 8 : -1)) return false;
        begin = end;
    }
    if (*s.fault || *s.group_count != groups) return false;
    for (uint32_t i = 0; i < routes.size(); ++i)
        if (s.original_routes[i] != routes[i] || s.inverse[routes[i]] != int(i) || s.grouped_weights[i] != p.weights[routes[i]]) return false;
    for (uint32_t i = routes.size(); i < s.route_capacity; ++i)
        if (s.original_routes[i] != -1 || s.inverse[i] != -1 || s.grouped_weights[i] != 0) return false;
    for (uint32_t i = groups; i < s.group_capacity; ++i) {
        if (s.groups[i].rows || s.groups[i].expert != -1 || s.groups[i].route_base != -1) return false;
        for (int j = 0; j < 16; ++j) if (s.groups[i].input_rows[j] != -1) return false;
    }
    return true;
}
int main() {
    try {
        uint32_t cases = 0;
        for (uint32_t cap : {256u, 2048u, 4096u}) {
            Storage store(cap);
            std::vector<int32_t> ids(cap * 8);
            std::vector<float> weights(cap * 8);
            std::array<uint8_t, 256> resident{};
            auto s = store.view();
            // Reuse the same pointers from largest live work back to empty/small.
            for (bool sparse : {false, true}) for (uint32_t rows : {cap, 64u, 33u, 31u, 17u, 16u, 8u, 4u, 2u, 1u, 0u, 1u}) {
                resident.fill(sparse ? 0 : 1);
                const int32_t chosen[] = {255, 0, 7, 63, 128, 32, 254, 1};
                for (auto e : chosen) resident[e] = 1;
                for (uint32_t i = 0; i < rows * 8; ++i) {
                    ids[i] = sparse ? chosen[(i % 8 + i / 8) % 8] : int((i * 17) % 256);
                    weights[i] = float((i % 13) + 1) / 16;
                }
                PlanInput p{cap, rows, ids.data(), weights.data(), uint64_t(rows) * 8, resident.data(), 256};
                check(plan_host(p, s) == Status::ok, "plan positive failed");
                check(equivalent(p, s), "stable-sort independent oracle mismatch");
                check(store.guards(), "plan redzone changed");
                if (rows && sparse) {
                    const auto old = s.groups[0].expert;
                    s.groups[0].expert = 2; // resident ordinal, not real expert ID.
                    check(!equivalent(p, s), "ordinal-for-ID negative escaped");
                    s.groups[0].expert = old;
                }
                ++cases;
            }
            resident.fill(1);
            for (int i = 0; i < 8; ++i) { ids[i] = i; weights[i] = .125f; }
            PlanInput p{cap, 1, ids.data(), weights.data(), 8, resident.data(), 256};
            check(plan_host(p, s) == Status::ok, "negative fixture positive failed");
            const auto original = store.groups.v;
            const auto inverse_before = store.inverse.v, routes_before = store.original.v;
            const auto weights_before = store.weights.v;
            const auto count_before = store.count.v;
            ids[7] = 256;
            check(plan_host(p, s) == Status::metadata && *s.fault != 0, "out-of-range ID accepted");
            check(std::memcmp(original.data(), store.groups.v.data(), original.size() * sizeof(Group)) == 0, "bad metadata changed groups");
            ids[7] = -1;
            check(plan_host(p, s) == Status::metadata, "negative ID accepted");
            ids[7] = 0;
            check(plan_host(p, s) == Status::metadata, "duplicate top8 ID accepted");
            ids[7] = 7; resident[7] = 0;
            check(plan_host(p, s) == Status::metadata, "nonresident sparse ID accepted");
            resident[7] = 1; uint32_t nan = 0x7fc00000; std::memcpy(&weights[0], &nan, 4);
            check(plan_host(p, s) == Status::nonfinite, "nonfinite weight accepted");
            weights[0] = -.125f;
            check(plan_host(p, s) == Status::nonfinite, "negative route weight accepted");
            weights[0] = .125f;
            p.route_elements = 7;
            check(plan_host(p, s) == Status::size, "short input accepted");
            p.route_elements = 8;
            auto bad = s; --bad.group_capacity;
            check(plan_host(p, bad) == Status::size, "short metadata output accepted");
            bad = s; bad.original_routes = bad.inverse;
            check(plan_host(p, bad) == Status::overlap, "aliased output accepted");
            resident[3] = 2;
            check(plan_host(p, s) == Status::metadata, "nonbinary resident mask accepted");
            resident[3] = 1;
            p.rows = cap + 1;
            check(plan_host(p, s) == Status::geometry, "live rows exceed capacity");
            p.rows = 1; p.route_elements = uint64_t(cap) * 8 + 1;
            check(plan_host(p, s) == Status::size, "oversized route extent accepted");
            p.route_elements = 8;
            auto missing = p; missing.expert_ids = nullptr;
            check(plan_host(missing, s) == Status::pointer, "null input IDs accepted");
            missing = p; missing.weights = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(weights.data()) + 1);
            check(plan_host(missing, s) == Status::pointer, "misaligned route weights accepted");
            bad = s; bad.inverse = ids.data();
            check(plan_host(p, bad) == Status::overlap, "input-output plan alias accepted");
            bad = s; bad.fault = nullptr;
            check(plan_host(p, bad) == Status::pointer, "null plan fault accepted");
            p.capacity = 257;
            check(plan_host(p, s) == Status::geometry, "unbaked capacity accepted");
            check(std::memcmp(original.data(), store.groups.v.data(), original.size() * sizeof(Group)) == 0 &&
                  inverse_before == store.inverse.v && routes_before == store.original.v &&
                  weights_before == store.weights.v && count_before == store.count.v,
                  "rejected plan changed a non-fault output");
            check(store.guards(), "negative plan redzone changed");
        }
        std::cout << "b1_plan_replay PASS " << cases << " stable CPU replays, 3 capacity classes, sparse/full IDs, M64 splitting and metadata negatives; NOT GPU graph proof\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "B1 PLAN FAIL: " << error.what() << '\n'; return 3; }
}
