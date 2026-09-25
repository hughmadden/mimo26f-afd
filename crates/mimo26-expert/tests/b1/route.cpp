// Synthetic once-weighted route/wire fixtures. E-W4A8-v1.
#include "route_reduce.cuh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace m26b1;
static void check(bool x, const char* text) { if (!x) throw std::runtime_error(text); }
template<class T> struct Guard {
    std::vector<T> v;
    explicit Guard(size_t n) : v(n + 2) { std::memset(v.data(), 0xa5, v.size() * sizeof(T)); }
    T* data() { return v.data() + 1; }
    bool intact() const {
        const auto* b = reinterpret_cast<const uint8_t*>(v.data());
        for (size_t i = 0; i < sizeof(T); ++i)
            if (b[i] != 0xa5 || b[(v.size() - 1) * sizeof(T) + i] != 0xa5) return false;
        return true;
    }
};
static void write_word(std::ofstream& f, uint32_t word, int bytes) {
    for (int i = 0; i < bytes; ++i) f.put(static_cast<char>(word >> (i * 8)));
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "route fixture output directory required");
        check(std::fegetround() == FE_TONEAREST, "CPU rounding mode is not RNE");
        constexpr uint32_t rows = 3;
        const uint64_t n = uint64_t(rows) * hidden;
        Guard<float> z(n * 8), weights(rows * 8), partial(n);
        Guard<uint16_t> codes(n);
        Guard<uint32_t> fault(1);
        RouteInput x{rows, z.data(), weights.data(), n * 8, rows * 8};
        RankOutput y{partial.data(), codes.data(), n, n, fault.data()};
        uint64_t weight_twice_mismatches = 0;
        for (uint32_t rank = 0; rank < 4; ++rank) {
            for (uint32_t t = 0; t < rows; ++t) for (uint32_t j = 0; j < 8; ++j) {
                weights.data()[t * 8 + j] = j == 0 ? 0.0f : float(j + 1) / 32;
                for (uint32_t h = 0; h < hidden; ++h)
                    z.data()[(uint64_t(t) * 8 + j) * hidden + h] =
                        float(int(h % 17) - 8) * float(j + 1) / 32 + float(t) / 4 + float(rank) / 8;
            }
            check(reduce_host(x, y) == Status::ok && !*y.fault, "route positive failed");
            std::ofstream fp(std::string(argv[1]) + "/rank" + std::to_string(rank) + ".f32", std::ios::binary);
            std::ofstream bf(std::string(argv[1]) + "/rank" + std::to_string(rank) + ".bf16", std::ios::binary);
            check(bool(fp) && bool(bf), "route output open failed");
            for (uint64_t i = 0; i < n; ++i) {
                const uint64_t t = i / hidden, h = i % hidden;
                double expected = 0, twice = 0;
                for (int j = 0; j < 8; ++j) {
                    const double w = weights.data()[t * 8 + j];
                    const double value = z.data()[(t * 8 + j) * hidden + h];
                    expected += w * value;
                    twice += w * w * value;
                }
                check(partial.data()[i] == float(expected), "FP32 once-weighted independent oracle mismatch");
                if (partial.data()[i] != float(twice)) ++weight_twice_mismatches;
                uint32_t bits; std::memcpy(&bits, partial.data() + i, 4);
                write_word(fp, bits, 4); write_word(bf, codes.data()[i], 2);
            }
            check(z.intact() && weights.intact() && partial.intact() && codes.intact() && fault.intact(), "route poison changed");
        }
        check(weight_twice_mismatches > 0, "weight-once negative escaped");
        // FP32 association must be route-slot order, not a reassociated sum.
        std::fill(z.data(), z.data() + n * 8, 0.0f);
        std::fill(weights.data(), weights.data() + rows * 8, 1.0f);
        z.data()[0] = 1e8f; z.data()[hidden] = 1.0f;
        z.data()[2 * hidden] = -1e8f; z.data()[3 * hidden] = 3.0f;
        check(reduce_host(x, y) == Status::ok && partial.data()[0] == 3.0f, "route addition order changed");
        // Distinguish separate FP32 multiply/add from a fused weighted sum.
        // (1 - 2^-23) * (1 + 2^-23) rounds to 1 before adding -1.
        std::fill(z.data(), z.data() + n * 8, 0.0f);
        z.data()[0] = -1.0f;
        weights.data()[1] = 1.0f - 0x1p-23f;
        z.data()[hidden] = 1.0f + 0x1p-23f;
        check(reduce_host(x, y) == Status::ok && partial.data()[0] == 0.0f, "weighted sum was fused");
        check(std::fma(weights.data()[1], z.data()[hidden], -1.0f) == -0x1p-46f,
              "FMA negative failed to distinguish the arithmetic");
        std::fill(weights.data(), weights.data() + rows * 8, 1.0f);
        // A nonzero FP32 product below the normal range must not be flushed.
        std::fill(z.data(), z.data() + n * 8, 0.0f);
        z.data()[0] = 0x1p-126f; weights.data()[0] = 0.5f;
        check(reduce_host(x, y) == Status::ok && partial.data()[0] == 0x1p-127f && codes.data()[0] == 0x0040,
              "subnormal product flushed");
        std::fill(weights.data(), weights.data() + rows * 8, 1.0f);
        // Exact BF16 midpoint checks: low code even rounds down, odd rounds up.
        std::fill(z.data(), z.data() + n * 8, 0.0f);
        z.data()[0] = 1.0f + 1.0f / 256;
        z.data()[1] = 1.0f + 3.0f / 256;
        z.data()[2] = -(1.0f + 1.0f / 256);
        check(reduce_host(x, y) == Status::ok, "BF16 midpoint fixture");
        check(codes.data()[0] == 0x3f80 && codes.data()[1] == 0x3f82 && codes.data()[2] == 0xbf80, "BF16 RNE ties failed");
        const auto structural = [&](const RouteInput& a, const RankOutput& b, Status want) {
            const auto f = partial.v; const auto c = codes.v; const auto fault_before = fault.v;
            check(reduce_host(a, b) == want, "structural route negative accepted");
            check(std::memcmp(f.data(), partial.v.data(), f.size() * sizeof(float)) == 0 &&
                  std::memcmp(c.data(), codes.v.data(), c.size() * sizeof(uint16_t)) == 0 &&
                  std::memcmp(fault_before.data(), fault.v.data(), fault_before.size() * sizeof(uint32_t)) == 0,
                  "structural rejection wrote output or fault");
        };
        auto bad = x; --bad.values; structural(bad, y, Status::size);
        bad = x; --bad.weight_elements; structural(bad, y, Status::size);
        bad = x; bad.rows = 4097; structural(bad, y, Status::geometry);
        bad = x; bad.unweighted = nullptr; structural(bad, y, Status::pointer);
        auto alias = y; alias.partial = z.data(); structural(x, alias, Status::overlap);
        alias = y; --alias.partial_elements; structural(x, alias, Status::size);
        alias = y; --alias.bf16_elements; structural(x, alias, Status::size);
        alias = y; alias.fault = nullptr; structural(x, alias, Status::pointer);
        alias = y; alias.partial = nullptr; structural(x, alias, Status::pointer);
        alias = y; alias.bf16 = reinterpret_cast<uint16_t*>(partial.data()); structural(x, alias, Status::overlap);
        alias = y; alias.fault = reinterpret_cast<uint32_t*>(partial.data()); structural(x, alias, Status::overlap);
        alias = y; alias.partial = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(partial.data()) + 1);
        structural(x, alias, Status::pointer);
        alias = y; alias.fault = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(fault.data()) + 1);
        structural(x, alias, Status::pointer);
        for (uint32_t bits : {0x7fc00000u, 0x7f800000u, 0xff800000u, 0xbf800000u}) {
            std::memcpy(weights.data(), &bits, 4);
            check(reduce_host(x, y) == Status::nonfinite && *y.fault, "invalid route weight accepted");
        }
        weights.data()[0] = 1.0f;
        uint32_t nan = 0x7fc00000; std::memcpy(z.data(), &nan, 4);
        check(reduce_host(x, y) == Status::nonfinite && *y.fault, "NaN route accepted");
        std::fill(z.data(), z.data() + n * 8, 0.0f);
        uint32_t max = 0x7f7fffff; std::memcpy(z.data(), &max, 4);
        check(reduce_host(x, y) == Status::nonfinite, "BF16 overflow accepted");
        weights.data()[0] = 2.0f;
        check(reduce_host(x, y) == Status::nonfinite, "FP32 product overflow accepted");
        weights.data()[0] = 1.0f; z.data()[hidden] = z.data()[0];
        check(reduce_host(x, y) == Status::nonfinite, "FP32 running-sum overflow accepted");
        check(z.intact() && weights.intact() && partial.intact() && codes.intact() && fault.intact(), "negative route poison changed");
        RouteInput empty{0, nullptr, nullptr, 0, 0};
        RankOutput empty_out{nullptr, nullptr, 0, 0, fault.data()};
        check(reduce_host(empty, empty_out) == Status::ok && !*fault.data(), "empty route reset failed");
        std::cout << "b1_route_weight_once PASS " << 4 * n << " FP32 outputs, " << weight_twice_mismatches
                  << " double-weight mismatches; RNE/order/poison/size/nonfinite/empty negatives\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "B1 ROUTE FAIL: " << error.what() << '\n'; return 3; }
}
