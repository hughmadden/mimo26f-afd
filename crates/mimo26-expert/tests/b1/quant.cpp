// Quantizer host comparison adapter and deliberate harness-only mutants.
// The production quantizer was frozen before this reference adapter was written.
#include "quant_v1.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace m26b1;
static void check(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
static void write_bytes(const std::string& path, const void* data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    check(bool(out), "open quantizer output");
    out.write(static_cast<const char*>(data), bytes);
    check(bool(out), "write quantizer output");
}
static double decoded_magnitude(uint8_t code) {
    const unsigned q = code & 127, e = q >> 3, m = q & 7;
    return e ? std::ldexp(double(8 + m), int(e) - 10) : std::ldexp(double(m), -9);
}
int main(int argc, char** argv) {
    try {
        check(argc == 4, "quant input-f32 output-prefix mode");
        const std::string mode = argv[3];
        check(mode == "correct" || mode == "trunc" || mode == "nearest_scale" ||
              mode == "no_floor" || mode == "bf16_preround" || mode == "e4m3_subnormal_flush", "unknown quantizer mode");
        const uint32_t endian = 1;
        check(*reinterpret_cast<const uint8_t*>(&endian) == 1 && sizeof(float) == 4, "test adapter requires little-endian FP32");
        check(quant_float_bits(1e-4f) == quant_floor_bits, "floor is not FP32(1e-4)");
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        check(bool(input), "open FP32 corpus");
        const auto bytes = input.tellg();
        check(bytes > 0 && bytes % 128 == 0, "incomplete K32 input corpus");
        const uint64_t elements = uint64_t(bytes) / 4, blocks = elements / 32;
        std::vector<float> source(elements);
        input.seekg(0); input.read(reinterpret_cast<char*>(source.data()), bytes);
        check(bool(input), "read FP32 corpus");
        if (mode == "bf16_preround") for (auto& v : source) {
            uint32_t bits = quant_float_bits(v);
            if ((bits & 0x7fffffffu) < 0x7f800000u) {
                bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
                std::memcpy(&v, &bits, 4);
            }
        }
        const auto source_before = source;
        std::vector<uint8_t> payload(elements + 64, 0xa5), scales(blocks + 64, 0xa5);
        std::vector<uint32_t> faults(blocks + 2, 0xa5a5a5a5u);
        QuantInput x{source.data(), elements};
        QuantOutput y{payload.data() + 32, scales.data() + 32, faults.data() + 1, elements, blocks, blocks};
        const auto status = quantize_host(x, y);
        uint64_t failed = 0;
        for (uint64_t b = 0; b < blocks; ++b) {
            if (y.block_faults[b]) {
                ++failed;
                check(y.scales[b] == 0xa5, "faulting block wrote scale");
                for (uint32_t j = 0; j < 32; ++j) check(y.payload[b * 32 + j] == 0xa5, "faulting block wrote payload");
                continue;
            }
            check(y.scales[b] >= 105 && y.scales[b] <= 247, "encoder scale range");
            for (uint32_t j = 0; j < 32; ++j) check((y.payload[b * 32 + j] & 127) != 127, "encoder emitted NaN");
            if (mode == "nearest_scale" || mode == "no_floor") {
                double amax = mode == "no_floor" ? 0.0 : double(1e-4f);
                for (uint32_t j = 0; j < 32; ++j) amax = std::max(amax, std::abs(double(source[b * 32 + j])));
                int k;
                if (mode == "nearest_scale") k = int(std::nearbyint(std::log2(amax / 448.0)));
                else if (!amax) k = -127;
                else { int e; const double m = std::frexp(amax, &e); k = e - (m <= .875 ? 9 : 8); }
                y.scales[b] = static_cast<uint8_t>(k + 127);
                for (uint32_t j = 0; j < 32; ++j)
                    y.payload[b * 32 + j] = quant_payload(quant_float_bits(source[b * 32 + j]), k);
            }
            for (uint32_t j = 0; j < 32; ++j) {
                auto& q = y.payload[b * 32 + j];
                if (mode == "e4m3_subnormal_flush" && (q & 127) < 8) q &= 128;
                if (mode == "trunc" && decoded_magnitude(q) >
                    std::ldexp(std::abs(double(source[b * 32 + j])), 127 - int(y.scales[b])))
                    q = uint8_t((q & 128) | ((q & 127) - 1));
            }
        }
        check(status == (failed ? Status::nonfinite : Status::ok), "aggregate host status mismatch");
        check(std::memcmp(source_before.data(), source.data(), elements * 4) == 0, "quantizer mutated input");
        for (uint32_t i = 0; i < 32; ++i)
            check(payload[i] == 0xa5 && payload[elements + 32 + i] == 0xa5 &&
                  scales[i] == 0xa5 && scales[blocks + 32 + i] == 0xa5, "quantizer poison overwritten");
        check(faults.front() == 0xa5a5a5a5u && faults.back() == 0xa5a5a5a5u, "fault redzone overwritten");
        const auto p_before = payload, s_before = scales;
        const auto f_before = faults;
        auto bad = x; bad.elements = 31;
        check(quantize_host(bad, y) == Status::geometry, "K16/ragged group accepted");
        bad = x; bad.elements = UINT64_MAX - 31;
        check(quantize_host(bad, y) == Status::geometry, "overflowing source extent accepted");
        bad = x; bad.values = nullptr;
        check(quantize_host(bad, y) == Status::pointer, "null input accepted");
        auto out = y; --out.payload_bytes;
        check(quantize_host(x, out) == Status::size, "short payload accepted");
        out = y; out.scales = out.payload;
        check(quantize_host(x, out) == Status::overlap, "quantizer outputs overlap");
        out = y; out.payload = reinterpret_cast<uint8_t*>(source.data());
        check(quantize_host(x, out) == Status::overlap, "quantizer input-output alias accepted");
        out = y; out.block_faults = nullptr;
        check(quantize_host(x, out) == Status::pointer, "null fault buffer accepted");
        check(payload == p_before && scales == s_before && faults == f_before, "structural rejection wrote output");
        check(quantize_host({nullptr, 0}, {nullptr, nullptr, nullptr, 0, 0, 0}) == Status::ok, "empty quantizer rejected");
        write_bytes(std::string(argv[2]) + ".payload", y.payload, elements);
        write_bytes(std::string(argv[2]) + ".scales", y.scales, blocks);
        write_bytes(std::string(argv[2]) + ".faults", y.block_faults, blocks * 4);
        std::cout << "quant_v1 CPU " << mode << ": " << blocks << " blocks, " << failed << " numerical faults; guards/metadata PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "QUANT FAIL: " << error.what() << '\n'; return 3; }
}
