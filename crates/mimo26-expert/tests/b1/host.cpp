// Independent B1 host fixtures. Numerical mode: E-W4A8-v1.
// This file does not import the upstream oracle, quantizer or held compute.
#include "prepared.cuh"
#include "staging.cuh"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace m26b1;
static void require(bool v, const char* what) { if (!v) throw std::runtime_error(what); }
struct Buffer {
    std::vector<uint8_t> allocation;
    uint8_t* p;
    uint64_t n;
    explicit Buffer(uint64_t bytes) : allocation(bytes + 96, 0xa5), n(bytes) {
        auto address = reinterpret_cast<uintptr_t>(allocation.data()) + 32;
        p = reinterpret_cast<uint8_t*>((address + 15) & ~uintptr_t(15));
    }
    bool guards() const {
        return std::all_of(allocation.data(), static_cast<const uint8_t*>(p), [](auto x){return x == 0xa5;}) &&
          std::all_of(static_cast<const uint8_t*>(p + n), allocation.data() + allocation.size(), [](auto x){return x == 0xa5;});
    }
};
static CanonicalInfo canonical(uint32_t rank = 0) { return {2, hidden, local_i, rank, image_bytes}; }
static PreparedInfo prepared(uint32_t rank = 0) { return {prepared_tag, 2, hidden, local_i, rank, image_bytes}; }
// Independent forward-coordinate formulation: canonical (projection,row,K byte)
// -> packed byte. Production instead walks destination bytes and inverts bits.
static uint64_t packed_weight(int proj, uint64_t row, uint64_t byte) {
    const uint64_t k = proj == 2 ? 512 : 4096;
    if (proj == 0) row += 512; // Gate occupies the SECOND half.
    const uint64_t word = byte / 4, n = row % 256;
    const uint64_t tile = (row / 256) * (k / 128) + word / 16;
    const uint64_t register_lane = (n % 8) * 4 + word % 4;
    const uint64_t local = ((word % 16) / 4) * 1024 + (n / 32) * 128 + register_lane * 4 + (n % 32) / 8;
    return (proj == 2 ? 2228224 : 0) + (tile * 4096 + local) * 4 + byte % 4;
}
static uint64_t packed_scale(int proj, uint64_t row, uint64_t block) {
    const uint64_t k = proj == 2 ? 512 : 4096;
    if (proj == 0) row += 512;
    const uint64_t tile = (row / 256) * (k / 128) + block / 4;
    return (proj == 2 ? 3276800 : 2097152) + tile * 1024 + (row % 256) * 4 + block % 4;
}
static std::vector<uint8_t> independent_restore(const uint8_t* bytes) {
    std::vector<uint8_t> out(image_bytes), visited(image_bytes, 0);
    const uint64_t wp[] = {0, 1114112, 2228224}, sp[] = {1048576, 2162688, 3276800};
    for (int proj = 0; proj < 3; ++proj) {
        const uint64_t rows = proj == 2 ? 4096 : 512, k = proj == 2 ? 512 : 4096;
        for (uint64_t row = 0; row < rows; ++row) {
            for (uint64_t b = 0; b < k / 2; ++b) {
                const auto index = packed_weight(proj, row, b);
                require(index < image_bytes && !visited[index]++, "payload mapping is not bijective");
                out[wp[proj] + row * k / 2 + b] = bytes[index];
            }
            for (uint64_t b = 0; b < k / 32; ++b) {
                const auto index = packed_scale(proj, row, b);
                require(index < image_bytes && !visited[index]++, "scale mapping is not bijective");
                out[sp[proj] + row * k / 32 + b] = bytes[index];
            }
        }
    }
    require(std::all_of(visited.begin(), visited.end(), [](auto n){return n == 1;}), "prepared holes");
    return out;
}
static void fill(Buffer& image) {
    uint32_t state = 0x94fe18u;
    for (uint64_t i = 0; i < image.n; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        image.p[i] = static_cast<uint8_t>(state);
    }
}
static void prepare_checks(Buffer& source, uint32_t rank) {
    Buffer dst(image_bytes);
    const std::vector<uint8_t> before(source.p, source.p + source.n);
    PreparedInfo info{};
    require(prepare_host(canonical(rank), source.p, image_bytes, dst.p, image_bytes, &info) == Status::ok, "prepare failed");
    require(check_prepared(info) == Status::ok && info.rank == rank, "prepared tag/rank missing");
    require(independent_restore(dst.p) == before, "independent real byte roundtrip mismatch");
    require(std::equal(before.begin(), before.end(), source.p), "canonical input mutated");
    require(dst.guards() && source.guards(), "prepare poison overwritten");
    dst.p[0] ^= 1;
    require(independent_restore(dst.p) != before, "corrupted prepared payload escaped comparator");
    dst.p[0] ^= 1;
    dst.p[2097152] ^= 1;
    require(independent_restore(dst.p) != before, "corrupted prepared scale escaped comparator");
    dst.p[2097152] ^= 1;
    std::swap_ranges(dst.p, dst.p + 1048576, dst.p + 1048576);
    require(independent_restore(dst.p) != before, "gate/up swap escaped comparator");
    std::fill(dst.p, dst.p + image_bytes, 0xa5);
    auto bad = canonical(rank); bad.version = 1;
    require(prepare_host(bad, nullptr, image_bytes, dst.p, image_bytes, &info) == Status::version, "v1 same-size accepted");
    require(std::all_of(dst.p, dst.p + image_bytes, [](auto b){return b == 0xa5;}), "invalid version wrote output");
    bad = canonical(rank); bad.intermediate_size = 2048;
    require(prepare_host(bad, source.p, image_bytes, dst.p, image_bytes, &info) == Status::geometry, "wrong local shape accepted");
    require(prepare_host(canonical(rank), source.p, image_bytes - 1, dst.p, image_bytes, &info) == Status::size, "short source accepted");
    require(prepare_host(canonical(rank), source.p, image_bytes, source.p, image_bytes, &info) == Status::overlap, "in-place transform accepted");
    require(prepare_host(canonical(rank), source.p, image_bytes, dst.p + 1, image_bytes, &info) == Status::pointer, "misaligned destination accepted");
    bad = canonical(rank); bad.rank = 4;
    require(prepare_host(bad, source.p, image_bytes, dst.p, image_bytes, &info) == Status::geometry, "bad canonical rank accepted");
    bad = canonical(rank); bad.hidden_size = 2048;
    require(prepare_host(bad, source.p, image_bytes, dst.p, image_bytes, &info) == Status::geometry, "bad hidden shape accepted");
    bad = canonical(rank); --bad.bytes;
    require(prepare_host(bad, source.p, image_bytes, dst.p, image_bytes, &info) == Status::size, "bad metadata byte count accepted");
    require(prepare_host(canonical(rank), source.p, image_bytes, dst.p, image_bytes, nullptr) == Status::pointer, "null metadata output accepted");
    require(prepare_host(canonical(rank), source.p, image_bytes, dst.p, image_bytes, reinterpret_cast<PreparedInfo*>(dst.p)) == Status::overlap, "metadata-output alias accepted");
    require(std::all_of(dst.p, dst.p + image_bytes, [](auto b){return b == 0xa5;}) && dst.guards() && source.guards(), "rejected prepare changed output poison");
}
static void roundtrip(const char* filename, uint32_t rank) {
    Buffer source(image_bytes);
    std::ifstream file(filename, std::ios::binary);
    require(bool(file), "real rank image missing");
    file.read(reinterpret_cast<char*>(source.p), image_bytes);
    require(file.gcount() == static_cast<std::streamsize>(image_bytes) && file.peek() == EOF, "real rank image size");
    prepare_checks(source, rank);
    std::cout << "b1_prepare_roundtrip PASS " << filename << " rank " << rank << " bytes " << image_bytes << '\n';
}
static void staging() {
    Buffer source(image_bytes), resident(image_bytes);
    fill(source);
    PreparedInfo info{};
    require(prepare_host(canonical(), source.p, image_bytes, resident.p, image_bytes, &info) == Status::ok, "stage fixture preparation");
    const Pool pool{info, image_bytes, 1};
    uint64_t cases = 0;
    for (uint32_t width : {64u, 128u, 192u}) {
        for (bool down : {false, true}) for (bool gate : {false, true}) {
            if (down && gate) continue;
            const uint32_t nmax = down ? 4096 - 128 : 512 - width;
            const uint32_t kmax = down ? 512 - width : 4096 - 128;
            for (uint32_t n = 0; n <= nmax; n += 32) for (uint32_t k = 0; k <= kmax; k += down ? 32 : 128) {
                const Tile t{0, n, k, width, down, gate, true};
                const uint32_t pb = width * 64, sb = staged_scale_bytes(t);
                Buffer payload(pb), scales(sb);
                require(stage_host(pool, t, resident.p, payload.p, pb, scales.p, sb) == Status::ok, "valid stage rejected");
                // Decode staged byte coordinates independently of source offset helpers.
                for (uint32_t b = 0; b < pb; ++b) {
                    const uint32_t word = b / 4;
                    const uint32_t warp = word % 4, lane = (word / 4) % 32;
                    const uint32_t chunks = down ? 4 : width / 32;
                    const uint32_t chunk = (word / 128) % chunks, kb = word / (128 * chunks);
                    const uint32_t row = n + chunk * 32 + warp * 8 + lane / 4;
                    const uint32_t col_byte = k / 2 + kb * 16 + (lane % 4) * 4 + b % 4;
                    const int proj = down ? 2 : gate ? 0 : 1;
                    const uint64_t base = proj == 0 ? 0 : proj == 1 ? 1114112 : 2228224;
                    require(payload.p[b] == source.p[base + uint64_t(row) * (down ? 256 : 2048) + col_byte], "staged payload differs from canonical rectangle");
                }
                for (uint32_t b = 0; b < sb; ++b) {
                    const uint32_t row = down ? n + (b / 4) % 128 : n + b / 4;
                    const uint32_t block = down ? ((b / 4) / 128) * 4 + b % 4 : b % 4;
                    const uint64_t base = down ? 3276800 : gate ? 1048576 : 2162688;
                    const uint8_t expected = down && block >= width / 32 ? 0 : source.p[base + uint64_t(row) * (down ? 16 : 128) + k / 32 + block];
                    require(scales.p[b] == expected, "staged scale mismatch or tail load");
                }
                require(payload.guards() && scales.guards(), "stage poison overwritten");
                ++cases;
            }
        }
    }
    Tile bad{0, 384, 0, 192, false, false, true};
    Buffer poison(12288), sf(1024);
    require(stage_host(pool, bad, nullptr, poison.p, poison.n, sf.p, sf.n) == Status::bounds, "width192 I512 tail accepted");
    bad = {0, 0, 384, 192, true, false, true};
    require(check_tile(pool, bad) == Status::bounds, "FC2 K tail accepted");
    bad = {0, 0, 0, 191, false, false, true};
    require(check_tile(pool, bad) == Status::geometry, "ragged width accepted");
    bad.width = 64; bad.slot = 1;
    require(check_tile(pool, bad) == Status::bounds, "nonresident slot accepted");
    bad.active = false; bad.slot = UINT32_MAX; bad.width = UINT32_MAX;
    require(stage_host({}, bad, nullptr, nullptr, 0, nullptr, 0) == Status::ok, "inactive poison accessed");
    require(std::all_of(poison.p, poison.p + poison.n, [](auto b){return b == 0xa5;}), "invalid stage changed poison");
    Pool wrong = pool; wrong.info.tag = 2;
    require(check_tile(wrong, {0, 0, 0, 64, false, false, true}) == Status::version, "canonical bytes relabeled as prepared");
    const Tile valid{0, 0, 0, 64, false, false, true};
    wrong = pool; wrong.info.source_version = 1;
    require(check_tile(wrong, valid) == Status::version, "stale prepared source version accepted");
    wrong = pool; wrong.info.rank = 4;
    require(check_tile(wrong, valid) == Status::geometry, "stale prepared rank accepted");
    wrong = pool; --wrong.bytes;
    require(check_tile(wrong, valid) == Status::bounds, "short resident pool accepted");
    bad = valid; bad.n_start = 1;
    require(check_tile(pool, bad) == Status::geometry, "unaligned N start accepted");
    bad = valid; bad.k_start = 32;
    require(check_tile(pool, bad) == Status::geometry, "unaligned FC1 K128 start accepted");
    require(stage_host(pool, valid, resident.p, resident.p, 4096, sf.p, 256) == Status::overlap, "staging source-output alias accepted");
    require(stage_host(pool, valid, resident.p, poison.p, 4096, poison.p, 256) == Status::overlap, "staging output-output alias accepted");
    require(stage_host(pool, valid, resident.p + 1, poison.p, 4096, sf.p, 256) == Status::pointer, "misaligned staging source accepted");
    require(std::all_of(poison.p, poison.p + poison.n, [](auto b){return b == 0xa5;}), "staging rejection wrote output");
    Pool huge{prepared(), 47ull * 256 * image_bytes, 47 * 256};
    Tile last{47 * 256 - 1, 3968, 384, 128, true, false, true};
    require(check_tile(huge, last) == Status::ok, "last resident slot rejected");
    const auto offset = payload_source(last, payload_transfers(last) - 1);
    require(offset > UINT32_MAX && offset + 16 <= huge.bytes, "64-bit payload offset wrapped");
    uint64_t scale = 0;
    require(scale_source(last, staged_scale_bytes(last) - 1, scale) && scale > UINT32_MAX && scale < huge.bytes, "64-bit scale offset wrapped");
    require(source.guards() && resident.guards(), "resident poison overwritten");
    std::cout << "b1_stage_bounds PASS " << cases << " rectangles; inactive, tail, version, sparse slot and >4 GiB negatives\n";
}
int main(int argc, char** argv) {
    try {
        require(argc >= 2, "missing case");
        if (std::string(argv[1]) == "prepare") {
            require(argc == 4, "prepare image rank");
            roundtrip(argv[2], static_cast<uint32_t>(std::stoul(argv[3])));
        } else if (std::string(argv[1]) == "staging") staging();
        else if (std::string(argv[1]) == "synthetic") { Buffer b(image_bytes); fill(b); prepare_checks(b, 3); }
        else throw std::runtime_error("unknown B1 case");
        return 0;
    } catch (const std::exception& e) { std::cerr << "B1 FAIL: " << e.what() << '\n'; return 3; }
}
