// Perf reset R3b: B1 FC1 with explicit expert weight reuse (Spark serving path).
//
// connected_fc1<128, true> runs one CTA per (16-row group, 128-column slice) and
// streams the slice's up+gate weights for every group: at 2,048 rows that is
// ~1,120 groups x 4 slices x 544 KB = 2.4 GB per layer, DRAM-bound at ~12 ms on
// a GB10 (24 MB L2, too small to catch the re-reads of ~14 experts in flight).
//
// Here one CTA takes one 128-column slice for a CHUNK of up to four groups of
// the same expert (64 rows). Each staged K128 weight tile (up + gate + scales,
// 17 KB) is read once and applied to all four groups, and the next tile is
// prefetched (cp.async double buffer) while the current one is consumed.
//
// Eight warps: warp = (group m = warp / 2, column half nh = warp % 2). Every
// (row, column) accumulator sees exactly connected_fc1's sequence (the same
// MMA/fallback on the same operands, kt then kb order) and the same SiLU*up
// finish, so `mid` is bit-identical; the quantizer and FC2 are unchanged.
#pragma once

namespace m26b1 {

/// Up to four consecutive groups of one expert (planner output, expert-major).
struct Chunk {
    int32_t first_group, groups;
};

// dot32's conforming exceptional-scale arm (grouped.cu), out of line: inlined at
// all 64 call sites of an M64 warp's K step it made ptxas run for many minutes.
// Accumulators travel by value (registers). Same statements in the same order,
// so the values are dot32's. Real checkpoints never take it (UE8M0 in [2, 252]).
template <int ByteB>
__device__ __noinline__ float4 dot32_exceptional(float4 acc, ActivationView x, int k32, uint32_t packed,
                                                 uint32_t sb) {
    float d[4] = {acc.x, acc.y, acc.z, acc.w};
    const int c = threadIdx.x % 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int n = 2 * c + e % 2, row = e < 2 ? x.low : x.high;
        const auto sw = __shfl_sync(0xffffffffu, sb, n * 4);
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            // Every lane participates, including padded A rows.
            const auto w = __shfl_sync(0xffffffffu, packed, n * 4 + k / 8);
            const float av = activation_value(x, row, k32 * 32 + k);
            const float bv = m26x::decode((w >> (4 * (k % 8))) & 15, (sw >> (8 * ByteB)) & 255, 0);
            d[e] = fmaf(av, bv, d[e]);
        }
    }
    return make_float4(d[0], d[1], d[2], d[3]);
}

// dot32 with the exceptional arm out of line (warp-uniform branch, as in dot32).
// CHECKED=false: the layer's scales were all scanned into [2, 252] at prepare
// time (m26s_b1_layer_new), where dot32 takes the MMA for every call, so the
// per-MMA warp vote is dropped. The vote cost 30-40% of FC1 (b1_bench, outputs
// bit-identical).
template <int ByteB, bool CHECKED = true>
__device__ __forceinline__ void dot32_m64(float (&d)[4], const ActivationView& x, int k32, const AFragment& a,
                                          uint32_t packed, uint32_t sb) {
    const unsigned scale = (sb >> (8 * ByteB)) & 255;
    if (!CHECKED || __all_sync(0xffffffffu, scale >= 2 && scale <= 252)) {
        const auto b = e2m1_containers(packed);
        mma_e4m3_e2m1<0, ByteB>(d, a.a0, a.a1, a.a2, a.a3, b.x, b.y, a.scale, sb);
    } else {
        const float4 r = dot32_exceptional<ByteB>(make_float4(d[0], d[1], d[2], d[3]), x, k32, packed, sb);
        d[0] = r.x;
        d[1] = r.y;
        d[2] = r.z;
        d[3] = r.w;
    }
}

// One K32 block of one staged K128 tile for this warp's 16 rows x 64 columns.
// The staged word for (kb, nf, lane) holds the four n8 tiles' B fragments at
// consecutive addresses (payload_source), so one 16-byte load serves all four.
template <int KB, bool CHECKED>
__device__ __forceinline__ void fc1_m64_kb(float (&gate)[8][4], float (&up)[8][4], const ActivationView& x,
                                           int kt, int nh, const uint32_t* b, const uint32_t* sf) {
    const int lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const auto a = load_a(x, kt * 4 + KB, c);
#pragma unroll
    for (int nfl = 0; nfl < 2; ++nfl) {
        const int nf = nh * 2 + nfl;
        const int task = (KB * 4 + nf) * 32 + lane;
        const uint4 bu = *reinterpret_cast<const uint4*>(b + task * 4);
        const uint4 bg = *reinterpret_cast<const uint4*>(b + 128 * 16 + task * 4);
        const uint32_t us[4] = {bu.x, bu.y, bu.z, bu.w}, gs[4] = {bg.x, bg.y, bg.z, bg.w};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t su = sf[nf * 32 + t * 8 + g], sg = sf[128 + nf * 32 + t * 8 + g];
            dot32_m64<KB, CHECKED>(gate[nfl * 4 + t], x, kt * 4 + KB, a, gs[t], sg);
            dot32_m64<KB, CHECKED>(up[nfl * 4 + t], x, kt * 4 + KB, a, us[t], su);
        }
    }
}

// One pipeline stage in dynamic shared memory: the up+gate payload words, then
// the packed scale words.
constexpr int kFc1StageWords = 128 * 32 + 256;
constexpr size_t fc1_smem_bytes(int stages) { return size_t(stages) * kFc1StageWords * 4; }

// Grid (4 slices, chunks), 256 threads, fc1_smem_bytes(STAGES) dynamic shared
// memory. STAGES-1 K tiles are in flight while one is consumed: 2 suits prefill
// (two CTAs per SM), deeper pipelines suit decode (few CTAs, latency-bound).
// Tiles are consumed in K order at any depth, so results are bit-identical.
// Faults per (chunk, slice): 1 plan fault, 2 bad chunk/group metadata, 3 staging
// geometry.
// G groups per chunk (perf reset P7): G = 4 is M64 (256 threads); G = 8 is M128
// (512 threads, two warps per group as before), which streams an expert's
// up+gate tiles once for up to 128 rows. Each accumulator's MMA sequence is the
// same at any G, so `mid` is bit-identical.
template <int STAGES, bool CHECKED = true, int G = 4>
__global__ void __launch_bounds__(64 * G, (STAGES <= 2 && G == 4) ? 2 : 1)
fc1_m64(Pool pool, const uint8_t* prepared, const int32_t* slots, PlanStorage plan, const Chunk* chunks,
        const uint32_t* payload, const uint8_t* scales, float* mid, uint32_t* faults) {
    static_assert(STAGES >= 2 && STAGES <= 5, "pipeline depth");
    static_assert(G == 4 || G == 8, "groups per chunk");
    constexpr uint32_t kThreads = 64 * G;
    const uint32_t ci = blockIdx.y, slice = blockIdx.x, tid = threadIdx.x, fi = ci * 4 + slice;
    if (tid == 0) faults[fi] = 0;
    if (*plan.fault) {
        if (tid == 0) faults[fi] = 1;
        return;
    }
    const Chunk ch = chunks[ci];
    const int warp = tid / 32, lane = tid % 32, g = lane / 4, c = lane % 4;
    const int m = warp / 2, nh = warp % 2;
    const bool shape_ok = ch.groups >= 1 && ch.groups <= G && ch.first_group >= 0 &&
                          uint32_t(ch.first_group + ch.groups) <= *plan.group_count;
    const bool live = shape_ok && m < ch.groups;
    Group group{};
    int32_t expert = -1;
    if (shape_ok) {
        expert = plan.groups[ch.first_group].expert;
        if (live) group = plan.groups[ch.first_group + m];
    }
    const bool bad = !shape_ok || expert < 0 || expert >= 256 || slots[expert] < 0 ||
                     (live && (group.expert != expert || group.rows < 1 || group.rows > 16));
    if (__syncthreads_or(bad)) {
        if (tid == 0) faults[fi] = 2;
        return;
    }
    extern __shared__ __align__(16) uint32_t fc1_smem[];
    auto b_of = [&](int st) { return fc1_smem + st * kFc1StageWords; };
    auto sf_of = [&](int st) { return fc1_smem + st * kFc1StageWords + 128 * 32; };
    const ActivationView x{payload, scales, 1024, 128, live ? group.input_rows[g] : -1,
                           live ? group.input_rows[g + 8] : -1};
    float gate[8][4] = {}, up[8][4] = {};
    const uint32_t slot = uint32_t(slots[expert]);
    // Issue one K128 tile (up then gate, payload + packed scales) into `buf`.
    auto stage = [&](int kt, int buf) -> bool {
        Tile t{slot, slice * 128, uint32_t(kt * 128), 128, false, false, true};
        const bool ok_up = stage_mlp<true>(pool, t, prepared, reinterpret_cast<uint8_t*>(b_of(buf)),
                                           reinterpret_cast<uint8_t*>(sf_of(buf)), tid, kThreads);
        t.gate = true;
        const bool ok_gate = stage_mlp<true>(pool, t, prepared, reinterpret_cast<uint8_t*>(b_of(buf) + 128 * 16),
                                             reinterpret_cast<uint8_t*>(sf_of(buf) + 128), tid, kThreads);
        asm volatile("cp.async.commit_group;" ::: "memory");
        return ok_up && ok_gate;  // Block-uniform: depends on tile geometry only.
    };
    // Prologue: tiles 0..STAGES-2 in flight.
    for (int st = 0; st < STAGES - 1; ++st) {
        if (!stage(st, st)) {
            if (tid == 0) faults[fi] = 3;
            return;
        }
    }
    for (int kt = 0; kt < 32; ++kt) {
        // Groups committed so far: STAGES-1+kt. Tile kt has landed once at most
        // STAGES-2 younger groups are still pending.
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();  // tile kt visible; every warp is past tile kt-1, so its buffer is free
        const int nk = kt + STAGES - 1;
        if (nk < 32) {
            if (!stage(nk, nk % STAGES)) {
                if (tid == 0) faults[fi] = 3;
                return;
            }
        } else {
            asm volatile("cp.async.commit_group;" ::: "memory");  // empty group keeps the count uniform
        }
        const int buf = kt % STAGES;
        if (live) {  // Warp-uniform: every lane of a warp shares m.
            fc1_m64_kb<0, CHECKED>(gate, up, x, kt, nh, b_of(buf), sf_of(buf));
            fc1_m64_kb<1, CHECKED>(gate, up, x, kt, nh, b_of(buf), sf_of(buf));
            fc1_m64_kb<2, CHECKED>(gate, up, x, kt, nh, b_of(buf), sf_of(buf));
            fc1_m64_kb<3, CHECKED>(gate, up, x, kt, nh, b_of(buf), sf_of(buf));
        }
    }
    if (!live) return;
    // fc1_finish's layout: [group][16 rows][512], zero padded rows.
    float* out = mid + size_t(ch.first_group + m) * 16 * 512 + slice * 128;
#pragma unroll
    for (int nfl = 0; nfl < 2; ++nfl) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + (e / 2) * 8, col = (nh * 2 + nfl) * 32 + t * 8 + 2 * c + e % 2;
                const float gv = row < group.rows ? gate[nfl * 4 + t][e] : 0,
                            uv = row < group.rows ? up[nfl * 4 + t][e] : 0;
                out[row * 512 + col] = row < group.rows ? silu_up(gv, uv) : 0;
            }
        }
    }
}

// Perf reset L5: FC1 below prefill sizes, one CTA per (16-row group, 128-column
// slice), weights straight to registers.
//
// At 1-16 rows every chunk is one group, so fc1_m64 runs 2 of its 8 warps: the
// other 6 wait at the per-tile barrier while each live warp chains 64 MMAs behind
// the 17 KB cp.async tile and four dependent A-fragment loads (Nsight Compute, 4
// rows: long scoreboard 46%, barrier 39%, 30% occupancy). Most chunks stay one
// group up to prefill sizes (the dumped frame: 113 chunks for 121 groups at 64
// rows, 193 for 249 at 256).
//
// Here each of the CTA's 4 warps owns one n32 column block (all four n8 tiles, up
// and gate) and loads its own operands from the prepared image into registers:
// no shared memory and no CTA barrier in the K loop. Tile kt+1's weights and A
// fragments are in flight while tile kt's 32 MMAs run. A lane's 16 bytes for
// (kb, n32 block nf) are the bytes stage_mlp stages for transfer
// (kb*4 + nf)*32 + lane (payload_source); its scale word is N row nf*32 + lane of
// the tile's packed K32 scales (scale16_source), so the warp reads its 32 rows as
// one 128-byte run and shuffles row t*8 + g to the lanes of n8 tile t. Every
// accumulator sees fc1_m64's MMA sequence (same operands, kt then kb order) and
// the same SiLU*up finish: `mid` is bit-identical. The groups of one expert are
// adjacent CTAs reading the same tiles in step, so a multi-group expert's
// re-reads come from L2 (inferred from timing: still 7-14% under fc1_m64 at 255
// rows). Unchecked MMAs only (a layer scanned all-normal). Faults per (group,
// slice), as connected_fc1: 1 plan fault, 2 bad group metadata, 3 staging
// geometry.

// One K128 tile of one warp's n32 block: B fragment words [kb][n8 tile] for up
// and gate, and this lane's packed scale row word for each.
struct Fc1Regs {
    uint32_t up[4][4], gate[4][4], su, sg;
};

// The prepared N256/K128 layout advances 16,384 payload bytes and 1,024 scale
// bytes per K128 tile and 4,096 payload bytes per K32 block (payload_source,
// scale16_source).
__device__ __forceinline__ void fc1_regs_load(Fc1Regs& r, const uint8_t* up, const uint8_t* gate, const uint8_t* su,
                                              const uint8_t* sg, int kt) {
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
        const uint4 u = *reinterpret_cast<const uint4*>(up + kt * 16384 + kb * 4096);
        const uint4 v = *reinterpret_cast<const uint4*>(gate + kt * 16384 + kb * 4096);
        r.up[kb][0] = u.x;
        r.up[kb][1] = u.y;
        r.up[kb][2] = u.z;
        r.up[kb][3] = u.w;
        r.gate[kb][0] = v.x;
        r.gate[kb][1] = v.y;
        r.gate[kb][2] = v.z;
        r.gate[kb][3] = v.w;
    }
    r.su = *reinterpret_cast<const uint32_t*>(su + kt * 1024);
    r.sg = *reinterpret_cast<const uint32_t*>(sg + kt * 1024);
}

template <int KB>
__device__ __forceinline__ void fc1_decode_kb(float (&gate)[4][4], float (&up)[4][4], const ActivationView& x, int kt,
                                              const AFragment& a, const Fc1Regs& r, const uint32_t (&su)[4],
                                              const uint32_t (&sg)[4]) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        dot32_m64<KB, false>(gate[t], x, kt * 4 + KB, a, r.gate[KB][t], sg[t]);
        dot32_m64<KB, false>(up[t], x, kt * 4 + KB, a, r.up[KB][t], su[t]);
    }
}

// Grid (4 slices, groups), 128 threads, no dynamic shared memory.
__global__ void __launch_bounds__(128)
fc1_decode(Pool pool, const uint8_t* prepared, const int32_t* slots, PlanStorage plan, const uint32_t* payload,
           const uint8_t* scales, float* mid, uint32_t* faults) {
    const uint32_t gi = blockIdx.y, slice = blockIdx.x, tid = threadIdx.x, fi = gi * 4 + slice;
    if (tid == 0) faults[fi] = 0;
    if (*plan.fault) {
        if (tid == 0) faults[fi] = 1;
        return;
    }
    const Group group = plan.groups[gi];
    if (group.rows < 1 || group.rows > 16 || group.expert < 0 || group.expert >= 256 || slots[group.expert] < 0) {
        if (tid == 0) faults[fi] = 2;
        return;
    }
    const int nf = tid / 32, lane = tid % 32, g = lane / 4, c = lane % 4;
    // stage_mlp's geometry check for all 64 tiles the CTA reads: thread tid takes
    // K tile tid % 32 of up (warps 0, 2) or gate (warps 1, 3).
    Tile tile{uint32_t(slots[group.expert]), slice * 128, (tid % 32) * 128, 128, false, (tid / 32) % 2 == 1, true};
    if (__syncthreads_or(check_scale16_tile(pool, tile) != Status::ok)) {
        if (tid == 0) faults[fi] = 3;
        return;
    }
    tile.k_start = 0;
    tile.gate = false;
    const uint8_t* up_p = prepared + payload_source(tile, nf * 32 + lane);
    const uint8_t* up_s = prepared + scale16_source(tile, nf * 8 + lane / 4) + lane % 4 * 4;
    tile.gate = true;
    const uint8_t* gate_p = prepared + payload_source(tile, nf * 32 + lane);
    const uint8_t* gate_s = prepared + scale16_source(tile, nf * 8 + lane / 4) + lane % 4 * 4;
    const ActivationView x{payload, scales, 1024, 128, group.input_rows[g], group.input_rows[g + 8]};
    float gate[4][4] = {}, up[4][4] = {};
    Fc1Regs r[2];
    fc1_regs_load(r[0], up_p, gate_p, up_s, gate_s, 0);
    AFragment a[4];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) a[kb] = load_a(x, kb, c);
    for (int kt = 0; kt < 32; kt += 2) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {  // tile k in r[b]; tile k+1 and its A fragments load meanwhile
            const int k = kt + b;
            AFragment an[4] = {};
            if (k + 1 < 32) {
                fc1_regs_load(r[b ^ 1], up_p, gate_p, up_s, gate_s, k + 1);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) an[kb] = load_a(x, (k + 1) * 4 + kb, c);
            }
            uint32_t su[4], sg[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                su[t] = __shfl_sync(0xffffffffu, r[b].su, t * 8 + g);
                sg[t] = __shfl_sync(0xffffffffu, r[b].sg, t * 8 + g);
            }
            fc1_decode_kb<0>(gate, up, x, k, a[0], r[b], su, sg);
            fc1_decode_kb<1>(gate, up, x, k, a[1], r[b], su, sg);
            fc1_decode_kb<2>(gate, up, x, k, a[2], r[b], su, sg);
            fc1_decode_kb<3>(gate, up, x, k, a[3], r[b], su, sg);
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) a[kb] = an[kb];
        }
    }
    // fc1_finish's layout: [group][16 rows][512], zero padded rows.
    float* out = mid + size_t(gi) * 16 * 512 + slice * 128;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = g + (e / 2) * 8, col = nf * 32 + t * 8 + 2 * c + e % 2;
            const float gv = row < group.rows ? gate[t][e] : 0, uv = row < group.rows ? up[t][e] : 0;
            out[row * 512 + col] = row < group.rows ? silu_up(gv, uv) : 0;
        }
    }
}

// FC2 operand policy for a layer whose scales all lie in [2, 252] (scanned at
// prepare time): Fc2Fp8's dot32 without the per-MMA warp vote, i.e. exactly its
// MMA arm.
struct Fc2Fp8Fast {
    using View = ActivationView;
    using Fragment = AFragment;
    __device__ __forceinline__ static Fragment load(const View& x, int kb, int c) { return load_a(x, kb, c); }
    template <int ByteB>
    __device__ __forceinline__ static void step(float (&acc)[4], const View&, int, const Fragment& a, uint32_t packed,
                                                uint32_t scale) {
        const auto b = e2m1_containers(packed);
        mma_e4m3_e2m1<0, ByteB>(acc, a.a0, a.a1, a.a2, a.a3, b.x, b.y, a.scale, scale);
    }
};

// Perf reset F2: FC2 with fc1_m64's expert weight reuse. connected_fc2_fp8 runs
// one CTA per (16-row group, 128-column output tile) and re-stages the tile's
// W2 slices (4 x 8.5 KB) for every group: ~1,120 groups x 1 MB per 2,048-row
// layer. Here one CTA takes one output tile for a CHUNK of up to four groups of
// one expert (fc1_m64's chunks) and stages each K128 slice once. All four slices
// are issued up front (34 KB, four cp.async groups) and consumed in K order.
//
// Eight warps: warp = (group m = warp / 2, column half nh = warp % 2), holding
// n32 blocks 2nh and 2nh+1 with all four n8 tiles in each (connected_fc2_fp8's
// warps 0..3). Every accumulator sees connected_fc2_fp8's MMA sequence (slice
// then kb order, the same A/B fragments and scale words), so the output is
// bit-identical. Faults keep connected_fc2_fp8's [group][32 tiles] layout: 1
// plan fault, 2 bad chunk/group metadata, 3 staging geometry, 4 quantizer
// fault, 5 route index out of range.
constexpr int kFc2SliceWords = 128 * 16 + 128;  // W2 payload (8 KB) + one packed scale word per N row
constexpr size_t kFc2SmemBytes = size_t(4) * kFc2SliceWords * 4;

template <typename Policy>
__global__ void __launch_bounds__(256, 2)
fc2_m64(Pool pool, const uint8_t* prepared, const int32_t* slots, PlanStorage plan, const Chunk* chunks,
        const uint8_t* payload, const uint8_t* scales, const uint32_t* quant_faults, float* output, uint32_t routes,
        uint32_t* faults) {
    const uint32_t ci = blockIdx.y, ot = blockIdx.x, tid = threadIdx.x;
    const Chunk ch = chunks[ci];
    const bool shape_ok = ch.groups >= 1 && ch.groups <= 4 && ch.first_group >= 0 &&
                          uint32_t(ch.first_group + ch.groups) <= *plan.group_count;
    // One fault word per (group, tile), as connected_fc2_fp8; a malformed chunk
    // reports at its own index (inside [ng * 32] since chunks <= groups).
    auto fault = [&](uint32_t code) {
        if (tid == 0) {
            if (shape_ok)
                for (int m = 0; m < ch.groups; ++m) faults[uint32_t(ch.first_group + m) * 32 + ot] = code;
            else
                faults[ci * 32 + ot] = code;
        }
    };
    if (*plan.fault) {
        fault(1);
        return;
    }
    const int warp = tid / 32, lane = tid % 32, g = lane / 4, c = lane % 4;
    const int m = warp / 2, nh = warp % 2;
    const bool live = shape_ok && m < ch.groups;
    Group group{};
    int32_t expert = -1;
    if (shape_ok) {
        expert = plan.groups[ch.first_group].expert;
        if (live) group = plan.groups[ch.first_group + m];
    }
    bool bad = !shape_ok || expert < 0 || expert >= 256 || slots[expert] < 0;
    if (shape_ok)
        for (int k = 0; k < ch.groups; ++k) {
            const Group gk = plan.groups[ch.first_group + k];
            bad |= gk.expert != expert || gk.rows < 1 || gk.rows > 16;
        }
    if (__syncthreads_or(bad)) {
        fault(2);
        return;
    }
    bool qf = false;
    for (int k = 0; k < ch.groups; ++k) qf |= quant_faults[uint32_t(ch.first_group + k) * 256 + tid] != 0;
    if (__syncthreads_or(qf)) {
        fault(4);
        return;
    }
    extern __shared__ __align__(16) uint32_t fc2_smem[];
    const uint32_t slot = uint32_t(slots[expert]);
    for (int s = 0; s < 4; ++s) {
        const Tile t{slot, ot * 128, uint32_t(s * 128), 128, true, false, true};
        uint32_t* b = fc2_smem + s * kFc2SliceWords;
        const bool ok = stage_mlp<true>(pool, t, prepared, reinterpret_cast<uint8_t*>(b),
                                        reinterpret_cast<uint8_t*>(b + 128 * 16), tid, 256);
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (__syncthreads_or(!ok)) {  // Block-uniform: tile geometry only.
            fault(3);
            return;
        }
    }
    float acc[8][4] = {};
    const uint32_t gi = uint32_t(ch.first_group + m);
    for (int s = 0; s < 4; ++s) {
        if (s == 0) asm volatile("cp.async.wait_group 3;" ::: "memory");
        else if (s == 1) asm volatile("cp.async.wait_group 2;" ::: "memory");
        else if (s == 2) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (!live) continue;  // Warp-uniform.
        const uint32_t* b = fc2_smem + s * kFc2SliceWords;
        const uint32_t* sf = b + 128 * 16;
        const ActivationView x{reinterpret_cast<const uint32_t*>(payload) + size_t(gi) * 16 * 128 + s * 32,
                               scales + size_t(gi) * 256 + s * 4, 128, 16, g < group.rows ? g : -1,
                               g + 8 < group.rows ? g + 8 : -1};
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            const auto a = Policy::load(x, kb, c);
#pragma unroll
            for (int nfl = 0; nfl < 2; ++nfl) {
                const int nf = nh * 2 + nfl;
                const uint4 w = *reinterpret_cast<const uint4*>(b + ((kb * 4 + nf) * 32 + lane) * 4);
                const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t sw = sf[nf * 32 + t * 8 + g];
                    if (kb == 0) Policy::template step<0>(acc[nfl * 4 + t], x, kb, a, ws[t], sw);
                    else if (kb == 1) Policy::template step<1>(acc[nfl * 4 + t], x, kb, a, ws[t], sw);
                    else if (kb == 2) Policy::template step<2>(acc[nfl * 4 + t], x, kb, a, ws[t], sw);
                    else Policy::template step<3>(acc[nfl * 4 + t], x, kb, a, ws[t], sw);
                }
            }
        }
    }
    const int lo = live && g < group.rows ? plan.original_routes[group.route_base + g] : 0;
    const int hi = live && g + 8 < group.rows ? plan.original_routes[group.route_base + g + 8] : 0;
    if (__syncthreads_or(lo < 0 || uint32_t(lo) >= routes || hi < 0 || uint32_t(hi) >= routes)) {
        fault(5);
        return;
    }
    if (tid < 4 && int(tid) < ch.groups) faults[uint32_t(ch.first_group + tid) * 32 + ot] = 0;
    if (!live) return;
#pragma unroll
    for (int nfl = 0; nfl < 2; ++nfl) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + (e / 2) * 8, col = ot * 128 + (nh * 2 + nfl) * 32 + t * 8 + 2 * c + e % 2;
                if (row < group.rows) output[size_t(e / 2 ? hi : lo) * 4096 + col] = acc[nfl * 4 + t][e];
            }
        }
    }
}

// Prepare-time scan: does any UE8M0 weight scale of the prepared images fall
// outside [2, 252] (dot32's exceptional arm)? The s13 and s2 regions hold only
// scale bytes.
__global__ void scale_scan(const uint8_t* __restrict__ prepared, uint32_t images, uint32_t* exceptional) {
    constexpr uint64_t w13 = region_bytes(Region::s13) / 4, per = w13 + region_bytes(Region::s2) / 4;
    const uint64_t total = uint64_t(images) * per;
    bool bad = false;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t img = i / per, w = i % per;
        const uint64_t off = img * image_bytes +
                             (w < w13 ? region_offset(Region::s13) + w * 4 : region_offset(Region::s2) + (w - w13) * 4);
        const uint32_t v = *reinterpret_cast<const uint32_t*>(prepared + off);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t s = (v >> (8 * k)) & 255;
            bad |= s < 2 || s > 252;
        }
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0) atomicOr(exceptional, 1u);
}

}  // namespace m26b1
