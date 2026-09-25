// Spark B1 serving path (perf reset R3, docs/design/perf-reset-vs-ds41rt.md):
// the E-W4A8-v1 expert FFN for one rank, on the B1 tensor-core compute core
// (crates/mimo26-expert/kernels/b1, a hand port of b12x w4a8_v41_slice at H4096/I512).
//
// Per request: E4M3 rows + K32 scales (the wire payload as received, no decode),
// top-8 expert ids + FP32 route weights  ->  parallel single-CTA route plan
// -> connected FC1 (scale-both) -> quantizer v1 on the intermediate -> connected
// FC2 -> ordered route reduce -> the rank's BF16 partial [rows, 4096].
//
// The planner replaces B1's serial one-thread scaffold (group_plan.cu) with one
// 1024-thread CTA (after upstream SparkInfer 7fcc094, "plan V4.1 decode routes in
// one CTA"). Route placement inside an expert group is not stable, which cannot
// change a value: every row's FC1/quantizer/FC2 arithmetic is row-local and the
// reduce reads routes in original order.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "grouped.cu"
#include "connected.cu"
#include "quant_v1.cuh"
#include "route_reduce.cuh"
#include "b1_fc1_m64.cuh"

namespace {

constexpr int kExperts = 256;
constexpr int kTopk = 8;
constexpr int kPlanThreads = 1024;

__global__ void __launch_bounds__(kPlanThreads)
plan_parallel(const int32_t* __restrict__ ids, const float* __restrict__ w, uint32_t rows,
              const uint8_t* __restrict__ resident, m26b1::PlanStorage s, m26b1::Chunk* chunks,
              uint32_t* chunk_count, uint32_t gpc) {
    __shared__ uint32_t cnt[kExperts], gbase[kExperts], rbase[kExperts], cursor[kExperts], cbase[kExperts];
    __shared__ uint32_t bad;
    const uint32_t tid = threadIdx.x, nt = blockDim.x, routes = rows * kTopk;
    for (uint32_t e = tid; e < kExperts; e += nt) { cnt[e] = 0; cursor[e] = 0; }
    if (tid == 0) bad = 0;
    __syncthreads();
    for (uint32_t r = tid; r < routes; r += nt) {
        const int32_t e = ids[r];
        const float wt = w[r];
        bool ok = e >= 0 && e < kExperts && resident[e] && wt >= 0.0f && wt <= 0x1.fffffep127f;
        if (ok) {
            for (uint32_t k = (r / kTopk) * kTopk; k < r; ++k) ok = ok && ids[k] != e;  // no duplicate in a row
        }
        if (!ok) atomicOr(&bad, 1u); else atomicAdd(&cnt[e], 1u);
    }
    __syncthreads();
    if (bad) {
        if (tid == 0) { *s.fault = 1u; *s.group_count = 0u; *chunk_count = 0u; }
        return;
    }
    if (tid == 0) {
        uint32_t g = 0, rb = 0, ch = 0;
        for (int e = 0; e < kExperts; ++e) {
            const uint32_t ng = (cnt[e] + 15u) / 16u;
            gbase[e] = g; rbase[e] = rb; cbase[e] = ch;
            g += ng; rb += cnt[e]; ch += (ng + gpc - 1u) / gpc;
        }
        *s.group_count = g;
        *chunk_count = ch;
        *s.fault = 0u;
    }
    __syncthreads();
    for (uint32_t r = tid; r < routes; r += nt) {
        const int32_t e = ids[r];
        const uint32_t pos = atomicAdd(&cursor[e], 1u);
        const uint32_t gr = rbase[e] + pos;
        s.original_routes[gr] = int32_t(r);
        s.inverse[r] = int32_t(gr);
        s.grouped_weights[gr] = w[r];
        s.groups[gbase[e] + pos / 16u].input_rows[pos % 16u] = int32_t(r / kTopk);
    }
    for (uint32_t e = tid; e < kExperts; e += nt) {
        const uint32_t c = cnt[e], ng = (c + 15u) / 16u;
        // FC1 weight-reuse chunks (b1_fc1_m64.cuh): up to `gpc` groups of one expert.
        for (uint32_t j = 0; j * gpc < ng; ++j)
            chunks[cbase[e] + j] =
                m26b1::Chunk{int32_t(gbase[e] + gpc * j), int32_t(ng - gpc * j < gpc ? ng - gpc * j : gpc)};
        for (uint32_t k = 0; k * 16u < c; ++k) {
            m26b1::Group& G = s.groups[gbase[e] + k];
            const uint32_t n = c - 16u * k < 16u ? c - 16u * k : 16u;
            G.expert = int32_t(e);
            G.rows = int32_t(n);
            G.route_base = int32_t(rbase[e] + 16u * k);
            for (uint32_t slot = n; slot < 16u; ++slot) G.input_rows[slot] = -1;
        }
    }
}

// Route reduce over BF16 route outputs (perf reset P8, opt-in MIMO26_B1_Y=bf16):
// route_reduce.cu's reduce_one on bf16 -> f32 inputs, in the same route order
// with the same RN multiply/add and fault rules, BF16 RNE out; no FP32 partial
// (the serving path never reads it). Eight columns per thread (16-B loads).
__global__ void __launch_bounds__(256) reduce_bf16y(const uint16_t* __restrict__ y, const float* __restrict__ w,
                                                    uint32_t rows, uint16_t* __restrict__ out,
                                                    uint32_t* __restrict__ fault) {
    const uint64_t n8 = uint64_t(rows) * 4096 / 8;
    bool failed = false;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = i / 512, col = (i % 512) * 8;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        bool ok = true;
        uint4 z[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) z[r] = *reinterpret_cast<const uint4*>(y + ((row * 8 + r) * 4096 + col));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float wt = w[row * 8 + r];
            ok = ok && wt >= 0.0f && wt <= 0x1.fffffep127f;
            const uint32_t words[4] = {z[r].x, z[r].y, z[r].z, z[r].w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float zv = __uint_as_float((k & 1 ? words[k / 2] & 0xffff0000u : words[k / 2] << 16));
                ok = ok && zv >= -0x1.fffffep127f && zv <= 0x1.fffffep127f;
                v[k] = __fadd_rn(v[k], __fmul_rn(wt, zv));
                ok = ok && v[k] >= -0x1.fffffep127f && v[k] <= 0x1.fffffep127f;
            }
        }
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            uint32_t lo = __float_as_uint(v[k]), hi = __float_as_uint(v[k + 1]);
            lo = (lo + 0x7fffu + ((lo >> 16) & 1u)) >> 16;
            hi = (hi + 0x7fffu + ((hi >> 16) & 1u)) >> 16;
            ok = ok && (lo & 0x7f80u) != 0x7f80u && (hi & 0x7f80u) != 0x7f80u;
            o[k / 2] = lo | (hi << 16);
        }
        if (!ok) {
            failed = true;
            o[0] = o[1] = o[2] = o[3] = 0;
        }
        *reinterpret_cast<uint4*>(out + row * 4096 + col) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    if (__syncthreads_or(failed) && threadIdx.x == 0) *fault = 5u;
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    cudaError_t ensure(size_t bytes) {
        if (bytes <= n) return cudaSuccess;
        if (p) cudaFree(p);
        p = nullptr; n = 0;
        const cudaError_t e = cudaMalloc(&p, bytes < 256 ? 256 : bytes);
        if (e == cudaSuccess) n = bytes < 256 ? 256 : bytes;
        return e;
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
    ~Dev() { if (p) cudaFree(p); }
};

void set_err(char* err, size_t len, const char* what, cudaError_t e) {
    if (err && len) std::snprintf(err, len, "%s: %s", what, cudaGetErrorString(e));
}

void set_msg(char* err, size_t len, const char* msg) {
    if (err && len) std::snprintf(err, len, "%s", msg);
}

}  // namespace

struct m26s_b1_layer {
    uint8_t* prepared = nullptr;
    int32_t* slots = nullptr;
    uint8_t* resident = nullptr;
    m26b1::Pool pool{};
    // Every UE8M0 weight scale in [2, 252] (prepare-time scan): the FC1/FC2 MMAs
    // can skip dot32's per-MMA exceptional-scale vote.
    bool all_normal = false;
};

// Perf reset L3: the OR of every fault word of one FFN call (FC1, quantizer,
// FC2, reduce) into one word, so the host reads 4 bytes per request instead of
// ~27 KB (decode) to ~1.3 MB (2K prefill) in four copies, and scans the full
// arrays only when it is nonzero. Grid-stride over up to 48 blocks (one per
// GB10 SM); `out` is zeroed first and each block ORs its result in once.
__global__ void __launch_bounds__(256) fault_any(const uint32_t* __restrict__ f1, uint32_t n1,
                                                 const uint32_t* __restrict__ q, uint32_t nq,
                                                 const uint32_t* __restrict__ f2, uint32_t n2,
                                                 const uint32_t* __restrict__ r, uint32_t* __restrict__ out) {
    const uint32_t stride = gridDim.x * blockDim.x, t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = 0;
    for (uint32_t i = t; i < n1; i += stride) v |= f1[i];
    for (uint32_t i = t; i < nq; i += stride) v |= q[i];
    for (uint32_t i = t; i < n2; i += stride) v |= f2[i];
    if (t == 0) v |= r[0];
    v = __reduce_or_sync(0xffffffffu, v);
    __shared__ uint32_t warp_or[8];
    if ((threadIdx.x & 31) == 0) warp_or[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t o = 0;
        for (int i = 0; i < 8; ++i) o |= warp_or[i];
        if (o) atomicOr(out, o);
    }
}

struct m26s_b1_scratch {
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr, ea = nullptr, eb = nullptr, ec = nullptr;
    Dev x, xs, inv, orig, gw, groups, chunks, mid, dq, dqs, qfault, f1, f2, y, partial, bf16, rfault, any, meta, iw;
    std::vector<uint8_t> hiw;  // route ids then weights, packed for one upload
    std::vector<uint32_t> hf;
};

extern "C" {

// Prepare one layer's 256 canonical-v2 rank images (device, contiguous) into the
// B1P1 pool. The canonical buffer may be freed after this returns.
int m26s_b1_layer_new(const uint8_t* canonical_dev, uint64_t canonical_bytes, uint32_t rank,
                      m26s_b1_layer** out, char* err, size_t errlen) {
    const uint64_t need = uint64_t(kExperts) * m26b1::image_bytes;
    if (!canonical_dev || canonical_bytes != need || rank >= 4 || !out) {
        set_msg(err, errlen, "b1 layer: bad canonical extent/rank");
        return 1;
    }
    auto* L = new m26s_b1_layer();
    cudaError_t e = cudaMalloc(&L->prepared, need);
    if (e != cudaSuccess) { set_err(err, errlen, "b1 layer prepared alloc", e); delete L; return 2; }
    m26b1::PreparedInfo info{};
    for (int s = 0; s < kExperts; ++s) {
        e = m26b1::prepare_async({2, 4096, 512, rank, m26b1::image_bytes},
                                 canonical_dev + uint64_t(s) * m26b1::image_bytes, m26b1::image_bytes,
                                 L->prepared + uint64_t(s) * m26b1::image_bytes, m26b1::image_bytes, &info, nullptr);
        if (e != cudaSuccess) { set_err(err, errlen, "b1 prepare", e); cudaFree(L->prepared); delete L; return 3; }
    }
    std::vector<int32_t> slots(kExperts);
    std::vector<uint8_t> present(kExperts, 1);
    for (int s = 0; s < kExperts; ++s) slots[s] = s;
    if ((e = cudaMalloc(&L->slots, kExperts * 4)) != cudaSuccess ||
        (e = cudaMalloc(&L->resident, kExperts)) != cudaSuccess ||
        (e = cudaMemcpy(L->slots, slots.data(), kExperts * 4, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaMemcpy(L->resident, present.data(), kExperts, cudaMemcpyHostToDevice)) != cudaSuccess ||
        (e = cudaDeviceSynchronize()) != cudaSuccess) {
        set_err(err, errlen, "b1 layer metadata", e);
        return 4;
    }
    L->pool = m26b1::Pool{info, need, uint32_t(kExperts)};
    {
        uint32_t* flag = nullptr;
        uint32_t h = 1;
        if ((e = cudaMalloc(&flag, 4)) != cudaSuccess || (e = cudaMemset(flag, 0, 4)) != cudaSuccess) {
            set_err(err, errlen, "b1 layer scale scan alloc", e);
            return 4;
        }
        m26b1::scale_scan<<<1024, 256>>>(L->prepared, kExperts, flag);
        if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaMemcpy(&h, flag, 4, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            cudaFree(flag);
            set_err(err, errlen, "b1 layer scale scan", e);
            return 4;
        }
        cudaFree(flag);
        L->all_normal = h == 0;
    }
    *out = L;
    return 0;
}

// Whether the layer's weight scales all lie in [2, 252] (1) or not (0).
int m26s_b1_layer_all_normal(const m26s_b1_layer* L) { return L && L->all_normal ? 1 : 0; }

void m26s_b1_layer_free(m26s_b1_layer* L) {
    if (!L) return;
    cudaFree(L->prepared);
    cudaFree(L->slots);
    cudaFree(L->resident);
    delete L;
}

int m26s_b1_scratch_new(m26s_b1_scratch** out, char* err, size_t errlen) {
    auto* S = new m26s_b1_scratch();
    cudaError_t e;
    if ((e = cudaStreamCreateWithFlags(&S->st, cudaStreamNonBlocking)) != cudaSuccess ||
        (e = cudaEventCreate(&S->e0)) != cudaSuccess || (e = cudaEventCreate(&S->e1)) != cudaSuccess ||
        (e = cudaEventCreate(&S->ea)) != cudaSuccess || (e = cudaEventCreate(&S->eb)) != cudaSuccess ||
        (e = cudaEventCreate(&S->ec)) != cudaSuccess) {
        set_err(err, errlen, "b1 scratch stream/events", e);
        delete S;
        return 1;
    }
    *out = S;
    return 0;
}

void m26s_b1_scratch_free(m26s_b1_scratch* S) {
    if (!S) return;
    if (S->e0) cudaEventDestroy(S->e0);
    if (S->e1) cudaEventDestroy(S->e1);
    if (S->ea) cudaEventDestroy(S->ea);
    if (S->eb) cudaEventDestroy(S->eb);
    if (S->ec) cudaEventDestroy(S->ec);
    if (S->st) cudaStreamDestroy(S->st);
    delete S;
}

// One rank's FFN for `rows` tokens. Host inputs: payload rows of 4096 E4M3 at
// `payload_pitch` bytes, scale rows of 128 UE8M0 at `scales_pitch` (4096/128 for
// separate arrays; the request frame's interleaved hidden rows use 4224 for
// both, read in place from the registered receive slot), ids/weights [rows*8]
// token-major top-8. Output:
// bf16_out [rows*4096]. ms[0..3] = upload+plan, GPU compute, download+checks,
// groups (as a float); ms[4..7] = the GPU phases FC1, quantizer, FC2, reduce.
// Returns 0 on success.
int m26s_b1_ffn(const m26s_b1_layer* L, m26s_b1_scratch* S, const uint8_t* payload, size_t payload_pitch,
                const uint8_t* scales, size_t scales_pitch, const int32_t* ids, const float* weights,
                uint32_t rows, uint16_t* bf16_out, float* ms, char* err, size_t errlen) {
    if (!L || !S || !rows || rows > 4096 || payload_pitch < 4096 || scales_pitch < 128) {
        set_msg(err, errlen, "b1 ffn: bad rows/handles/pitch");
        return 1;
    }
    const uint32_t cap = rows <= 256 ? 256u : rows <= 2048 ? 2048u : 4096u;
    const uint64_t routes = uint64_t(rows) * kTopk, route_cap = uint64_t(cap) * kTopk;
    cudaError_t e;
    auto t0 = std::chrono::steady_clock::now();
#define M26S_CK(expr, what) do { if ((e = (expr)) != cudaSuccess) { set_err(err, errlen, what, e); return 2; } } while (0)
    M26S_CK(S->x.ensure(size_t(rows) * 4096), "b1 x alloc");
    M26S_CK(S->xs.ensure(size_t(rows) * 128), "b1 xs alloc");
    // Route ids and weights in one device buffer, uploaded in one copy (perf reset L3).
    M26S_CK(S->iw.ensure(routes * 8), "b1 ids/w alloc");
    int32_t* const ids_d = S->iw.as<int32_t>();
    float* const w_d = reinterpret_cast<float*>(S->iw.as<uint8_t>() + routes * 4);
    M26S_CK(S->inv.ensure(route_cap * 4), "b1 inv alloc");
    M26S_CK(S->orig.ensure(route_cap * 4), "b1 orig alloc");
    M26S_CK(S->gw.ensure(route_cap * 4), "b1 gw alloc");
    M26S_CK(S->groups.ensure(route_cap * sizeof(m26b1::Group)), "b1 groups alloc");
    // The plan's group count, fault and chunk count share one buffer: one download.
    M26S_CK(S->meta.ensure(16), "b1 plan meta alloc");
    uint32_t* const count_d = S->meta.as<uint32_t>();
    const uint64_t chunk_cap = route_cap / 16 + kExperts;
    M26S_CK(S->chunks.ensure(chunk_cap * sizeof(m26b1::Chunk)), "b1 chunks alloc");

    M26S_CK(cudaMemcpy2DAsync(S->x.p, 4096, payload, payload_pitch, 4096, rows, cudaMemcpyHostToDevice, S->st),
            "b1 x upload");
    M26S_CK(cudaMemcpy2DAsync(S->xs.p, 128, scales, scales_pitch, 128, rows, cudaMemcpyHostToDevice, S->st),
            "b1 xs upload");
    S->hiw.resize(routes * 8);
    std::memcpy(S->hiw.data(), ids, routes * 4);
    std::memcpy(S->hiw.data() + routes * 4, weights, routes * 4);
    M26S_CK(cudaMemcpyAsync(S->iw.p, S->hiw.data(), routes * 8, cudaMemcpyHostToDevice, S->st), "b1 ids/w upload");
    const m26b1::PlanStorage ps{S->inv.as<int32_t>(), S->orig.as<int32_t>(), S->gw.as<float>(),
                                S->groups.as<m26b1::Group>(), count_d, count_d + 1,
                                route_cap, route_cap};
    // FC1 groups per weight-reuse chunk (perf reset P7): 4 (M64, default) or 8
    // (M128, MIMO26_B1_FC1_GROUPS=8). Bit-identical, but M128 is slower (b1_bench
    // rank 0 layer 1: FC1 3.68 vs 3.26 ms at 2,048 rows, 5.07 vs 4.70 at 4,096):
    // M64's re-reads of an expert already hit L2 and M128 halves the CTAs.
    static const uint32_t gpc = [] {
        const char* v = std::getenv("MIMO26_B1_FC1_GROUPS");
        return v && std::strcmp(v, "8") == 0 ? 8u : 4u;
    }();
    plan_parallel<<<1, kPlanThreads, 0, S->st>>>(ids_d, w_d, rows, L->resident, ps, S->chunks.as<m26b1::Chunk>(),
                                                  count_d + 2, gpc);
    M26S_CK(cudaGetLastError(), "b1 plan launch");
    uint32_t hc[3] = {0, 0, 0};  // group count, fault, chunk count
    M26S_CK(cudaMemcpyAsync(hc, count_d, 12, cudaMemcpyDeviceToHost, S->st), "b1 plan meta download");
    M26S_CK(cudaStreamSynchronize(S->st), "b1 plan sync");
    if (hc[1]) { set_msg(err, errlen, "b1 plan: invalid routes (id/residency/weight/duplicate)"); return 3; }
    const uint32_t ng = hc[0], nc = hc[2];
    if (!ng || ng > route_cap) { set_msg(err, errlen, "b1 plan: bad group count"); return 3; }
    if (!nc || nc > ng || nc > chunk_cap) { set_msg(err, errlen, "b1 plan: bad chunk count"); return 3; }
    // FC1 schedule: M64 weight-reuse chunks (default from 256 rows), fc1_decode
    // (below), or the M16 connected_fc1 (MIMO26_B1_FC1=m16, the R3 reference for
    // A/B). MIMO26_B1_FC1=m64 keeps fc1_m64 at every size (A/B).
    static const int fc1_mode = [] {
        const char* v = std::getenv("MIMO26_B1_FC1");
        return v && std::strcmp(v, "m16") == 0 ? 1 : v && std::strcmp(v, "m64") == 0 ? 2 : 0;
    }();
    const bool fc1_m16 = fc1_mode == 1;
    // Unchecked MMAs only for a layer scanned all-normal; MIMO26_B1_CHECKED=1 forces
    // the conforming per-MMA exceptional-scale check everywhere (A/B).
    static const bool force_checked = [] {
        const char* v = std::getenv("MIMO26_B1_CHECKED");
        return v && std::strcmp(v, "1") == 0;
    }();
    const bool checked = force_checked || !L->all_normal;
    // Perf reset L5: below 256 rows an unchecked layer runs fc1_decode, one CTA per
    // (group, slice), `mid` bit-identical to fc1_m64 (b1_bench rank 0 layer 1, FC1
    // against MIMO26_B1_FC1=m64: 0.268 -> 0.242 ms at 4 rows, 0.676 -> 0.592 at 16,
    // 1.344 -> 1.180 at 64, 2.303 -> 1.981 at 255). Prefill sizes and the checked
    // kernel keep fc1_m64.
    const bool fc1_dec = fc1_mode == 0 && !checked && rows < 256;
    const size_t f1n = size_t(fc1_m16 || fc1_dec ? ng : nc) * 4;
    auto t1 = std::chrono::steady_clock::now();
    const size_t mid_n = size_t(ng) * 16 * 512;
    M26S_CK(S->mid.ensure(mid_n * 4), "b1 mid alloc");
    M26S_CK(S->dq.ensure(mid_n), "b1 dq alloc");
    M26S_CK(S->dqs.ensure(mid_n / 32), "b1 dqs alloc");
    M26S_CK(S->qfault.ensure(mid_n / 32 * 4), "b1 qfault alloc");
    M26S_CK(S->f1.ensure(f1n * 4), "b1 f1 alloc");
    M26S_CK(S->f2.ensure(size_t(ng) * 32 * 4), "b1 f2 alloc");
    // Route outputs: FP32 (E-W4A8-v1, default) or BF16 (perf reset P8, opt-in
    // MIMO26_B1_Y=bf16: half the FC2 write and reduce read; a numerical-mode
    // change, not bit-identical).
    static const bool y_bf16 = [] {
        const char* v = std::getenv("MIMO26_B1_Y");
        return v && std::strcmp(v, "bf16") == 0;
    }();
    M26S_CK(S->y.ensure(routes * 4096 * (y_bf16 ? 2 : 4)), "b1 y alloc");
    M26S_CK(S->partial.ensure(size_t(rows) * 4096 * 4), "b1 partial alloc");
    M26S_CK(S->bf16.ensure(size_t(rows) * 4096 * 2), "b1 bf16 alloc");
    M26S_CK(S->rfault.ensure(4), "b1 rfault alloc");
    M26S_CK(cudaEventRecord(S->e0, S->st), "b1 event 0");
    if (fc1_m16) {
        m26b1::connected_fc1<128, true><<<dim3(512 / 128, ng), 128, 0, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->x.as<uint32_t>(), S->xs.as<uint8_t>(), S->mid.as<float>(),
            nullptr, nullptr, S->f1.as<uint32_t>());
    } else if (fc1_dec) {
        m26b1::fc1_decode<<<dim3(512 / 128, ng), 128, 0, S->st>>>(L->pool, L->prepared, L->slots, ps,
                                                                   S->x.as<uint32_t>(), S->xs.as<uint8_t>(),
                                                                   S->mid.as<float>(), S->f1.as<uint32_t>());
    } else {
        // Pipeline depth (b1_bench, rank 0 layer 1, outputs bit-identical at every
        // depth). With the per-MMA check two stages win at every size (1 row: FC1
        // 0.079 / 0.091 / 0.091 ms for 2 / 3 / 4 stages; deeper pipelines cost
        // occupancy); the checked kernel's best at 1 row was 3 stages, 0.149 ms.
        // MIMO26_B1_FC1_STAGES=2..4 overrides the unchecked depth.
        static const int forced = [] {
            const char* v = std::getenv("MIMO26_B1_FC1_STAGES");
            const int n = v ? std::atoi(v) : 0;
            return n >= 2 && n <= 4 ? n : 0;
        }();
        const int stages = forced ? forced : 2;
        static bool attrs = false;
        if (!attrs) {
            M26S_CK(cudaFuncSetAttribute(m26b1::fc1_m64<3, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(m26b1::fc1_smem_bytes(3))), "b1 fc1 smem 3");
            M26S_CK(cudaFuncSetAttribute(m26b1::fc1_m64<4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(m26b1::fc1_smem_bytes(4))), "b1 fc1 smem 4");
            M26S_CK(cudaFuncSetAttribute(m26b1::fc1_m64<3, false, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(m26b1::fc1_smem_bytes(3))), "b1 fc1 smem 3 g8");
            M26S_CK(cudaFuncSetAttribute(m26b1::fc1_m64<4, false, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(m26b1::fc1_smem_bytes(4))), "b1 fc1 smem 4 g8");
            attrs = true;
        }
#define M26S_FC1(ST, CK, G)                                                                                  \
    m26b1::fc1_m64<ST, CK, G><<<dim3(512 / 128, nc), 64 * G, m26b1::fc1_smem_bytes(ST), S->st>>>(            \
        L->pool, L->prepared, L->slots, ps, S->chunks.as<m26b1::Chunk>(), S->x.as<uint32_t>(),               \
        S->xs.as<uint8_t>(), S->mid.as<float>(), S->f1.as<uint32_t>())
        if (gpc == 8) {
            if (checked) M26S_FC1(2, true, 8);  // the conforming kernel (an exceptional scale in the layer)
            else if (stages == 4) M26S_FC1(4, false, 8);
            else if (stages == 3) M26S_FC1(3, false, 8);
            else M26S_FC1(2, false, 8);
        } else {
            if (checked) M26S_FC1(2, true, 4);
            else if (stages == 4) M26S_FC1(4, false, 4);
            else if (stages == 3) M26S_FC1(3, false, 4);
            else M26S_FC1(2, false, 4);
        }
#undef M26S_FC1
    }
    M26S_CK(cudaGetLastError(), "b1 fc1 launch");
    M26S_CK(cudaEventRecord(S->ea, S->st), "b1 event a");
    M26S_CK(m26b1::quantize_async({S->mid.as<float>(), mid_n},
                                  {S->dq.as<uint8_t>(), S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(), mid_n,
                                   mid_n / 32, mid_n / 32},
                                  S->st),
            "b1 quantize");
    M26S_CK(cudaEventRecord(S->eb, S->st), "b1 event b");
    // FC2 schedule: the per-group connected_fc2_fp8 (default) or the M64
    // weight-reuse chunks (MIMO26_B1_FC2=m64, perf reset F2). The M64 kernel is
    // bit-identical but slower (b1_bench rank 0 layer 1: 3.10 vs 2.70 ms at
    // 2,048 rows, 5.06 vs 4.61 ms at 4,096): FC2's weight re-reads already hit
    // L2, and one CTA per chunk runs fewer warps per SM.
    static const bool fc2_g16 = [] {
        const char* v = std::getenv("MIMO26_B1_FC2");
        return !(v && std::strcmp(v, "m64") == 0) || gpc != 4 || y_bf16;  // fc2_m64: 4-group chunks, FP32 y
    }();
    if (!fc2_g16 && checked)
        m26b1::fc2_m64<m26b1::Fc2Fp8><<<dim3(32, nc), 256, m26b1::kFc2SmemBytes, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->chunks.as<m26b1::Chunk>(), S->dq.as<uint8_t>(),
            S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(), S->y.as<float>(), uint32_t(routes), S->f2.as<uint32_t>());
    else if (!fc2_g16)
        m26b1::fc2_m64<m26b1::Fc2Fp8Fast><<<dim3(32, nc), 256, m26b1::kFc2SmemBytes, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->chunks.as<m26b1::Chunk>(), S->dq.as<uint8_t>(),
            S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(), S->y.as<float>(), uint32_t(routes), S->f2.as<uint32_t>());
    else if (y_bf16 && checked)
        m26b1::connected_fc2_fp8<128, true, m26b1::Fc2Fp8, uint16_t><<<dim3(32, ng), 128, 0, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->dq.as<uint8_t>(), S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(),
            S->y.as<uint16_t>(), uint32_t(routes), S->f2.as<uint32_t>(), w_d, 0u);
    else if (y_bf16)
        m26b1::connected_fc2_fp8<128, true, m26b1::Fc2Fp8Fast, uint16_t><<<dim3(32, ng), 128, 0, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->dq.as<uint8_t>(), S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(),
            S->y.as<uint16_t>(), uint32_t(routes), S->f2.as<uint32_t>(), w_d, 0u);
    else if (checked)
        m26b1::connected_fc2_fp8<128, true><<<dim3(32, ng), 128, 0, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->dq.as<uint8_t>(), S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(),
            S->y.as<float>(), uint32_t(routes), S->f2.as<uint32_t>(), w_d, 0u);
    else
        m26b1::connected_fc2_fp8<128, true, m26b1::Fc2Fp8Fast><<<dim3(32, ng), 128, 0, S->st>>>(
            L->pool, L->prepared, L->slots, ps, S->dq.as<uint8_t>(), S->dqs.as<uint8_t>(), S->qfault.as<uint32_t>(),
            S->y.as<float>(), uint32_t(routes), S->f2.as<uint32_t>(), w_d, 0u);
    M26S_CK(cudaGetLastError(), "b1 fc2 launch");
    M26S_CK(cudaEventRecord(S->ec, S->st), "b1 event c");
    if (y_bf16) {
        M26S_CK(cudaMemsetAsync(S->rfault.p, 0, 4, S->st), "b1 reduce fault clear");
        const uint64_t n8 = uint64_t(rows) * 512;
        const uint32_t blocks = uint32_t(n8 / 256 + 1 < 4096 ? n8 / 256 + 1 : 4096);
        reduce_bf16y<<<blocks, 256, 0, S->st>>>(S->y.as<uint16_t>(), w_d, rows, S->bf16.as<uint16_t>(),
                                                S->rfault.as<uint32_t>());
        M26S_CK(cudaGetLastError(), "b1 reduce bf16y");
    } else {
        M26S_CK(m26b1::reduce_async({rows, S->y.as<float>(), w_d, routes * 4096, routes},
                                    {S->partial.as<float>(), S->bf16.as<uint16_t>(), uint64_t(rows) * 4096,
                                     uint64_t(rows) * 4096, S->rfault.as<uint32_t>()},
                                    S->st),
                "b1 reduce");
    }
    M26S_CK(cudaEventRecord(S->e1, S->st), "b1 event 1");
    M26S_CK(cudaMemcpyAsync(bf16_out, S->bf16.p, size_t(rows) * 4096 * 2, cudaMemcpyDeviceToHost, S->st),
            "b1 bf16 download");
    // Fault words: FC1 [f1n], quantizer [ng*256], FC2 [ng*32], reduce [1]; their OR
    // comes back as one word (perf reset L3), the arrays only when it is nonzero.
    const size_t nf = f1n + mid_n / 32 + size_t(ng) * 32 + 1;
    M26S_CK(S->any.ensure(4), "b1 any alloc");
    M26S_CK(cudaMemsetAsync(S->any.p, 0, 4, S->st), "b1 any clear");
    const size_t fwords = nf - 1;
    const unsigned fblocks = unsigned(fwords / 4096 + 1 < 48 ? fwords / 4096 + 1 : 48);
    fault_any<<<fblocks, 256, 0, S->st>>>(S->f1.as<uint32_t>(), uint32_t(f1n), S->qfault.as<uint32_t>(),
                                    uint32_t(mid_n / 32), S->f2.as<uint32_t>(), uint32_t(ng) * 32,
                                    S->rfault.as<uint32_t>(), S->any.as<uint32_t>());
    M26S_CK(cudaGetLastError(), "b1 fault any");
    uint32_t any_fault = 0;
    M26S_CK(cudaMemcpyAsync(&any_fault, S->any.p, 4, cudaMemcpyDeviceToHost, S->st), "b1 any download");
    M26S_CK(cudaStreamSynchronize(S->st), "b1 compute sync");
    uint32_t* h = nullptr;
    if (any_fault) {
        S->hf.resize(nf);
        h = S->hf.data();
        M26S_CK(cudaMemcpy(h, S->f1.p, f1n * 4, cudaMemcpyDeviceToHost), "b1 f1 download");
        M26S_CK(cudaMemcpy(h + f1n, S->qfault.p, mid_n / 32 * 4, cudaMemcpyDeviceToHost), "b1 qfault download");
        M26S_CK(cudaMemcpy(h + f1n + mid_n / 32, S->f2.p, size_t(ng) * 32 * 4, cudaMemcpyDeviceToHost),
                "b1 f2 download");
        M26S_CK(cudaMemcpy(h + nf - 1, S->rfault.p, 4, cudaMemcpyDeviceToHost), "b1 rfault download");
    }
#undef M26S_CK
    float gpu_ms = 0.0f, ph[4] = {0, 0, 0, 0};
    cudaEventElapsedTime(&gpu_ms, S->e0, S->e1);
    cudaEventElapsedTime(&ph[0], S->e0, S->ea);
    cudaEventElapsedTime(&ph[1], S->ea, S->eb);
    cudaEventElapsedTime(&ph[2], S->eb, S->ec);
    cudaEventElapsedTime(&ph[3], S->ec, S->e1);
    for (size_t i = 0; h && i < nf; ++i) {
        if (h[i]) {
            const char* stage = i < f1n ? "fc1" : i < f1n + mid_n / 32 ? "quantizer"
                              : i + 1 < nf ? "fc2" : "reduce";
            if (err && errlen) std::snprintf(err, errlen, "b1 %s fault %u at word %zu (groups %u)", stage, h[i], i, ng);
            return 4;
        }
    }
    if (any_fault) {  // flagged on the device but no word found on the host: still a failure
        if (err && errlen) std::snprintf(err, errlen, "b1 fault word %u (groups %u)", any_fault, ng);
        return 4;
    }
    auto t2 = std::chrono::steady_clock::now();
    if (ms) {
        ms[0] = std::chrono::duration<float, std::milli>(t1 - t0).count();
        ms[1] = gpu_ms;
        ms[2] = std::chrono::duration<float, std::milli>(t2 - t1).count() - gpu_ms;
        ms[3] = float(ng);
        for (int i = 0; i < 4; ++i) ms[4 + i] = ph[i];
    }
    return 0;
}

}  // extern "C"
