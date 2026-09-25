// R18c phase-wise mixed-M B2 decode dispatch. Owned new code; the frozen
// exact-M arithmetic bodies (stage_x, lane_scale, gemm loop, reduction,
// activate) are copied VERBATIM from expert_gemm.cu. Only the group/tile/
// validity scaffolding is replaced by a per-CTA width switch so one kernel per
// phase covers every active expert. C1-w8 decode widths are 1..8; wider
// (M16/M64 prefill) uses the separate native-wide adapter, not this kernel.
#pragma once
#include "mimo26_expert_kernels.h"
#include "mimo26_expert_device.cuh"
#include "mimo26_expert_tile.h"

namespace m26x_mixed {

struct Geom { int rows, cols, payload, scales; };
__host__ __device__ Geom geometry(int p) {
    if (p == M26X_PROJ_GATE) return {M26X_GATE_ROWS,M26X_GATE_COLS,M26X_GATE_PAYLOAD_OFF,M26X_GATE_SCALE_OFF};
    if (p == M26X_PROJ_UP) return {M26X_UP_ROWS,M26X_UP_COLS,M26X_UP_PAYLOAD_OFF,M26X_UP_SCALE_OFF};
    return {M26X_DOWN_ROWS,M26X_DOWN_COLS,M26X_DOWN_PAYLOAD_OFF,M26X_DOWN_SCALE_OFF};
}

// Verbatim from expert_gemm.cu stage_x<M>.
template<int M>
__device__ void stage_x(float* tile, const float* x, int start, int stop, int cols, int k0) {
    for (int v = threadIdx.x; v < M*64; v += M26X_THREADS) {
        const int t = start + v/64, k = (v%64)*4;
        const float* src = x + uint64_t(t < stop ? t : start)*cols + k0 + k;
        m26x::copy16(tile + (v/64)*256 + m26x_x_swizzle(k), src, t < stop);
    }
    m26x::commit_copies();
}
// Verbatim from expert_gemm.cu lane_scale.
__device__ uint32_t lane_scale(const uint8_t* s, int k0, int cols, uint32_t naive) {
    const int lane = threadIdx.x % 8;
    if (naive & M26X_NAIVE_SCALE_OFF_BY_ONE)
        return m26x::scale_byte(s,k0/32+lane,cols/32,naive);
    uint2 pair = make_uint2(0,0);
    if (!lane) pair = *reinterpret_cast<const uint2*>(s+k0/32);
    const uint32_t lo = __shfl_sync(0xffffffffu,pair.x,0,8);
    const uint32_t hi = __shfl_sync(0xffffffffu,pair.y,0,8);
    return ((lane < 4 ? lo : hi) >> ((lane%4)*8)) & 255u;
}
// Verbatim from expert_gemm.cu activate.
__global__ void activate(float* gate, const float* up, uint64_t count) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) { const float g=gate[i]; gate[i]=(g/(1.0f+expf(-g)))*up[i]; }
}

// The frozen gemm<M,Correct> body, with the group/tile/validity scaffolding
// moved to mixed_gemm_impl. Arithmetic, loop order, shared layout and reduction
// are unchanged so stored rows are bitwise-identical to the frozen per-M launch.
// The __shared__ tile array is declared per body (static shared addressing, as
// in the frozen kernel) so the dispatcher holds minimal live state across the
// width switch; only one body runs per CTA.
template<int M, bool Correct>
__device__ __forceinline__ void gemm_body(float* activations,
    const uint8_t* image, const float* x,
    Geom g, int begin, int end, int row, int lane, int dtype, uint32_t flags, void* out) {
    const uint32_t naive = Correct ? 0u : flags;
    const uint8_t* w = image + g.payload + uint64_t(row)*(g.cols/2);
    const uint8_t* s = image + g.scales + uint64_t(row)*(g.cols/32);
    float acc0[M] = {}, acc1[M] = {};
    stage_x<M>(activations,x,begin,end,g.cols,0);
    stage_x<M>(activations+M*256,x,begin,end,g.cols,256);
    uint4 current = *reinterpret_cast<const uint4*>(w+lane*16);
    uint32_t current_scale = lane_scale(s,0,g.cols,naive);
    m26x::wait_one();
    __syncthreads();
    for (int k0=0; k0<g.cols; k0+=256) {
        uint4 next = make_uint4(0,0,0,0);
        uint32_t next_scale = 0;
        if (k0+256 < g.cols) {
            next = *reinterpret_cast<const uint4*>(w+(k0+256)/2+lane*16);
            next_scale = lane_scale(s,k0+256,g.cols,naive);
        }
        const float* tile = activations + ((k0/256)&1)*M*256;
        #pragma unroll
        for (int word=0; word<4; ++word) {
            const uint32_t packed = word==0 ? current.x : word==1 ? current.y : word==2 ? current.z : current.w;
            #pragma unroll
            for (int pair=0; pair<4; ++pair) {
                const uint32_t byte = (packed >> (pair*8)) & 255u;
                const int swap = (naive & M26X_NAIVE_NIBBLE_SWAP) ? 4 : 0;
                const float lo = m26x::decode((byte >> swap)&15u,current_scale,naive);
                const float hi = m26x::decode((byte >> (4-swap))&15u,current_scale,naive);
                const int k = lane*32+word*8+pair*2;
                #pragma unroll
                for (int t=0; t<M; ++t) {
                    acc0[t] = m26x::round_acc(fmaf(lo,tile[t*256+m26x_x_swizzle(k)],acc0[t]),naive);
                    acc1[t] = m26x::round_acc(fmaf(hi,tile[t*256+m26x_x_swizzle(k+1)],acc1[t]),naive);
                }
            }
        }
        __syncthreads();
        if (k0+512 < g.cols)
            stage_x<M>(activations+((k0/256)&1)*M*256,x,begin,end,g.cols,k0+512);
        if (k0+256 < g.cols) {
            if (k0+512 < g.cols) m26x::wait_one();
            else m26x::wait_all();
            __syncthreads();
        }
        current=next; current_scale=next_scale;
    }
    #pragma unroll
    for (int t=0; t<M; ++t) {
        float sum = acc0[t]+acc1[t];
        #pragma unroll
        for (int d=4; d; d/=2) sum += __shfl_down_sync(0xffffffffu,sum,d,8);
        if (!lane && begin+t < end) m26x::store(out,uint64_t(begin+t)*g.rows+row,sum,dtype);
    }
}

template<bool Correct>
__global__ __launch_bounds__(M26X_THREADS) void mixed_gemm(m26x_plan plan,
    const uint8_t* __restrict__ grouped, const float* __restrict__ x,
    Geom g, int dtype, uint32_t flags, void* __restrict__ out) {
    const uint32_t naive = Correct ? 0u : flags;
    const int group = int(blockIdx.y);
    if (group >= plan.n_groups) {
        if ((naive & M26X_NAIVE_PAD_ROW_READ) && !threadIdx.x && !blockIdx.x && !blockIdx.z) {
            const volatile uint8_t* bad = grouped + uint64_t(plan.resident_experts)*M26X_QUARTER_SLICE_BYTES;
            const uint8_t poison = *bad;
            atomicOr(plan.fault,8u | uint32_t(poison));
        }
        return;
    }
    const int begin = plan.group_offsets[group], end = plan.group_offsets[group+1];
    if (begin < 0 || end < begin || end > plan.total_tokens || end-begin > plan.max_m) {
        if (!threadIdx.x) atomicOr(plan.fault,1u);
        return;
    }
    const int m = end - begin;
    if (m < 1) return;
    if (m > 8) { if (!threadIdx.x) atomicOr(plan.fault,4u); return; } // decode-only kernel; M>8 tiles are an I5 item
    if (begin >= end) return;
    const int expert = plan.expert_ids[group];
    if (expert < 0 || expert >= plan.resident_experts) {
        if (!threadIdx.x) atomicOr(plan.fault,2u);
        return;
    }
    const int row = m26x_owned_row(blockIdx.x,threadIdx.x);
    const int lane = threadIdx.x % 8;
    const uint8_t* image = grouped + uint64_t(expert)*M26X_QUARTER_SLICE_BYTES;
    __shared__ __align__(16) float activations[2*8*256];
    switch (m) {
        case 1: gemm_body<1,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 2: gemm_body<2,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 3: case 4: gemm_body<4,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 5: gemm_body<5,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 6: gemm_body<6,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 7: gemm_body<7,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        case 8: gemm_body<8,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out); break;
        default: break;
    }
}

// Host dispatch helper: one launch per phase over every group. Correct and
// naive are separate kernel instantiations (separate entries); the correct path
// is the timed entry and must not spill.
template<bool Correct>
cudaError_t launch_mixed(const m26x_plan& p,const uint8_t* w,const float* x,Geom g,int dtype,
    uint32_t naive,void* out,cudaStream_t stream) {
    if (!p.total_tokens) return cudaSuccess;
    const dim3 grid(g.rows/M26X_ROWS_PER_BLOCK, p.n_groups+p.padded_groups, 1);
    mixed_gemm<Correct><<<grid,M26X_THREADS,0,stream>>>(p,w,x,g,dtype,naive,out);
    return cudaGetLastError();
}
cudaError_t dispatch_mixed(const m26x_plan& p,const uint8_t* w,const float* x,Geom g,int dtype,
    uint32_t naive,void* out,cudaStream_t stream) {
    return naive ? launch_mixed<false>(p,w,x,g,dtype,naive,out,stream)
                 : launch_mixed<true >(p,w,x,g,dtype,naive,out,stream);
}

// O1 diet exploration (Track D, pre-registered 6469de4): the per-M sibling — no
// width switch — with the dynamic per-M `2*M*256` f32 ring (M7 = 14,336 B + 1,024
// = 15,360; M8 = 16,384 + 1,024). `group_index` maps blockIdx.y to the full-plan
// group index (the caller buckets groups by M). sm_121a (CUDA 13.0) re-baseline:
// frozen `gemm<8,1,0>` is 64 regs, the width-switch `mixed_gemm` is 80, and the
// per-M sibling is <=64 regs for M<=7 (M7 64 / M4 48 / M2 44 / M1 63) and 73 for
// M8 — so the O1 "<=64 regs => cap 4" diet transfers to sm_121 for the 455/512
// M<=7 steps; the 57/512 M8 steps stay the wider entry (cap 3). The sm_120a
// (CUDA 12.8) 102-reg reading was a misleading host proxy, not the target.
template<int M, bool Correct>
__global__ __launch_bounds__(M26X_THREADS) void mixed_gemm_per_m(m26x_plan plan,
    const int* __restrict__ group_index,
    const uint8_t* __restrict__ grouped, const float* __restrict__ x,
    Geom g, int dtype, uint32_t flags, void* __restrict__ out) {
    const uint32_t naive = Correct ? 0u : flags;
    const int group = group_index[blockIdx.y];
    const int begin = plan.group_offsets[group], end = plan.group_offsets[group+1];
    // A sibling is keyed by the BODY width M (1,2,4,5,6,7,8); groups with
    // 1 <= m <= M share it (m=3 rides the M=4 body, store bounded by `end`).
    if (begin < 0 || end < begin || end > plan.total_tokens || end - begin < 1 || end - begin > M) {
        if (!threadIdx.x) atomicOr(plan.fault,1u);
        return;
    }
    const int expert = plan.expert_ids[group];
    if (expert < 0 || expert >= plan.resident_experts) {
        if (!threadIdx.x) atomicOr(plan.fault,2u);
        return;
    }
    const int row = m26x_owned_row(blockIdx.x,threadIdx.x);
    const int lane = threadIdx.x % 8;
    const uint8_t* image = grouped + uint64_t(expert)*M26X_QUARTER_SLICE_BYTES;
    __shared__ __align__(16) float activations[2*M*256];
    gemm_body<M,Correct>(activations,image,x,g,begin,end,row,lane,dtype,flags,out);
}
// Explicit instantiations so ptxas reports the per-M register/shared footprint
// even before the per-M host dispatch is wired (the O1 record-pin check).
template __global__ void mixed_gemm_per_m<1,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<2,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<4,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<5,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<6,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<7,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<8,true>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<1,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<2,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<4,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<5,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<6,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<7,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);
template __global__ void mixed_gemm_per_m<8,false>(m26x_plan,const int*,const uint8_t*,const float*,Geom,int,uint32_t,void*);

// O1 per-M host dispatch: bucket the groups by body width M (1,2,4,5,6,7,8) and
// launch one sibling per non-empty class. `group_index_dev` is a caller-owned
// device int array of size n_groups, reused across classes. `host_offsets` is the
// host copy of `group_offsets` (n_groups+1 entries).
template<bool Correct>
cudaError_t launch_mixed_per_m(const m26x_plan& p,const int32_t* host_offsets,
    const uint8_t* w,const float* x,Geom g,int dtype,uint32_t naive,void* out,
    int* group_index_dev,cudaStream_t stream) {
    if (!p.total_tokens) return cudaSuccess;
    const int body_M[7] = {1,2,4,5,6,7,8};
    int bucket[256];
    for (int mi=0;mi<7;++mi) {
        const int M=body_M[mi], prev=mi==0?0:body_M[mi-1];
        int n=0;
        for (int gi=0;gi<p.n_groups;++gi) {
            const int m=host_offsets[gi+1]-host_offsets[gi];
            if (m>prev && m<=M) bucket[n++]=gi;
        }
        if (n==0) continue;
        cudaMemcpyAsync(group_index_dev,bucket,size_t(n)*sizeof(int),cudaMemcpyHostToDevice,stream);
        const dim3 grid(g.rows/M26X_ROWS_PER_BLOCK,n,1);
        switch (M) {
            case 1: mixed_gemm_per_m<1,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 2: mixed_gemm_per_m<2,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 4: mixed_gemm_per_m<4,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 5: mixed_gemm_per_m<5,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 6: mixed_gemm_per_m<6,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 7: mixed_gemm_per_m<7,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
            case 8: mixed_gemm_per_m<8,Correct><<<grid,M26X_THREADS,0,stream>>>(p,group_index_dev,w,x,g,dtype,naive,out); break;
        }
        const cudaError_t err=cudaGetLastError(); if (err!=cudaSuccess) return err;
    }
    return cudaSuccess;
}
cudaError_t dispatch_mixed_per_m(const m26x_plan& p,const int32_t* host_offsets,
    const uint8_t* w,const float* x,Geom g,int dtype,uint32_t naive,void* out,
    int* gi,cudaStream_t stream) {
    return naive ? launch_mixed_per_m<false>(p,host_offsets,w,x,g,dtype,naive,out,gi,stream)
                 : launch_mixed_per_m<true >(p,host_offsets,w,x,g,dtype,naive,out,gi,stream);
}

} // namespace m26x_mixed

// Phase-wise coalesced FFN: one gate, one up, one activation, one down kernel
// per layer-step. validate_launch is re-declared here to reuse the frozen AOT
// and overlap checks; the frozen m26x_expert_ffn_v2 is NOT modified.
static inline bool m26x_mixed_validate(const m26x_plan* p,const void* w,const void* x,int proj,int dtype,const void* out) {
    const auto& valid_plan=[](const m26x_plan* q){
        return q && q->layout_version==M26X_LAYOUT_VERSION && q->resident_experts>0 &&
            q->n_groups>=0 && q->n_groups<=256 && q->padded_groups>=0 && q->padded_groups<=256 &&
            (q->capacity_class==256||q->capacity_class==2048||q->capacity_class==4096) &&
            q->total_tokens>=0 && q->total_tokens<=8*q->capacity_class &&
            q->max_m>=1 && q->max_m<=8 &&
            (!q->total_tokens||(q->n_groups>0 && q->max_m>0)) &&
            q->grouped_bytes>=uint64_t(q->resident_experts)*M26X_QUARTER_SLICE_BYTES;
    };
    const auto aligned16=[](const void* a){return a && !(reinterpret_cast<uintptr_t>(a)&15u);};
    const auto overlaps=[](const void* a,uint64_t an,const void* b,uint64_t bn){
        const uintptr_t av=reinterpret_cast<uintptr_t>(a),bv=reinterpret_cast<uintptr_t>(b);
        return av<=bv ? bv-av<an : av-bv<bn;
    };
    if(!valid_plan(p)||proj<0||proj>2||(dtype!=0&&dtype!=1)||
       !aligned16(w)||!aligned16(x)||!aligned16(out)||!p->fault||
       !p->expert_ids||!p->group_offsets) return false;
    const auto g=m26x_mixed::geometry(proj);
    const uint64_t xb=uint64_t(p->total_tokens)*g.cols*4;
    const uint64_t ob=uint64_t(p->total_tokens)*g.rows*(dtype==0?4:2);
    if(p->x_bytes<xb||p->out_bytes<ob||overlaps(x,xb,out,ob)||overlaps(w,p->grouped_bytes,out,ob)) return false;
    return true;
}

extern "C" cudaError_t m26x_expert_ffn_mixed_v2(const m26x_plan* p,const uint8_t* w,
    const float* x,int32_t dtype,uint32_t naive,float* scratch,void* out,m26x_stream_t stream) {
    using namespace m26x_mixed;
    if(!m26x_mixed_validate(p,w,x,M26X_PROJ_GATE,dtype,out)) return cudaErrorInvalidValue;
    const uint64_t count=uint64_t(p->total_tokens)*512;
    const uint64_t output_bytes=uint64_t(p->total_tokens)*4096*(dtype==0?4:2);
    const auto aligned16=[](const void* a){return a && !(reinterpret_cast<uintptr_t>(a)&15u);};
    const auto overlaps=[](const void* a,uint64_t an,const void* b,uint64_t bn){
        const uintptr_t av=reinterpret_cast<uintptr_t>(a),bv=reinterpret_cast<uintptr_t>(b);
        return av<=bv ? bv-av<an : av-bv<bn;
    };
    if(!aligned16(scratch)||p->scratch_bytes<2*count*4||p->out_bytes<output_bytes||
       overlaps(scratch,p->scratch_bytes,out,output_bytes)||overlaps(scratch,p->scratch_bytes,x,p->x_bytes)||
       overlaps(scratch,p->scratch_bytes,w,p->grouped_bytes)||overlaps(x,p->x_bytes,out,output_bytes))
        return cudaErrorInvalidValue;
    if(!p->total_tokens) return cudaSuccess;
    cudaError_t err=dispatch_mixed(*p,w,x,geometry(0),0,naive,scratch,stream); if(err!=cudaSuccess) return err;
    err=dispatch_mixed(*p,w,x,geometry(1),0,naive,scratch+count,stream); if(err!=cudaSuccess) return err;
    activate<<<unsigned((count+255)/256),256,0,stream>>>(scratch,scratch+count,count);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    return dispatch_mixed(*p,w,scratch,geometry(2),dtype,naive,out,stream);
}

// O1 entry (Track D, pre-registered 6469de4): the same phase-wise FFN as
// m26x_expert_ffn_mixed_v2, but dispatched to per-M siblings (<=64 regs for
// M<=7 on sm_121a) instead of the width-switch kernel. `host_offsets` is the
// host copy of `group_offsets` (n_groups+1) used for the M bucketing; the caller
// owns it and it must match the device array. Bitwise-identical to the width-
// switch entry; it is the O1 timed subject, not a promotion.
extern "C" cudaError_t m26x_expert_ffn_mixed_o1_v2(const m26x_plan* p,const int32_t* host_offsets,
    const uint8_t* w,const float* x,int32_t dtype,uint32_t naive,float* scratch,void* out,m26x_stream_t stream) {
    using namespace m26x_mixed;
    if(!m26x_mixed_validate(p,w,x,M26X_PROJ_GATE,dtype,out)||!host_offsets) return cudaErrorInvalidValue;
    const uint64_t count=uint64_t(p->total_tokens)*512;
    const uint64_t output_bytes=uint64_t(p->total_tokens)*4096*(dtype==0?4:2);
    const auto aligned16=[](const void* a){return a && !(reinterpret_cast<uintptr_t>(a)&15u);};
    const auto overlaps=[](const void* a,uint64_t an,const void* b,uint64_t bn){
        const uintptr_t av=reinterpret_cast<uintptr_t>(a),bv=reinterpret_cast<uintptr_t>(b);
        return av<=bv ? bv-av<an : av-bv<bn;
    };
    if(!aligned16(scratch)||p->scratch_bytes<2*count*4||p->out_bytes<output_bytes||
       overlaps(scratch,p->scratch_bytes,out,output_bytes)||overlaps(scratch,p->scratch_bytes,x,p->x_bytes)||
       overlaps(scratch,p->scratch_bytes,w,p->grouped_bytes)||overlaps(x,p->x_bytes,out,output_bytes))
        return cudaErrorInvalidValue;
    if(!p->total_tokens) return cudaSuccess;
    int* group_index=nullptr;
    cudaError_t err=cudaMalloc(&group_index,size_t(p->n_groups)*sizeof(int)); if(err!=cudaSuccess) return err;
    err=dispatch_mixed_per_m(*p,host_offsets,w,x,geometry(0),0,naive,scratch,group_index,stream); if(err!=cudaSuccess){cudaFree(group_index);return err;}
    err=dispatch_mixed_per_m(*p,host_offsets,w,x,geometry(1),0,naive,scratch+count,group_index,stream); if(err!=cudaSuccess){cudaFree(group_index);return err;}
    activate<<<unsigned((count+255)/256),256,0,stream>>>(scratch,scratch+count,count);
    err=cudaGetLastError(); if(err!=cudaSuccess){cudaFree(group_index);return err;}
    err=dispatch_mixed_per_m(*p,host_offsets,w,scratch,geometry(2),dtype,naive,out,group_index,stream);
    cudaFree(group_index);
    return err;
}
