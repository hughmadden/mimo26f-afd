/* First-party layout-v2 grouped MXFP4 implementation.
 * B2: 32 rows/CTA, eight lanes/row, vector loads, token-inner reuse,
 * arithmetic decode, two-stage K256 activation pipeline. M>8 is explicitly
 * an M8-tiled SIMT fallback, NOT the proposed tensor-core prefill path.
 */
#include "mimo26_expert_kernels.h"
#include "mimo26_expert_device.cuh"
#include "mimo26_expert_tile.h"
#ifndef M26X_DIAGNOSTIC_FORCE4
#define M26X_DIAGNOSTIC_FORCE4 0
#endif
static_assert(M26X_DIAGNOSTIC_FORCE4 == 0 || M26X_DIAGNOSTIC_FORCE4 == 1);
#ifndef M26X_EXACT_M
#define M26X_EXACT_M 0
#endif
static_assert(M26X_EXACT_M == 0 || M26X_EXACT_M == 1);
static_assert(!(M26X_EXACT_M && M26X_DIAGNOSTIC_FORCE4), "experimental policies must be isolated");
#include <algorithm>
#include <math.h>
#ifndef M26X_BAKED_ARCH
#error "compile with an explicit M26X_BAKED_ARCH"
#endif
#ifndef M26X_BAKED_SMS
#error "compile with an explicit M26X_BAKED_SMS"
#endif
#ifndef M26X_CAPACITY_CLASS
#error "compile with an explicit M26X_CAPACITY_CLASS"
#endif
static_assert(M26X_CAPACITY_CLASS == 256 || M26X_CAPACITY_CLASS == 2048 || M26X_CAPACITY_CLASS == 4096);
namespace {
struct Geom { int rows, cols, payload, scales; };
__host__ __device__ Geom geometry(int p) {
    if (p == M26X_PROJ_GATE) return {M26X_GATE_ROWS,M26X_GATE_COLS,M26X_GATE_PAYLOAD_OFF,M26X_GATE_SCALE_OFF};
    if (p == M26X_PROJ_UP) return {M26X_UP_ROWS,M26X_UP_COLS,M26X_UP_PAYLOAD_OFF,M26X_UP_SCALE_OFF};
    return {M26X_DOWN_ROWS,M26X_DOWN_COLS,M26X_DOWN_PAYLOAD_OFF,M26X_DOWN_SCALE_OFF};
}
bool valid_plan(const m26x_plan* p) {
    return p && p->layout_version == M26X_LAYOUT_VERSION && p->resident_experts > 0 &&
        p->n_groups >= 0 && p->n_groups <= 256 && p->padded_groups >= 0 && p->padded_groups <= 256 &&
        (p->capacity_class == 256 || p->capacity_class == 2048 || p->capacity_class == 4096) &&
        p->total_tokens >= 0 && p->total_tokens <= 8*p->capacity_class &&
        p->max_m >= 0 && p->max_m <= p->capacity_class &&
        (!p->total_tokens || (p->n_groups > 0 && p->max_m > 0)) &&
        p->grouped_bytes >= uint64_t(p->resident_experts)*M26X_QUARTER_SLICE_BYTES;
}
bool aligned16(const void* p) { return p && !(reinterpret_cast<uintptr_t>(p) & 15u); }
bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
    const uintptr_t av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv-av < an : av-bv < bn;
}

template<int M>
__device__ void stage_x(float* tile, const float* x, int start, int stop, int cols, int k0) {
    for (int v = threadIdx.x; v < M*64; v += M26X_THREADS) {
        const int t = start + v/64, k = (v%64)*4;
        const float* src = x + uint64_t(t < stop ? t : start)*cols + k0 + k;
        m26x::copy16(tile + (v/64)*256 + m26x_x_swizzle(k), src, t < stop);
    }
    m26x::commit_copies();
}
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

template<int M, bool Correct, bool Paired=false>
__global__ __launch_bounds__(M26X_THREADS) void gemm(m26x_plan plan,
    const uint8_t* __restrict__ grouped, const float* __restrict__ x,
    Geom g, int dtype, uint32_t flags, void* __restrict__ out) {
    const uint32_t naive = Correct ? 0u : flags;
    const int group = m26x_tile_group<Paired>(int(blockIdx.y));
    if (group >= plan.n_groups) {
        // Correct path returns before ANY metadata or weight access. The wrong
        // path really reads a nonresident slice; guard allocation/sanitizer detects it.
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
    const int start = begin + m26x_tile_offset<Paired,M>(int(blockIdx.y),int(blockIdx.z));
    if (start >= end) return; // no expert-ID/weight read for empty or inactive tiles
    const int expert = plan.expert_ids[group];
    if (expert < 0 || expert >= plan.resident_experts) {
        if (!threadIdx.x) atomicOr(plan.fault,2u);
        return;
    }
    const int row = m26x_owned_row(blockIdx.x,threadIdx.x);
    const int lane = threadIdx.x % 8;
    const uint8_t* image = grouped + uint64_t(expert)*M26X_QUARTER_SLICE_BYTES;
    const uint8_t* w = image + g.payload + uint64_t(row)*(g.cols/2);
    const uint8_t* s = image + g.scales + uint64_t(row)*(g.cols/32);
    __shared__ __align__(16) float activations[2*M*256];
    float acc0[M] = {}, acc1[M] = {};
    // Both v2 K widths are multiples of 256 and contain at least two tiles.
    stage_x<M>(activations,x,start,end,g.cols,0);
    stage_x<M>(activations+M*256,x,start,end,g.cols,256);
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
        __syncthreads(); // all consumers finished before this ring slot is reused
        if (k0+512 < g.cols)
            stage_x<M>(activations+((k0/256)&1)*M*256,x,start,end,g.cols,k0+512);
        if (k0+256 < g.cols) {
            if (k0+512 < g.cols) m26x::wait_one();
            else m26x::wait_all(); // one-group tail: wait_group 1 is NOT sufficient
            __syncthreads();
        }
        current=next; current_scale=next_scale;
    }
    #pragma unroll
    for (int t=0; t<M; ++t) {
        float sum = acc0[t]+acc1[t];
        #pragma unroll
        for (int d=4; d; d/=2) sum += __shfl_down_sync(0xffffffffu,sum,d,8);
        if (!lane && start+t < end) m26x::store(out,uint64_t(start+t)*g.rows+row,sum,dtype);
    }
}
__global__ void activate(float* gate, const float* up, uint64_t count) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) { const float g=gate[i]; gate[i]=(g/(1.0f+expf(-g)))*up[i]; }
}
__global__ void unpack_kernel(const uint8_t* w,const uint8_t* s,int rows,int cols,uint32_t naive,float* out) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<uint64_t(rows)*cols) {
        const uint64_t row=i/cols;
        out[i]=m26x::unpack(w+row*(cols/2),s+row*(cols/32),int(i%cols),cols,naive);
    }
}
template<int M, bool Paired=false>
cudaError_t launch(const m26x_plan& p,const uint8_t* w,const float* x,Geom g,int dtype,
    uint32_t naive,void* out,cudaStream_t stream) {
    dim3 grid(g.rows/M26X_ROWS_PER_BLOCK,(p.n_groups+p.padded_groups)*(Paired?2:1),Paired?1:(p.max_m+M-1)/M);
    if(naive) gemm<M,false,Paired><<<grid,M26X_THREADS,0,stream>>>(p,w,x,g,dtype,naive,out);
    else gemm<M,true,Paired><<<grid,M26X_THREADS,0,stream>>>(p,w,x,g,dtype,0,out);
    return cudaGetLastError();
}
cudaError_t dispatch(const m26x_plan& p,const uint8_t* w,const float* x,Geom g,int dtype,
    uint32_t naive,void* out,cudaStream_t stream) {
    if (!p.total_tokens) return cudaSuccess;
#if M26X_DIAGNOSTIC_FORCE4
    // R13 F1 control only. Two adjacent CTAs cover rows0..3 and row4.
    // This duplicates weight/decode instructions and is not a staged-once design.
    if(p.max_m==5) return launch<4,true>(p,w,x,g,dtype,naive,out,stream);
#endif
    if(p.max_m<=1) return launch<1>(p,w,x,g,dtype,naive,out,stream);
    if(p.max_m<=2) return launch<2>(p,w,x,g,dtype,naive,out,stream);
    if(p.max_m<=4) return launch<4>(p,w,x,g,dtype,naive,out,stream);
#if M26X_EXACT_M
    // Exact-M control: only padded work changes; lane/K order stays fixed.
    if(p.max_m==5) return launch<5>(p,w,x,g,dtype,naive,out,stream);
    if(p.max_m==6) return launch<6>(p,w,x,g,dtype,naive,out,stream);
    if(p.max_m==7) return launch<7>(p,w,x,g,dtype,naive,out,stream);
#endif
    return launch<8>(p,w,x,g,dtype,naive,out,stream);
}
cudaError_t validate_launch(const m26x_plan* p,const void* w,const void* x,int proj,int dtype,const void* out,uint32_t naive) {
    if(!valid_plan(p) || proj<0 || proj>2 || (dtype!=0 && dtype!=1) ||
       !aligned16(w) || !aligned16(x) || !aligned16(out) || !p->fault ||
       !p->expert_ids || !p->group_offsets) return cudaErrorInvalidValue;
    const Geom g=geometry(proj);
    const uint64_t xb=uint64_t(p->total_tokens)*g.cols*4;
    const uint64_t ob=uint64_t(p->total_tokens)*g.rows*(dtype==0?4:2);
    if(p->x_bytes<xb || p->out_bytes<ob || overlaps(x,xb,out,ob) ||
       overlaps(w,p->grouped_bytes,out,ob)) return cudaErrorInvalidValue;
    return m26x_check_aot(p->manifest_arch,p->manifest_sms,p->capacity_class,naive);
}
} // namespace

extern "C" int m26x_validate_host_plan(const m26x_plan* p,const int32_t* ids,const int32_t* offsets) {
    if(!valid_plan(p) || !ids || !offsets || offsets[0]!=0 || offsets[p->n_groups]!=p->total_tokens) return 1;
    int max_m=0;
    for(int i=0;i<p->n_groups;++i) {
        const int a=offsets[i],b=offsets[i+1];
        if(a<0 || b<a || b>p->total_tokens || (b>a && (ids[i]<0 || ids[i]>=p->resident_experts))) return 2;
        max_m=std::max(max_m,b-a);
    }
    return max_m==p->max_m ? 0 : 3;
}
extern "C" cudaError_t m26x_device_identity(int32_t* arch,int32_t* sms) {
    if(!arch || !sms) return cudaErrorInvalidValue;
    int device=0; cudaError_t err=cudaGetDevice(&device); if(err!=cudaSuccess) return err;
    cudaDeviceProp prop{}; err=cudaGetDeviceProperties(&prop,device); if(err!=cudaSuccess) return err;
    *arch=10*prop.major+prop.minor; *sms=prop.multiProcessorCount; return cudaSuccess;
}
extern "C" cudaError_t m26x_check_aot(int32_t arch,int32_t sms,int32_t capacity,uint32_t naive) {
    if(!(naive&M26X_NAIVE_AOT_CAPACITY_IGNORED) && capacity!=M26X_CAPACITY_CLASS) return cudaErrorInvalidValue;
    int32_t live_arch=0,live_sms=0;
    cudaError_t err=m26x_device_identity(&live_arch,&live_sms); if(err!=cudaSuccess) return err;
    if(!(naive&M26X_NAIVE_AOT_MIXED_GATE) &&
       (arch!=M26X_BAKED_ARCH || sms!=M26X_BAKED_SMS || live_arch!=arch || live_sms!=sms))
        return cudaErrorInvalidDevice;
    return cudaSuccess;
}
extern "C" cudaError_t m26x_grouped_gemm_v2(const m26x_plan* p,const uint8_t* w,
    const float* x,int32_t proj,int32_t dtype,uint32_t naive,void* out,m26x_stream_t stream) {
    cudaError_t err=validate_launch(p,w,x,proj,dtype,out,naive);
    return err==cudaSuccess ? dispatch(*p,w,x,geometry(proj),dtype,naive,out,stream) : err;
}
extern "C" cudaError_t m26x_expert_ffn_v2(const m26x_plan* p,const uint8_t* w,
    const float* x,int32_t dtype,uint32_t naive,float* scratch,void* out,m26x_stream_t stream) {
    cudaError_t err=validate_launch(p,w,x,M26X_PROJ_GATE,dtype,out,naive); if(err!=cudaSuccess) return err;
    const uint64_t count=uint64_t(p->total_tokens)*512;
    const uint64_t output_bytes=uint64_t(p->total_tokens)*4096*(dtype==0?4:2);
    if(!aligned16(scratch) || p->scratch_bytes<2*count*4 || p->out_bytes<output_bytes ||
       overlaps(scratch,p->scratch_bytes,out,output_bytes) || overlaps(scratch,p->scratch_bytes,x,p->x_bytes) ||
       overlaps(scratch,p->scratch_bytes,w,p->grouped_bytes) || overlaps(x,p->x_bytes,out,output_bytes))
        return cudaErrorInvalidValue;
    if(!p->total_tokens) return cudaSuccess;
    err=dispatch(*p,w,x,geometry(0),0,naive,scratch,stream); if(err!=cudaSuccess) return err;
    err=dispatch(*p,w,x,geometry(1),0,naive,scratch+count,stream); if(err!=cudaSuccess) return err;
    activate<<<unsigned((count+255)/256),256,0,stream>>>(scratch,scratch+count,count);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    return dispatch(*p,w,scratch,geometry(2),dtype,naive,out,stream);
}
extern "C" cudaError_t m26x_unpack_matrix(const uint8_t* w,uint64_t wb,const uint8_t* s,uint64_t sb,
    int32_t rows,int32_t cols,uint32_t naive,float* out,uint64_t ob,m26x_stream_t stream) {
    if(!w || !s || !out || rows<=0 || cols<=0 || cols%32) return cudaErrorInvalidValue;
    const uint64_t count=uint64_t(rows)*cols;
    if(wb!=count/2 || sb!=count/32 || ob<count*4 || count>uint64_t(INT32_MAX)*256 ||
       overlaps(w,wb,out,ob) || overlaps(s,sb,out,ob)) return cudaErrorInvalidValue;
    unpack_kernel<<<unsigned((count+255)/256),256,0,stream>>>(w,s,rows,cols,naive,out);
    return cudaGetLastError();
}
