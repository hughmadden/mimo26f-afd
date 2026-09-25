/* Copyright (c) 2025 by the b12x authors. Apache-2.0; LICENSE.b1-compute.
 * http://www.apache.org/licenses/LICENSE-2.0
 * Distributed AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * Approved REUSE row 118: w4a8_v41_slice.py:48-256,290-364 at
 * 3882b935ede761d6c73a5d6fd68e690f1e3f5380, SHA256
 * 4fbaacaeea6791dcb03b41a2b4e2a99725f98ede8c4c09e67aa4e350068145d7.
 * Changed: handwritten CUDA, H4096/I512, FP32 boundaries, stable activation,
 * no clamps/BF16/route weights/atomics. New conforming exceptional-scale arm.
 * Compute-only routines: caller owns validated metadata, tile staging/barriers,
 * quantizer v1 and ordered slice/route reduction. No production launch yet.
 */
#pragma once
#include "mxfp4_ptx.cuh"
#include "mimo26_expert_device.cuh"
namespace m26b1 {
struct ActivationView {
    const uint32_t* payload; // Four E4M3FN bytes per word; validated extent.
    const uint8_t* scales; // K32 bytes, row-major, not packed scale registers.
    int word_stride, scale_stride;
    int low, high; // Validated physical input row, or -1 for a padded row.
};
struct AFragment {uint32_t a0,a1,a2,a3,scale;};
__device__ __forceinline__ AFragment load_a(const ActivationView& x,int k32,int c) {
    AFragment a{0,0,0,0,0};
    if(x.low>=0){a.a0=x.payload[x.low*x.word_stride+k32*8+c*2];a.a2=x.payload[x.low*x.word_stride+k32*8+c*2+1];}
    if(x.high>=0){a.a1=x.payload[x.high*x.word_stride+k32*8+c*2];a.a3=x.payload[x.high*x.word_stride+k32*8+c*2+1];}
    const int row=c==0?x.low:c==1?x.high:-1;
    if(row>=0){const auto s=x.scales[row*x.scale_stride+k32];a.scale=s>=105 && s<=247?s:255;}
    return a;
}
// Decoder only, not an encoder. Exact E4M3FN values, then a power-of-two scale.
// Invalid codes/scales propagate NaN for the caller's numerical-fault check.
__device__ __forceinline__ float activation_value(const ActivationView& x,int row,int k) {
    if(row<0)return 0.0f;
    const unsigned b=(x.payload[row*x.word_stride+k/4]>>(8*(k%4)))&255;
    const unsigned s=x.scales[row*x.scale_stride+k/32];
    if((b&127)==127 || s<105 || s>247)return __uint_as_float(0x7fc00000u);
    const int e=(b>>3)&15,m=b&7;
    float v=ldexpf(float(e?8+m:m),int(s)-127+(e?e-10:-9));
    return b&128?-v:v;
}
// Collective over all 32 lanes. Ordinary scales use the released MMA.
// 0/1 can decode to FP32 subnormals; 253..255 can require finite saturation
// (255 also differs from native UE8M0 NaN). Decode those weights before FMA.
template<int ByteB>
__device__ __forceinline__ void dot32(float (&d)[4],const ActivationView& x,
        int k32,const AFragment& a,uint32_t packed,uint32_t sb) {
    const unsigned scale=(sb>>(8*ByteB))&255;
    if(__all_sync(0xffffffffu,scale>=2 && scale<=252)){
        const auto b=e2m1_containers(packed);
        mma_e4m3_e2m1<0,ByteB>(d,a.a0,a.a1,a.a2,a.a3,b.x,b.y,a.scale,sb);
    }else{
        const int c=threadIdx.x%4;
        #pragma unroll
        for(int e=0;e<4;++e){
            const int n=2*c+e%2,row=e<2?x.low:x.high;
            const auto sw=__shfl_sync(0xffffffffu,sb,n*4);
            #pragma unroll
            for(int k=0;k<32;++k){
                // Every lane participates, including padded A rows.
                const auto w=__shfl_sync(0xffffffffu,packed,n*4+k/8);
                const float av=activation_value(x,row,k32*32+k);
                const float bv=m26x::decode((w>>(4*(k%8)))&15,(sw>>(8*ByteB))&255,0);
                d[e]=fmaf(av,bv,d[e]);
            }
        }
    }
}
// One staged K128 step. Up precedes gate in the prepared w13 tile.
// Repeated exactly 32 times by the H4096 caller; accumulators start at zero.
template<int Width>
__device__ __forceinline__ void fc1_k128(float (&gate)[Width/32][4],float (&up)[Width/32][4],
        const ActivationView& x,int kt,const uint32_t* b,const uint32_t* sf){
    static_assert(Width==64 || Width==128,"baseline width");
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,c=lane%4;
    #pragma unroll
    for(int kb=0;kb<4;++kb){
        const auto a=load_a(x,kt*4+kb,c);
        #pragma unroll
        for(int nf=0;nf<Width/32;++nf){
            const int i=((kb*(Width/32)+nf)*32+lane)*4+warp;
            const auto su=sf[nf*32+warp*8+g],sg=sf[Width+nf*32+warp*8+g];
            #define M26B1_STEP(K) dot32<K>(gate[nf],x,kt*4+kb,a,b[i+Width*16],sg); dot32<K>(up[nf],x,kt*4+kb,a,b[i],su)
            if(kb==0){M26B1_STEP(0);}else if(kb==1){M26B1_STEP(1);}else if(kb==2){M26B1_STEP(2);}else{M26B1_STEP(3);}
            #undef M26B1_STEP
        }
    }
}
__device__ __forceinline__ float silu_up(float g,float u){
    const float e=expf(g>=0?-g:g);
    const float s=g>=0?g/(1.0f+e):(g*e)/(1.0f+e);
    return s*u;
}
// All 16 rows are defined, with zero padding. No intermediate quantization here.
// Caller supplies non-overlapping 16-row storage, stride >= Width. For full-rank
// output use stride 512 and offset pointers by the channel slice; group scratch
// must be padded per M16 group, not indexed by a dense unpadded route_base.
template<int Width>
__device__ __forceinline__ void fc1_finish(const float (&gate)[Width/32][4],const float (&up)[Width/32][4],
        int rows,float* mid,float* gate_out=nullptr,float* up_out=nullptr,int output_stride=Width){
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,c=lane%4;
    #pragma unroll
    for(int nf=0;nf<Width/32;++nf){
        #pragma unroll
        for(int e=0;e<4;++e){
            const int row=g+(e/2)*8,col=nf*32+warp*8+2*c+e%2,index=row*output_stride+col;
            const float gv=row<rows?gate[nf][e]:0,uv=row<rows?up[nf][e]:0;
            mid[index]=row<rows?silu_up(gv,uv):0;
            if(gate_out)gate_out[index]=gv;
            if(up_out)up_out[index]=uv;
        }
    }
}
// Compile-time FC2 operand policy. Only E-W4A8-v1 is implemented. A future
// BF16 policy can supply its own view/fragments and two K16 MMAs per K32 step,
// while retaining these packed FP4/scales tiles and conforming special handling.
// The caller must pair the policy with the matching intermediate conversion;
// fc1_finish deliberately leaves the intermediate FP32. No runtime/M dispatch.
struct Fc2Fp8 {
    using View=ActivationView;
    using Fragment=AFragment;
    __device__ __forceinline__ static Fragment load(const View& x,int kb,int c){
        return load_a(x,kb,c);
    }
    template<int ByteB>
    __device__ __forceinline__ static void step(float (&acc)[4],const View& x,int kb,
            const Fragment& a,uint32_t packed,uint32_t scale){
        dot32<ByteB>(acc,x,kb,a,packed,scale);
    }
};
// Update one N128 output tile over K=Width. Initialize acc only before the
// first K slice; retain it across all 512/Width slices for a complete route.
// Offset the activation view's payload/scales by the slice's K start, retaining
// row strides. Caller stages 32 N tiles for H4096. No route weighting here;
// The lead's route reducer requires complete unweighted FC2 results, not planes.
template<int Width,typename OperandPolicy=Fc2Fp8>
__device__ __forceinline__ void fc2_slice(float (&acc)[4][4],const typename OperandPolicy::View& x,
        const uint32_t* b,const uint32_t* sf){
    static_assert(Width==64 || Width==128,"baseline width");
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,c=lane%4;
    #pragma unroll
    for(int kb=0;kb<Width/32;++kb){
        const auto a=OperandPolicy::load(x,kb,c);
        #pragma unroll
        for(int nf=0;nf<4;++nf){
            const int i=((kb*4+nf)*32+lane)*4+warp;
            const auto sw=sf[(kb/4)*128+nf*32+warp*8+g];
            if(kb%4==0)OperandPolicy::template step<0>(acc[nf],x,kb,a,b[i],sw);
            else if(kb%4==1)OperandPolicy::template step<1>(acc[nf],x,kb,a,b[i],sw);
            else if(kb%4==2)OperandPolicy::template step<2>(acc[nf],x,kb,a,b[i],sw);
            else OperandPolicy::template step<3>(acc[nf],x,kb,a,b[i],sw);
        }
    }
}
} // namespace m26b1
