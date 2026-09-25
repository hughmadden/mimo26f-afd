// E-W4A8-v1 compute schedule. The lead's storage/plan/quantizer/reducer stay separate.
// Caller validates allocation extents and immutable plan/pool lifetime; consumes
// outputs only after ALL plan, compute, quantizer and reducer faults are checked.
#pragma once
#include "mlp_staging.cuh"
#include "group_plan.cuh"
// grouped.cu must be included first (also used by the independent proof driver).
namespace m26b1 {
template<int Width,bool AsyncScales=false>
__global__ void connected_fc1(Pool pool,const uint8_t* prepared,const int32_t* slots,
        PlanStorage plan,const uint32_t* payload,const uint8_t* scales,
        float* mid,float* gates,float* ups,uint32_t* faults){
    static_assert(!AsyncScales || Width==128,"packed scales require Width128");
    const uint32_t gi=blockIdx.y,slice=blockIdx.x,tid=threadIdx.x;
    const uint32_t fi=gi*(512/Width)+slice;
    if(tid==0)faults[fi]=0;
    if(*plan.fault){if(tid==0)faults[fi]=1;return;}
    const Group group=plan.groups[gi];
    if(group.rows<1 || group.rows>16 || group.expert<0 || group.expert>=256 || slots[group.expert]<0){if(tid==0)faults[fi]=2;return;}
    __shared__ __align__(16) uint32_t b[Width*32],sf[Width*2];
    const int g=(tid%32)/4;
    ActivationView x{payload,scales,1024,128,group.input_rows[g],group.input_rows[g+8]};
    float gate[Width/32][4]={},up[Width/32][4]={};
    for(int kt=0;kt<32;++kt){
        Tile t{uint32_t(slots[group.expert]),slice*Width,uint32_t(kt*128),Width,false,false,true};
        const bool a=stage_mlp<AsyncScales>(pool,t,prepared,reinterpret_cast<uint8_t*>(b),reinterpret_cast<uint8_t*>(sf),tid,128);
        t.gate=true;
        const bool c=stage_mlp<AsyncScales>(pool,t,prepared,reinterpret_cast<uint8_t*>(b+Width*16),reinterpret_cast<uint8_t*>(sf+Width),tid,128);
        // Validated metadata makes both statuses block-uniform.
        if(!a || !c){if(tid==0)faults[fi]=3;return;}
        stage_commit_wait();
        fc1_k128<Width>(gate,up,x,kt,b,sf);
        __syncthreads(); // No producer may overwrite a tile still being consumed.
    }
    const size_t offset=size_t(gi)*16*512+slice*Width;
    fc1_finish<Width>(gate,up,group.rows,mid+offset,gates?gates+offset:nullptr,ups?ups+offset:nullptr,512);
}
// FC2 output element store: FP32 as-is (E-W4A8-v1), or BF16 RNE (the Spark's
// opt-in MIMO26_B1_Y=bf16 route outputs, perf reset P8).
__device__ __forceinline__ void fc2_store(float* p,float v){*p=v;}
__device__ __forceinline__ void fc2_store(uint16_t* p,float v){
    const uint32_t b=__float_as_uint(v);
    *p=uint16_t((b+0x7fffu+((b>>16)&1u))>>16);
}
// Policy: Fc2Fp8 (dot32, with the exceptional-scale arm) or, for a layer whose
// scales were scanned normal at prepare time, the Spark's Fc2Fp8Fast. OutT:
// float (the unit's FP32 boundary) or uint16_t (BF16 bits).
template<int Width,bool AsyncScales=false,typename Policy=Fc2Fp8,typename OutT=float>
__global__ void connected_fc2_fp8(Pool pool,const uint8_t* prepared,const int32_t* slots,
        PlanStorage plan,const uint8_t* payload,const uint8_t* scales,const uint32_t* quant_faults,
        OutT* output,uint32_t routes,uint32_t* faults,const float* weights,unsigned naive){
    static_assert(!AsyncScales || Width==128,"packed scales require Width128");
    const uint32_t gi=blockIdx.y,ot=blockIdx.x,tid=threadIdx.x,fi=gi*32+ot;
    if(tid==0)faults[fi]=0;
    if(*plan.fault){if(tid==0)faults[fi]=1;return;}
    const Group group=plan.groups[gi];
    if(group.rows<1 || group.rows>16 || group.expert<0 || group.expert>=256 || slots[group.expert]<0){if(tid==0)faults[fi]=2;return;}
    if(__syncthreads_or(quant_faults[gi*256+tid] || quant_faults[gi*256+tid+128])){if(tid==0)faults[fi]=4;return;}
    __shared__ __align__(16) uint32_t b[Width*16],sf[128];
    const int lane=tid%32,warp=tid/32,g=lane/4,c=lane%4;
    float acc[4][4]={};
    for(int slice=0;slice<512/Width;++slice){
        Tile t{uint32_t(slots[group.expert]),ot*128,uint32_t(slice*Width),Width,true,false,true};
        if(!stage_mlp<AsyncScales>(pool,t,prepared,reinterpret_cast<uint8_t*>(b),reinterpret_cast<uint8_t*>(sf),tid,128)){if(tid==0)faults[fi]=3;return;}
        stage_commit_wait();
        ActivationView x{reinterpret_cast<const uint32_t*>(payload)+size_t(gi)*16*128+slice*(Width/4),
            scales+size_t(gi)*256+slice*(Width/32),128,16,g<group.rows?g:-1,g+8<group.rows?g+8:-1};
        fc2_slice<Width,Policy>(acc,x,b,sf);
        __syncthreads();
    }
    const int lo=g<group.rows?plan.original_routes[group.route_base+g]:0;
    const int hi=g+8<group.rows?plan.original_routes[group.route_base+g+8]:0;
    if(__syncthreads_or(lo<0 || uint32_t(lo)>=routes || hi<0 || uint32_t(hi)>=routes)){if(tid==0)faults[fi]=5;return;}
    for(int nf=0;nf<4;++nf)for(int e=0;e<4;++e){
        const int row=g+(e/2)*8,col=ot*128+nf*32+warp*8+2*c+e%2;
        if(row<group.rows){const int original=e/2?hi:lo;
            fc2_store(output+size_t(original)*4096+col,naive==1?acc[nf][e]*weights[original]:acc[nf][e]);}
    }
}
} // namespace m26b1
