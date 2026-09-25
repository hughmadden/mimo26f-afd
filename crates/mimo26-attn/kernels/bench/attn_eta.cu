// N5 diagnostic only: resident/interior QK -> softmax -> PV tiles, not P1.
// Original handwritten MMA/packing; no third-party kernel copied.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/decode_tc_layout.h"
#include "metrics.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define EC(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
  fprintf(stderr,"RESULT: FAIL eta CUDA %s: %s\n",#call,cudaGetErrorString(e)); exit(2); } } while(0)
namespace {
using BF=__nv_bfloat16;
__device__ __forceinline__ void mma(float (&d)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void la(const BF* p,int rows,int row,int depth,uint32_t (&a)[4]) {
  int lane=threadIdx.x&31;
#pragma unroll
  for(int r=0;r<4;++r) a[r]=*reinterpret_cast<const uint32_t*>(p+m26tc::tile_index(rows,row+m26tc::a_row(lane,r),depth+m26tc::a_col(lane,r)));
}
__device__ __forceinline__ void lb(const BF* p,int rows,int col,int depth,uint32_t (&b)[2]) {
  int lane=threadIdx.x&31;
#pragma unroll
  for(int r=0;r<2;++r) b[r]=*reinterpret_cast<const uint32_t*>(p+m26tc::tile_index(rows,col+m26tc::b_col(lane),depth+m26tc::b_row(lane,r)));
}
template<int M,int N> struct __align__(16) Shared {
  BF q[M*192], k[N*192], v[128*N], p[M*N];
  float partial[2*M], maximum[M], denominator[M];
  uint8_t raw[N*320];
};
// Two warps per m16 strip. Each warp owns N/2 score columns and 64 PV columns.
// One Q plane is reused for all correction passes; no fictitious 3-plane fit.
template<int M,int N,bool Residual,bool Instrument>
__global__ void eta_kernel(const float* q,const uint8_t* k,const uint8_t* v,
                           int iterations,unsigned naive,float* out,unsigned long long* clocks) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto& sh=*reinterpret_cast<Shared<M,N>*>(storage);
  constexpr int Threads=M*4, Frags=N/16;
  int tid=threadIdx.x,lane=tid&31,warp=tid/32;
  int mr=(warp/2)*16,npart=warp%2,nc=npart*(N/2);
  unsigned long long accumulated[4]={},stamp=0;
  float output[8][4];
  for(int it=0;it<iterations;++it) {
    __syncthreads();
    if constexpr(Instrument) { if(!tid) stamp=clock64(); }
    for(int i=tid;i<N*320;i+=Threads) sh.raw[i]=i<N*192?k[i]:v[i-N*192];
    __syncthreads();
    for(int i=tid;i<N*192;i+=Threads)
      sh.k[m26tc::tile_index(N,i/192,i%192)]=__float2bfloat16_rn(__uint_as_float(m26tc::e4m3_bits(sh.raw[i])));
    for(int i=tid;i<N*128;i+=Threads)
      sh.v[m26tc::tile_index(128,i%128,i/128)]=__float2bfloat16_rn(__uint_as_float(m26tc::e4m3_bits(sh.raw[N*192+i])));
    __syncthreads();
    if constexpr(Instrument) { if(!tid) { auto now=clock64(); accumulated[0]+=now-stamp; stamp=now; } }
    float score[Frags][4]={};
#pragma unroll
    for(int term=0;term<(Residual?3:1);++term) {
      if(term && (naive&1)) continue;
      for(int i=tid;i<M*192;i+=Threads) {
        float x=q[i]; BF h=__float2bfloat16_rn(x);
        float residual=x-__bfloat162float(h); BF l=__float2bfloat16_rn(residual);
        BF value=term==0?h:term==1?l:__float2bfloat16_rn(residual-__bfloat162float(l));
        sh.q[m26tc::tile_index(M,i/192,i%192)]=value;
      }
      __syncthreads();
#pragma unroll
      for(int d=0;d<192;d+=16) {
        uint32_t a[4]; la(sh.q,M,mr,d,a);
#pragma unroll
        for(int f=0;f<Frags;++f) { uint32_t b[2]; lb(sh.k,N,nc+8*f,d,b); mma(score[f],a,b); }
      }
      __syncthreads();
    }
    if constexpr(Instrument) { if(!tid) { auto now=clock64(); accumulated[1]+=now-stamp; stamp=now; } }
    float maximum[2]={-INFINITY,-INFINITY};
#pragma unroll
    for(int f=0;f<Frags;++f) {
#pragma unroll
      for(int e=0;e<4;++e) { score[f][e]*=0.07216878364870322f; maximum[e/2]=fmaxf(maximum[e/2],score[f][e]); }
    }
#pragma unroll
    for(int r=0;r<2;++r) {
      maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffff,maximum[r],2,4));
      maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffff,maximum[r],1,4));
      if(!(lane&3)) sh.partial[(mr+lane/4+r*8)*2+npart]=maximum[r];
    }
    __syncthreads();
    if(tid<M) sh.maximum[tid]=fmaxf(sh.partial[2*tid],sh.partial[2*tid+1]);
    __syncthreads();
    float sum[2]={};
#pragma unroll
    for(int f=0;f<Frags;++f) {
#pragma unroll
      for(int e=0;e<4;++e) {
        int row=mr+m26tc::c_row(lane,e);
        score[f][e]=expf(score[f][e]-sh.maximum[row]);
        sum[e/2]+=score[f][e];
      }
    }
#pragma unroll
    for(int r=0;r<2;++r) {
      sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],2,4);
      sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],1,4);
      if(!(lane&3)) sh.partial[(mr+lane/4+r*8)*2+npart]=sum[r];
    }
    __syncthreads();
    if(tid<M) sh.denominator[tid]=sh.partial[2*tid]+sh.partial[2*tid+1];
#pragma unroll
    for(int f=0;f<Frags;++f) {
#pragma unroll
      for(int e=0;e<4;++e) {
        int row=mr+m26tc::c_row(lane,e),col=nc+f*8+m26tc::c_col(lane,e);
        sh.p[m26tc::tile_index(M,row,col)]=__float2bfloat16_rn(score[f][e]);
      }
    }
    __syncthreads();
    if constexpr(Instrument) { if(!tid) { auto now=clock64(); accumulated[2]+=now-stamp; stamp=now; } }
#pragma unroll
    for(int c=0;c<8;++c) {
#pragma unroll
      for(int e=0;e<4;++e) output[c][e]=0;
    }
#pragma unroll
    for(int term=0;term<2;++term) {
      if(term && (naive&2)) continue;
      if(term) {
#pragma unroll
        for(int f=0;f<Frags;++f) {
#pragma unroll
          for(int e=0;e<4;++e) {
            int row=mr+m26tc::c_row(lane,e),col=nc+f*8+m26tc::c_col(lane,e);
            float p=score[f][e];
            sh.p[m26tc::tile_index(M,row,col)]=__float2bfloat16_rn(p-__bfloat162float(__float2bfloat16_rn(p)));
          }
        }
        __syncthreads();
      }
#pragma unroll
      for(int d=0;d<N;d+=16) {
        uint32_t a[4]; la(sh.p,M,mr,d,a);
#pragma unroll
        for(int c=0;c<8;++c) { uint32_t b[2]; lb(sh.v,128,npart*64+c*8,d,b); mma(output[c],a,b); }
      }
      __syncthreads();
    }
    if constexpr(Instrument) { if(!tid) { auto now=clock64(); accumulated[3]+=now-stamp; } }
  }
#pragma unroll
  for(int c=0;c<8;++c) {
#pragma unroll
    for(int e=0;e<4;++e) {
      int row=mr+m26tc::c_row(lane,e),col=npart*64+c*8+m26tc::c_col(lane,e);
      out[(size_t(blockIdx.x)*M+row)*128+col]=output[c][e]/sh.denominator[row];
    }
  }
  if constexpr(Instrument) { if(!tid) for(int i=0;i<4;++i) clocks[size_t(blockIdx.x)*4+i]=accumulated[i]; }
}
uint32_t hash(uint32_t x) { x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16); }
float round_bf16(float x) { uint32_t u; memcpy(&u,&x,4);u=(u+0x7fff+((u>>16)&1))&0xffff0000;memcpy(&x,&u,4);return x; }
double value(uint8_t c) { return ((c&128)?-1:1)*std::ldexp(1.0+(c&7)/8.0,((c>>3)&15)-7); }
template<class T> T* allocate(size_t n) { T* p;EC(cudaMalloc(&p,n*sizeof(T)));return p; }
void reserve() { size_t f,t; EC(cudaMemGetInfo(&f,&t)); if(f<bench::reserve_bytes) {fprintf(stderr,"RESULT: REFUSE eta reserve\n");exit(3);} }
template<int M,int N,bool Residual> void run(int sms) {
  constexpr int Threads=M*4; constexpr size_t SharedBytes=sizeof(Shared<M,N>);
  static_assert(SharedBytes==bench::eta_shared_bytes(M,N),"host/shared budget mismatch");
  int blocks=sms*4, iterations=32;
  std::vector<float> q(M*192);
  std::vector<uint8_t> k(N*192),v(N*128);
  for(size_t i=0;i<q.size();++i) q[i]=(int(hash(i+11)&0x1ffff)-65536)*0x1p-17f;
  // Host-generated post-RoPE FP32 operands. This is not a rotary qualification.
  int rep=M==64?16:8;
  for(int r=0;r<M;++r) for(int d=0;d<32;++d) {
    double angle=(127+r/rep)/std::pow(M==64?1e7:1e4,2.0*d/64);
    float a=q[r*192+d],b=q[r*192+d+32],c=float(std::cos(angle)),s=float(std::sin(angle));
    q[r*192+d]=a*c-b*s;q[r*192+d+32]=a*s+b*c;
  }
  for(size_t i=0;i<k.size();++i) {uint32_t h=hash(i+23);k[i]=uint8_t(0x20+h%32)|uint8_t((h>>8)&128);}
  for(size_t i=0;i<v.size();++i) {uint32_t h=hash(i+47);v[i]=uint8_t(0x20+h%27)|uint8_t((h>>8)&128);}
  size_t free,total;EC(cudaMemGetInfo(&free,&total));
  size_t needed=q.size()*4+k.size()+v.size()+size_t(blocks)*M*128*4+size_t(blocks)*4*8+(uint64_t(2)<<30);
  if(!bench::memory_fits(free,needed)) {fprintf(stderr,"RESULT: REFUSE eta allocation reserve\n");exit(3);}
  float* dq=allocate<float>(q.size()); auto* dk=allocate<uint8_t>(k.size()); auto* dv=allocate<uint8_t>(v.size());
  float* out=allocate<float>(size_t(blocks)*M*128);auto* clocks=allocate<unsigned long long>(size_t(blocks)*4);
  EC(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));EC(cudaMemcpy(dk,k.data(),k.size(),cudaMemcpyHostToDevice));EC(cudaMemcpy(dv,v.data(),v.size(),cudaMemcpyHostToDevice));
  EC(cudaFuncSetAttribute(eta_kernel<M,N,Residual,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  EC(cudaFuncSetAttribute(eta_kernel<M,N,Residual,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  int active;EC(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,eta_kernel<M,N,Residual,false>,Threads,SharedBytes));
  cudaFuncAttributes attr;EC(cudaFuncGetAttributes(&attr,eta_kernel<M,N,Residual,false>));
  auto launch=[&](bool instrument,unsigned naive=0) {
    if(instrument) eta_kernel<M,N,Residual,true><<<blocks,Threads,SharedBytes>>>(dq,dk,dv,naive?1:iterations,naive,out,clocks);
    else eta_kernel<M,N,Residual,false><<<blocks,Threads,SharedBytes>>>(dq,dk,dv,naive?1:iterations,naive,out,clocks);
    EC(cudaGetLastError());
  };
  // Independent full CPU FP64 reference, for the explicitly selected Q lattice.
  std::vector<double> ref(M*128);
  for(int r=0;r<M;++r) {
    double score[N],mx=-INFINITY,den=0;
    for(int j=0;j<N;++j) {double dot=0;for(int d=0;d<192;++d) dot+=(Residual?q[r*192+d]:round_bf16(q[r*192+d]))*value(k[j*192+d]);score[j]=dot/std::sqrt(192.0);mx=std::max(mx,score[j]);}
    for(int j=0;j<N;++j) {score[j]=std::exp(score[j]-mx);den+=score[j];}
    for(int d=0;d<128;++d) {double o=0;for(int j=0;j<N;++j)o+=score[j]*value(v[j*128+d]);ref[r*128+d]=o/den;}
  }
  double worst=0;
  for(bool instrument:{false,true}) {
    EC(cudaMemset(out,0xff,size_t(blocks)*M*128*4));launch(instrument);EC(cudaDeviceSynchronize());
    std::vector<float> result(size_t(blocks)*M*128);EC(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<result.size();++i) {double e=std::abs(double(result[i])-ref[i%(M*128)]);if(!std::isfinite(result[i])||e>2e-5){fprintf(stderr,"RESULT: FAIL eta oracle M=%d N=%d residual=%d instrument=%d index=%zu diff=%.9g\n",M,N,Residual,instrument,i,e);exit(5);}worst=std::max(worst,e);}
  }
  for(unsigned naive:{1u,2u}) {
    if(naive==1 && !Residual) continue;
    launch(false,naive);EC(cudaDeviceSynchronize());
    std::vector<float> bad(size_t(blocks)*M*128);EC(cudaMemcpy(bad.data(),out,bad.size()*4,cudaMemcpyDeviceToHost));
    double maximum_error=0;
    for(size_t i=0;i<bad.size();++i) {
      if(!std::isfinite(bad[i])) {fprintf(stderr,"RESULT: FAIL eta negative nonfinite output\n");exit(5);}
      maximum_error=std::max(maximum_error,std::abs(double(bad[i])-ref[i%(M*128)]));
    }
    if(maximum_error<=2e-5) {fprintf(stderr,"RESULT: FAIL eta negative did not distinguish flag=%u\n",naive);exit(5);}
    printf("ETA_NEGATIVE M=%d N=%d mode=%s flag=%u max_abs=%.9g detected=PASS\n",M,N,Residual?"f32q":"bf16q",naive,maximum_error);
  }
  std::vector<unsigned long long> cycles(size_t(blocks)*4);EC(cudaMemcpy(cycles.data(),clocks,cycles.size()*8,cudaMemcpyDeviceToHost));
  double phase[4]={};for(int b=0;b<blocks;++b)for(int p=0;p<4;++p)phase[p]+=double(cycles[b*4+p])/blocks/iterations;
  for(int i=0;i<3;++i){launch(false);EC(cudaDeviceSynchronize());}
  cudaEvent_t a,b;EC(cudaEventCreate(&a));EC(cudaEventCreate(&b));std::vector<float> times;
  for(int i=0;i<7;++i){reserve();EC(cudaEventRecord(a));launch(false);EC(cudaEventRecord(b));EC(cudaEventSynchronize(b));float ms;EC(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);printf("ETA_SAMPLE M=%d N=%d mode=%s index=%d ms=%.6f\n",M,N,Residual?"f32q":"bf16q",i,ms);}
  double ms=bench::median(times),useful=2.0*M*N*320*blocks*iterations,executed=bench::eta_mma_flops(M,N,Residual)*blocks*iterations;
  double rate=bench::tflops(executed,ms),us=bench::tflops(useful,ms),sum=phase[0]+phase[1]+phase[2]+phase[3];
  printf("ETA M=%d N=%d warps=%d mode=%s Q_round=%s K=E4M3-unit V=E4M3-unit cache=FP8-prescaled-V K_abs_max=1.875 V_abs_max=1.25 stats=f32 blocks=%d iterations=%d shared_bytes=%zu registers=%d active_CTAs_per_SM=%d median_ms=%.6f useful_flops=%.0f executed_flops=%.0f useful_TFLOPS=%.6f executed_TFLOPS=%.6f eta=%.6f load_cycles=%.3f qk_pack_cycles=%.3f softmax_hi_cycles=%.3f pv_lowpack_cycles=%.3f softmax_hi_fraction=%.6f max_abs=%.9g verdict=%s scope=interior-resident-micro-not-prefill\n",
      M,N,Threads/32,Residual?"f32q":"bf16q",Residual?"Q3":"BF16-RNE-post-RoPE-once",blocks,iterations,SharedBytes,attr.numRegs,active,ms,useful,executed,us,rate,rate/209.5,phase[0],phase[1],phase[2],phase[3],phase[2]/sum,worst,(Residual?rate>=125.7:us>=100)?"PASS":"MISS");
  if(rate>209.5) {fprintf(stderr,"RESULT: FAIL eta above assumed peak; review accounting\n");exit(5);}
  EC(cudaEventDestroy(a));EC(cudaEventDestroy(b));EC(cudaFree(dq));EC(cudaFree(dk));EC(cudaFree(dv));EC(cudaFree(out));EC(cudaFree(clocks));reserve();
}
}
int m26_attn_eta_micro(int sms) {
  puts("ETA_SCOPE synthetic post-RoPE Q; full FP64 tile comparison for both instrumented/uninstrumented variants; BF16-Q excluded from f32q parity; clocks are CTA means, not additive wall time; P-low packing is charged to PV; no inter-tile online rescale, mask/sink boundary or streaming-DRAM qualification");
  run<64,64,true>(sms);run<64,64,false>(sms);
  run<32,32,true>(sms);run<32,32,false>(sms);
  run<32,64,true>(sms);run<32,64,false>(sms);
  puts("RESULT: PASS eta micro harness (performance verdicts per row)");return 0;
}
