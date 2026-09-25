// Experimental D0: cooperative GQA-packed tensor-core split-KV decode.
// Handwritten PTX; no upstream kernel copied. Baseline entrypoints unchanged.
// D0.1 retains synchronous cooperative KV staging. The opt-in pipe below
// combines C0 accumulators with C3 single-stage/phase-disjoint shared storage.
// Q uses high/residual/tail BF16 products; P uses high/residual. Actual MMA
// work factor at QK192/V128 is (3*192 + 2*128)/320 = 2.6, not nominal 2.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "include/mimo26_attn_device.cuh"
#include "include/decode_tc_layout.h"
#include "include/decode_pipe_storage.h"
#include "include/decode_c1_model_storage.h"
#include "include/prefill_tc_storage.h"

namespace {
using BF = __nv_bfloat16;
constexpr int M = 16, N = 32, THREADS = 128;
struct __align__(16) Shared {
  BF qh[M*192], ql[M*192], qt[M*192], k[N*192], v[128*N];
  BF ph[M*N], pl[M*N];
  float score[M*N], maximum[M], sum[M], alpha[M];
  int physical[N], visible[N];
};
static_assert(sizeof(Shared) <= 48*1024, "two-CTA shared-memory budget");

__device__ __forceinline__ uint32_t pair(const BF* p, int index) {
  return *reinterpret_cast<const uint32_t*>(p + index);
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void load_a(const BF* p, int rows, int depth,
                                      uint32_t (&a)[4]) {
  int lane = threadIdx.x & 31;
#pragma unroll
  for (int r=0; r<4; ++r)
    a[r] = pair(p, m26tc::tile_index(rows, m26tc::a_row(lane,r), depth+m26tc::a_col(lane,r)));
}
__device__ __forceinline__ void load_b(const BF* p, int columns, int column,
                                      int depth, uint32_t (&b)[2]) {
  int lane = threadIdx.x & 31;
#pragma unroll
  for (int r=0; r<2; ++r)
    b[r] = pair(p, m26tc::tile_index(columns, column+m26tc::b_col(lane), depth+m26tc::b_row(lane,r)));
}
__device__ void poison(float* partials, int T, int t, int kvh, int rep, int sp, int splits) {
  int64_t count = (int64_t)T*64*splits;
  if (threadIdx.x < rep) {
    auto idx = m26tc::partial_index(t,kvh*rep+threadIdx.x,sp,splits);
    partials[idx] = partials[count+idx] = NAN;
  }
  for (int i=threadIdx.x; i<rep*128; i+=THREADS) {
    auto idx = m26tc::partial_index(t,kvh*rep+i/128,sp,splits);
    partials[2*count+idx*128+i%128] = NAN;
  }
}
__global__ void decode_tc(m26_geom g, const float* q, const uint8_t* kc,
                          const uint8_t* vc, const int32_t* pages, int page_tokens,
                          const int64_t* qpos, const int64_t* kpos, int T, int S,
                          int splits, uint32_t naive, float* partials) {
  __shared__ Shared sh;
  int tid=threadIdx.x, lane=tid&31, warp=tid/32;
  int t=blockIdx.z, kvh=blockIdx.y, sp=blockIdx.x, rep=64/g.n_kv;
  int64_t lo=(int64_t)S*sp/splits, hi=(int64_t)S*(sp+1)/splits;
  int64_t window=g.window;
  if ((naive&M26_NAIVE_GA_WINDOWED) && window<=0) window=128;
  float scale=float(m26::attn_scale(192,128,naive));
  for (int i=tid; i<M*192; i+=THREADS) {
    int row=i/192, col=i%192;
    float x=row<rep ? q[((int64_t)t*64+kvh*rep+row)*192+col] : 0;
    BF high=__float2bfloat16_rn(x);
    int dst=m26tc::tile_index(M,row,col);
    float residual=x-__bfloat162float(high);
    BF low=__float2bfloat16_rn(residual);
    sh.qh[dst]=high;
    sh.ql[dst]=low;
    sh.qt[dst]=__float2bfloat16_rn(residual-__bfloat162float(low));
  }
  if (tid<M) { sh.maximum[tid]=-INFINITY; sh.sum[tid]=0; }
  float output[4][4] = {}; // Four m16n8 fragments/warp = 32 output columns.
  __syncthreads();
  for (int64_t first=lo; first<hi; first+=N) {
    bool live=false;
    if (tid<N) {
      int64_t logical=first+tid;
      if (logical<hi) {
        int j=int(logical);
        live=m26::is_visible(qpos[t],kpos[j],window,naive);
        sh.physical[tid]=(pages && !(naive&M26_NAIVE_TC_IGNORE_PAGES))
            ? pages[j/page_tokens]*page_tokens+j%page_tokens : j;
      } else sh.physical[tid]=0;
      sh.visible[tid]=live;
    }
    // Uniform CTA decision, including empty/fully masked split tiles.
    if (!__syncthreads_or(live)) continue;
    for (int i=tid; i<N*192; i+=THREADS) {
      int row=i/192, d=i%192;
      uint8_t code=sh.visible[row] ? kc[((int64_t)sh.physical[row]*g.n_kv+kvh)*192+d] : 0;
      sh.k[m26tc::tile_index(N,row,d)]=__float2bfloat16_rn(__uint_as_float(m26tc::e4m3_bits(code)));
    }
    for (int i=tid; i<N*128; i+=THREADS) {
      int row=i/128, d=i%128;
      uint8_t code=sh.visible[row] ? vc[((int64_t)sh.physical[row]*g.n_kv+kvh)*128+d] : 0;
      sh.v[m26tc::tile_index(128,d,row)]=__float2bfloat16_rn(__uint_as_float(m26tc::e4m3_bits(code)));
    }
    __syncthreads();
    float score[4]={};
#pragma unroll
    for (int d=0; d<192; d+=16) {
      uint32_t a[4], b[2];
      load_b(sh.k,N,warp*8,d,b);
      load_a(sh.qh,M,d,a); mma(score,a,b);
      if (!(naive&M26_NAIVE_TC_DROP_Q_LOW)) {
        load_a(sh.ql,M,d,a); mma(score,a,b);
        if (!(naive&M26_NAIVE_TC_DROP_Q_TAIL)) { load_a(sh.qt,M,d,a); mma(score,a,b); }
      }
    }
    bool bad=false;
#pragma unroll
    for (int e=0; e<4; ++e) {
      int r=m26tc::c_row(lane,e), col=warp*8+m26tc::c_col(lane,e);
      float s=score[e]*scale;
      bool visible=r<rep && sh.visible[col];
      bad |= visible && !isfinite(s);
      sh.score[r*N+col]=visible ? s : -INFINITY;
    }
    if (__syncthreads_or(bad)) { poison(partials,T,t,kvh,rep,sp,splits); return; }
    // Eight consecutive lanes per Q head; all lanes execute the collectives.
    int r=tid/8, sub=tid%8;
    float scores[4], row_max=-INFINITY;
#pragma unroll
    for (int k=0;k<4;++k) { scores[k]=sh.score[r*N+sub+k*8]; row_max=fmaxf(row_max,scores[k]); }
#pragma unroll
    for (int d=4;d;d/=2) row_max=fmaxf(row_max,__shfl_xor_sync(0xffffffff,row_max,d,8));
    float old=sh.maximum[r], next=fmaxf(old,row_max);
    float alpha=(naive&M26_NAIVE_NO_RUNNING_RESCALE) ? 1.0f : (isfinite(old) ? expf(old-next) : 0.0f);
    float l=0;
#pragma unroll
    for (int k=0;k<4;++k) {
      float p=isfinite(scores[k]) ? expf(scores[k]-next) : 0;
      BF high=__float2bfloat16_rn(p);
      int dst=m26tc::tile_index(M,r,sub+k*8);
      sh.ph[dst]=high;
      sh.pl[dst]=__float2bfloat16_rn(p-__bfloat162float(high));
      l+=p;
    }
#pragma unroll
    for (int d=4;d;d/=2) l+=__shfl_xor_sync(0xffffffff,l,d,8);
    if (!sub) { sh.maximum[r]=next; sh.sum[r]=sh.sum[r]*alpha+l; sh.alpha[r]=alpha; }
    __syncthreads();
#pragma unroll
    for (int c=0;c<4;++c) {
#pragma unroll
      for (int e=0;e<4;++e) output[c][e]*=sh.alpha[m26tc::c_row(lane,e)];
#pragma unroll
      for (int k=0;k<N;k+=16) {
        uint32_t a[4], b[2];
        load_b(sh.v,128,warp*32+c*8,k,b);
        load_a(sh.ph,M,k,a); mma(output[c],a,b);
        if (!(naive&M26_NAIVE_TC_DROP_P_LOW)) { load_a(sh.pl,M,k,a); mma(output[c],a,b); }
      }
    }
    __syncthreads(); // No shared tile is overwritten before every consumer exits.
  }
  int64_t count=(int64_t)T*64*splits;
  if (tid<rep) {
    int64_t idx=m26tc::partial_index(t,kvh*rep+tid,sp,splits);
    partials[idx]=sh.maximum[tid]; partials[count+idx]=sh.sum[tid];
  }
#pragma unroll
  for (int c=0;c<4;++c) {
#pragma unroll
    for (int e=0;e<4;++e) {
      int r=m26tc::c_row(lane,e), col=warp*32+c*8+m26tc::c_col(lane,e);
      if (r<rep) {
        int64_t idx=m26tc::partial_index(t,kvh*rep+r,sp,splits);
        partials[2*count+idx*128+col]=output[c][e];
      }
    }
  }
}

// C3 keeps D0.1 intact. One raw stage plus phase-disjoint K/P storage.
using PipeShared=m26tc::CompactPipe<BF>;
static_assert(sizeof(PipeShared)==49536, "C3 shared budget incl. raw/index stage");
__device__ __forceinline__ void copy16(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t address=uint32_t(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(address), "l"(src), "r"(valid?16:0) : "memory");
}
__device__ __forceinline__ void wait_current(bool another) {
  if(another) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template<int Warps>
__device__ void prefetch(PipeShared& p,int stage,int64_t first,int64_t hi,
                         m26_geom g,int kvh,const uint8_t* kc,const uint8_t* vc,
                         const int32_t* pages,int page_tokens,uint32_t naive) {
  int tid=threadIdx.x;
  if(tid<N) {
    int j=int(first+tid);
    p.physical[stage][tid]=j<hi ? ((pages && !(naive&M26_NAIVE_TC_IGNORE_PAGES))
      ? pages[j/page_tokens]*page_tokens+j%page_tokens : j) : 0;
  }
  __syncthreads();
  // Every thread commits one group, including threads whose tail copies zero-fill.
  for(int vector=tid;vector<N*320/16;vector+=Warps*32) {
    bool key=vector<N*192/16;
    int local=key?vector:vector-N*192/16;
    int width=key?192:128,row=local/(width/16),d=(local%(width/16))*16;
    bool valid=first+row<hi;
    const uint8_t* base=key?kc:vc;
    const uint8_t* src=valid?base+((int64_t)p.physical[stage][row]*g.n_kv+kvh)*width+d:base;
    int dst=key?m26tc::raw_k(row,d):N*192+row*128+d;
    copy16(p.raw[stage]+dst,src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int Warps, bool QResidual=true>
__global__ void decode_pipe(m26_geom g,const float* q,const uint8_t* kc,const uint8_t* vc,
                            const int32_t* pages,int page_tokens,const int64_t* qpos,
                            const int64_t* kpos,int T,int S,int splits,uint32_t naive,float* partials) {
  extern __shared__ __align__(16) unsigned char memory[];
  auto& pipe=*reinterpret_cast<PipeShared*>(memory); auto& sh=pipe.math;
  constexpr int Threads=Warps*32, Frags=16/Warps;
  int tid=threadIdx.x,lane=tid&31,warp=tid/32;
  int t=blockIdx.z,kvh=blockIdx.y,sp=blockIdx.x,rep=64/g.n_kv;
  int64_t lo=(int64_t)S*sp/splits,hi=(int64_t)S*(sp+1)/splits;
  int tiles=int((hi-lo+N-1)/N);
  int64_t window=g.window;
  if((naive&M26_NAIVE_GA_WINDOWED)&&window<=0) window=128;
  float scale=float(m26::attn_scale(192,128,naive));
  for(int i=tid;i<M*192;i+=Threads) {
    int row=i/192,col=i%192;
    float x=row<rep?q[((int64_t)t*64+kvh*rep+row)*192+col]:0;
    // Input is post-RoPE FP32; native BF16-Q has exactly this one RNE cast.
    BF h=__float2bfloat16_rn(x);
    int dst=m26tc::tile_index(M,row,col);sh.qh[dst]=h;
    if constexpr(QResidual) {
      float residual=x-__bfloat162float(h);BF l=__float2bfloat16_rn(residual);
      sh.ql[dst]=l;sh.qt[dst]=__float2bfloat16_rn(residual-__bfloat162float(l));
    }
  }
  if(tid<M){sh.maximum[tid]=-INFINITY;sh.sum[tid]=0;}
  // C0: independent P term x k-group recurrences survive across tiles.
  // Each gets the same online-softmax alpha; combine only at final emission.
  float output[Frags][2][2][4]={};
  __syncthreads();
  if(tiles) prefetch<Warps>(pipe,0,lo,hi,g,kvh,kc,vc,pages,page_tokens,naive);
  for(int tile=0;tile<tiles;++tile) {
    constexpr int stage=0;int64_t first=lo+int64_t(tile)*N;
    wait_current(false);__syncthreads();
    bool live=false;
    if(tid<N){live=first+tid<hi && m26::is_visible(qpos[t],kpos[first+tid],window,naive);sh.visible[tid]=live;}
    bool any=__syncthreads_or(live);
    if(any) {
      // C0 keeps the K-store ride-along but retires F4's slower V transpose.
      // V returns to the measured D1 traversal; no free/bank-safe V claim.
      for(int i=tid;i<N*192/2;i+=Threads) {
        int row=m26tc::producer_row(i,192),d=m26tc::producer_col(i,192);
        uint16_t codes=sh.visible[row]?*reinterpret_cast<const uint16_t*>(pipe.raw[stage]+m26tc::raw_k(row,d)):0;
        *reinterpret_cast<uint32_t*>(sh.phase.k+m26tc::tile_index(N,row,d))=m26tc::e4m3x2_bf16(codes);
      }
      for(int i=tid;i<N*128/2;i+=Threads) {
        int row=(i/128)*2,d=i%128;
        uint16_t a=sh.visible[row]?pipe.raw[stage][N*192+row*128+d]:0;
        uint16_t b=sh.visible[row+1]?pipe.raw[stage][N*192+(row+1)*128+d]:0;
        *reinterpret_cast<uint32_t*>(sh.v+m26tc::tile_index(128,d,row))=m26tc::e4m3x2_bf16(a|(b<<8));
      }
    }
    __syncthreads(); // Raw readers finish before the slot is recycled.
    if(tile+1<tiles) prefetch<Warps>(pipe,stage,first+N,hi,g,kvh,kc,vc,pages,page_tokens,naive);
    if(!any) continue; // Uniform, but the async queue still advances above.
    float final_score[4]={};
    bool bad=false;
    if(warp<4) {
      constexpr int QTerms=QResidual?3:1;
      float score[QTerms][2][4]={};
#pragma unroll
      for(int d=0;d<192;d+=16) {
        int group=m26tc::q_acc_group(d/16);
        uint32_t a[4],b[2];load_b(sh.phase.k,N,warp*8,d,b);
        load_a(sh.qh,M,d,a);mma(score[0][group],a,b);
        if constexpr(QResidual) {
          if(!(naive&M26_NAIVE_TC_DROP_Q_LOW)) {
            load_a(sh.ql,M,d,a);mma(score[1][group],a,b);
            if(!(naive&M26_NAIVE_TC_DROP_Q_TAIL)){load_a(sh.qt,M,d,a);mma(score[2][group],a,b);}
          }
        }
      }
#pragma unroll
      for(int e=0;e<4;++e) {
        float total=score[0][0][e]+score[0][1][e];
        if constexpr(QResidual) {
          total+=score[1][0][e]+score[1][1][e];
          total+=score[2][0][e]+score[2][1][e];
        }
        int r=m26tc::c_row(lane,e),col=warp*8+m26tc::c_col(lane,e);
        float s=total*scale;bool visible=r<rep&&sh.visible[col];
        bad|=visible&&!isfinite(s);final_score[e]=visible?s:-INFINITY;
      }
    }
    if(__syncthreads_or(bad)) {
      wait_current(false);__syncthreads();
      if(tid<128) poison(partials,T,t,kvh,rep,sp,splits);
      return;
    }
    // The reduction barrier above retires every QK read of phase.k.
    // Only now may score/P/alpha reuse that storage. Publish score before
    // softmax reads it; the tile-end barrier retires all P/alpha readers.
    if(warp<4) {
#pragma unroll
      for(int e=0;e<4;++e)
        sh.phase.p.score[m26tc::c_row(lane,e)*N+warp*8+m26tc::c_col(lane,e)]=final_score[e];
    }
    __syncthreads();
    if(tid<128) {
      int r=tid/8,sub=tid%8;float scores[4],row_max=-INFINITY;
#pragma unroll
      for(int k=0;k<4;++k){scores[k]=sh.phase.p.score[r*N+sub+k*8];row_max=fmaxf(row_max,scores[k]);}
#pragma unroll
      for(int d=4;d;d/=2)row_max=fmaxf(row_max,__shfl_xor_sync(0xffffffff,row_max,d,8));
      float old=sh.maximum[r],next=fmaxf(old,row_max);
      float alpha=(naive&M26_NAIVE_NO_RUNNING_RESCALE)?1.f:(isfinite(old)?expf(old-next):0.f),l=0;
#pragma unroll
      for(int k=0;k<4;++k) {
        float p=isfinite(scores[k])?expf(scores[k]-next):0;BF high=__float2bfloat16_rn(p);
        int dst=m26tc::tile_index(M,r,sub+k*8);sh.phase.p.ph[dst]=high;sh.phase.p.pl[dst]=__float2bfloat16_rn(p-__bfloat162float(high));l+=p;
      }
#pragma unroll
      for(int d=4;d;d/=2)l+=__shfl_xor_sync(0xffffffff,l,d,8);
      if(!sub){sh.maximum[r]=next;sh.sum[r]=sh.sum[r]*alpha+l;sh.phase.p.alpha[r]=alpha;}
    }
    __syncthreads();
#pragma unroll
    for(int c=0;c<Frags;++c) {
#pragma unroll
      for(int e=0;e<4;++e) {
        float alpha=sh.phase.p.alpha[m26tc::c_row(lane,e)];
#pragma unroll
        for(int term=0;term<2;++term)
#pragma unroll
          for(int group=0;group<2;++group)output[c][term][group][e]*=alpha;
      }
#pragma unroll
      for(int k=0;k<N;k+=16) {
        uint32_t a[4],b[2];load_b(sh.v,128,warp*(128/Warps)+c*8,k,b);
        load_a(sh.phase.p.ph,M,k,a);mma(output[c][0][k/16],a,b);
        if(!(naive&M26_NAIVE_TC_DROP_P_LOW)){load_a(sh.phase.p.pl,M,k,a);mma(output[c][1][k/16],a,b);}
      }
    }
    __syncthreads();
  }
  wait_current(false);__syncthreads(); // Drain before any CTA exit, including empty splits.
  int64_t count=(int64_t)T*64*splits;
  if(tid<rep){int64_t idx=m26tc::partial_index(t,kvh*rep+tid,sp,splits);partials[idx]=sh.maximum[tid];partials[count+idx]=sh.sum[tid];}
#pragma unroll
  for(int c=0;c<Frags;++c) {
#pragma unroll
    for(int e=0;e<4;++e) {
      int r=m26tc::c_row(lane,e),col=warp*(128/Warps)+c*8+m26tc::c_col(lane,e);
      if(r<rep){int64_t idx=m26tc::partial_index(t,kvh*rep+r,sp,splits);
        partials[2*count+idx*128+col]=(output[c][0][0][e]+output[c][0][1][e])+(output[c][1][0][e]+output[c][1][1][e]);}
    }
  }
}

#include "include/decode_c1_impl.cuh"
#include "include/prefill_tc_impl.cuh"

__global__ void reduce_tc(m26_geom g, const float* p, const float* sink, int T,
                          int splits, uint32_t naive, float* out) {
  __shared__ float maxima[4], sums[4], values[128];
  int tid=threadIdx.x, lane=tid&31, warp=tid/32;
  int idx=blockIdx.x, h=idx%64, kvh=h/(64/g.n_kv), col=blockIdx.y*32+lane;
  int64_t count=(int64_t)T*64*splits, base=(int64_t)idx*splits;
  const float *pm=p, *pl=p+count, *po=p+2*count;
  float maximum=-INFINITY; bool bad=false;
  for (int sp=tid;sp<splits;sp+=THREADS) {
    float m=pm[base+sp], l=pl[base+sp];
    bad |= !isfinite(l) || l<0 || (l>0 && !isfinite(m)) || (l==0 && m!=-INFINITY);
    if (l>0) maximum=fmaxf(maximum,m);
  }
  if (__syncthreads_or(bad)) { if (tid<32) out[(int64_t)idx*128+col]=NAN; return; }
#pragma unroll
  for (int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  if (!lane) maxima[warp]=maximum;
  __syncthreads();
  maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
  bool sink_on=sink && (g.window>0 || (naive&M26_NAIVE_SINK_ON_GA));
  float bias=sink_on ? sink[(naive&M26_NAIVE_SINK_PER_KV)?kvh:h] : 0;
  if (sink_on) maximum=fmaxf(maximum,bias);
  float value=0, sum=0;
  for (int sp=warp;sp<splits;sp+=4) {
    float l=pl[base+sp];
    if (l==0) continue;
    float w=expf(pm[base+sp]-maximum);
    value+=w*po[(base+sp)*128+col]; sum+=w*l;
  }
  values[tid]=value; if (!lane) sums[warp]=sum;
  __syncthreads();
  if (tid<32) {
    float l=sums[0]+sums[1]+sums[2]+sums[3];
    if (sink_on) l+=expf(bias-maximum)*((naive&M26_NAIVE_SINK_PER_SPLIT)?splits:1);
    float o=values[lane]+values[32+lane]+values[64+lane]+values[96+lane];
    if (naive&M26_NAIVE_VSCALE_ON_READ) o*=0.707f;
    out[(int64_t)idx*128+col]=l>0 ? o/l : 0;
  }
}
bool geometry(const m26_geom* g, int T, int splits) {
  return g && g->n_q==64 && (g->n_kv==4 || g->n_kv==8) &&
      g->d_qk==192 && g->d_v==128 && T>0 && T<=65535 && splits>0 && splits<=65535;
}
} // namespace

extern "C" cudaError_t m26_attn_decode_splitkv_fp8_tc(
    const m26_geom* g, const float* q, const uint8_t* kc, const uint8_t* vc,
    const int32_t* pages, int32_t page_tokens, const int64_t* qpos, const int64_t* kpos,
    int32_t T, int32_t S, int32_t splits, uint32_t naive, float* partials, m26_stream_t stream) {
  if (!geometry(g,T,splits) || S<0 || !q || !kc || !vc || !qpos || !kpos || !partials ||
      (pages && page_tokens!=256)) return cudaErrorInvalidValue;
  decode_tc<<<dim3(splits,g->n_kv,T),THREADS,0,(cudaStream_t)stream>>>(
      *g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,partials);
  return cudaGetLastError();
}
template<bool QResidual>
cudaError_t pipe_config(int32_t warps,int32_t* registers,int32_t* active_ctas) {
  if((warps!=4&&warps!=8)||!registers||!active_ctas)return cudaErrorInvalidValue;
  const void* fn=warps==4?(const void*)decode_pipe<4,QResidual>:(const void*)decode_pipe<8,QResidual>;
  auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(PipeShared));
  if(e!=cudaSuccess)return e;
  cudaFuncAttributes attrs{};e=cudaFuncGetAttributes(&attrs,fn);if(e!=cudaSuccess)return e;
  *registers=attrs.numRegs;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_ctas,fn,warps*32,sizeof(PipeShared));
}
extern "C" cudaError_t m26_attn_decode_pipe_config(int32_t w,int32_t* r,int32_t* c) { return pipe_config<true>(w,r,c); }
extern "C" cudaError_t m26_attn_decode_pipe_config_bf16q(int32_t w,int32_t* r,int32_t* c) { return pipe_config<false>(w,r,c); }
template<bool QResidual>
cudaError_t launch_pipe(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  if(!geometry(g,T,splits)||S<0||!q||!kc||!vc||!qpos||!kpos||!partials||
     (pages&&page_tokens!=256)||(warps!=4&&warps!=8)||((uintptr_t(kc)|uintptr_t(vc))&15))return cudaErrorInvalidValue;
  if(warps==4)decode_pipe<4,QResidual><<<dim3(splits,g->n_kv,T),128,sizeof(PipeShared),(cudaStream_t)stream>>>(*g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,partials);
  else decode_pipe<8,QResidual><<<dim3(splits,g->n_kv,T),256,sizeof(PipeShared),(cudaStream_t)stream>>>(*g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,partials);
  return cudaGetLastError();
}
extern "C" cudaError_t m26_attn_decode_splitkv_fp8_pipe(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  return launch_pipe<true>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,warps,partials,stream);
}
extern "C" cudaError_t m26_attn_decode_splitkv_fp8_pipe_bf16q(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  return launch_pipe<false>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,warps,partials,stream);
}
// C1 is explicit opt-in. Call configuration before capture/launch, as for C3.
template<bool QResidual>
cudaError_t c1_config(int32_t warps,int32_t* registers,int32_t* active_ctas) {
  if(warps!=8||!registers||!active_ctas)return cudaErrorInvalidValue;
  const void* fn=(const void*)decode_c1<QResidual>;
  auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(C1Shared));
  if(e!=cudaSuccess)return e;
  e=cudaFuncSetAttribute(fn,cudaFuncAttributePreferredSharedMemoryCarveout,100);
  if(e!=cudaSuccess)return e;
  cudaFuncAttributes attrs{};e=cudaFuncGetAttributes(&attrs,fn);if(e!=cudaSuccess)return e;
  *registers=attrs.numRegs;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_ctas,fn,256,sizeof(C1Shared));
}
extern "C" cudaError_t m26_attn_decode_c1_config(int32_t w,int32_t* r,int32_t* c){return c1_config<true>(w,r,c);}
extern "C" cudaError_t m26_attn_decode_c1_config_bf16q(int32_t w,int32_t* r,int32_t* c){return c1_config<false>(w,r,c);}
template<bool QResidual>
cudaError_t launch_c1(const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  if(!geometry(g,T,splits)||S<0||!q||!kc||!vc||!qpos||!kpos||!partials||
     (pages&&page_tokens!=256)||warps!=8||((uintptr_t(kc)|uintptr_t(vc))&15))return cudaErrorInvalidValue;
  decode_c1<QResidual><<<dim3(splits,g->n_kv,T),256,sizeof(C1Shared),(cudaStream_t)stream>>>(
      *g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,partials);
  return cudaGetLastError();
}
extern "C" cudaError_t m26_attn_decode_splitkv_fp8_c1(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  return launch_c1<true>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,warps,partials,stream);
}
extern "C" cudaError_t m26_attn_decode_splitkv_fp8_c1_bf16q(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,int32_t splits,uint32_t naive,int32_t warps,float* partials,m26_stream_t stream) {
  return launch_c1<false>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,splits,naive,warps,partials,stream);
}
// Experimental P1: configure before standalone launch/capture. Not a default.
// Split=true is the trial-1 occupancy variant (f32q only, bitwise-identical).
template<bool QResidual,bool Split=false>
cudaError_t p1_config(int32_t* registers,int32_t* active_ctas) {
  if(!registers||!active_ctas)return cudaErrorInvalidValue;
  const void* fn=(const void*)prefill_tc<QResidual,Split>;
  constexpr size_t shared=sizeof(m26tc::PrefillStorage<BF,QResidual,Split>);
  auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
  if(e!=cudaSuccess)return e;
  e=cudaFuncSetAttribute(fn,cudaFuncAttributePreferredSharedMemoryCarveout,100);
  if(e!=cudaSuccess)return e;
  cudaFuncAttributes attrs{};e=cudaFuncGetAttributes(&attrs,fn);if(e!=cudaSuccess)return e;
  *registers=attrs.numRegs;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_ctas,fn,Split?128:256,shared);
}
extern "C" cudaError_t m26_attn_prefill_tc_config(int32_t* r,int32_t* c){return p1_config<true>(r,c);}
extern "C" cudaError_t m26_attn_prefill_tc_config_bf16q(int32_t* r,int32_t* c){return p1_config<false>(r,c);}
extern "C" cudaError_t m26_attn_prefill_tc_config_split(int32_t* r,int32_t* c){return p1_config<true,true>(r,c);}
template<bool QResidual,bool Split=false>
cudaError_t launch_p1(const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream) {
  if(!geometry(g,T,1)||S<0||!q||!kc||!vc||!qpos||!kpos||!out||
     (pages&&page_tokens!=256)||((uintptr_t(kc)|uintptr_t(vc))&15))return cudaErrorInvalidValue;
  constexpr int PM=Split?m26tc::P1M_SPLIT:m26tc::P1M;
  constexpr int NT=Split?m26tc::P1Threads_SPLIT:m26tc::P1Threads;
  int qt=PM/(64/g->n_kv);
  prefill_tc<QResidual,Split><<<dim3((T+qt-1)/qt,g->n_kv),NT,sizeof(m26tc::PrefillStorage<BF,QResidual,Split>),(cudaStream_t)stream>>>(
      *g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,naive,sink,out);
  return cudaGetLastError();
}
extern "C" cudaError_t m26_attn_prefill_fp8_tc(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream) {
  return launch_p1<true>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,naive,sink,out,stream);
}
extern "C" cudaError_t m26_attn_prefill_fp8_tc_bf16q(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream) {
  return launch_p1<false>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,naive,sink,out,stream);
}
// Trial-1 f32q split variant (P1M=32). bf16q baseline is unchanged.
extern "C" cudaError_t m26_attn_prefill_fp8_tc_split(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream) {
  return launch_p1<true,true>(g,q,kc,vc,pages,page_tokens,qpos,kpos,T,S,naive,sink,out,stream);
}
extern "C" cudaError_t m26_attn_reduce_tc(
    const m26_geom* g, const float* partials, const float* sink, int32_t T,
    int32_t splits, uint32_t naive, float* out, m26_stream_t stream) {
  if (!geometry(g,T,splits) || !partials || !out) return cudaErrorInvalidValue;
  reduce_tc<<<dim3(T*64,4),THREADS,0,(cudaStream_t)stream>>>(*g,partials,sink,T,splits,naive,out);
  return cudaGetLastError();
}
