// P1 trial-1 throwaway probe (attn-lead). NOT a permanent gate, NOT promoted.
// Run on the coordinator (sm_120) only, compiled standalone with attn_decode_tc.cu.
// Two arms: (1) bitwise split==baseline; (2) 8K/32K/64K tok/s for split vs
// baseline f32q. The split (P1M=32, 128 threads, grid.x doubled) must be
// bitwise-identical to the baseline (P1M=64): the T-split only doubles grid.x;
// each query row's S-axis online softmax stays inside one CTA.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include "mimo26_attn_kernels.h"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);} } while(0)

static void* dev_upload(const void* h,size_t n){void* d;CK(cudaMalloc(&d,n));CK(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice));return d;}

static uint32_t g_seed=0x1234567u;
static uint32_t rnd(){g_seed^=g_seed<<13;g_seed^=g_seed>>17;g_seed^=g_seed<<5;return g_seed;}

// Non-NaN E4M3 code (0x00..0x7c) for the timing arm.
static uint8_t valid_code(){return uint8_t(rnd()%125);}
// Arbitrary byte for the bitwise arm (includes NaN codes 0x7f/0xff to exercise
// the poison path deterministically).
static uint8_t raw_byte(){return uint8_t(rnd()&0xff);}

struct Geom { int T,nkv,S; uint32_t naive; };

static int bitwise_arm() {
  int cases=0;size_t outputs=0;
  std::vector<Geom> geoms;
  for(int nkv:{4,8})for(int T:{1,2,3,4,5,8,9,17,33,65})geoms.push_back({T,nkv,273,0});
  for(int nkv:{4,8})geoms.push_back({65,nkv,4096,0});
  geoms.push_back({9,8,1024,M26_NAIVE_SINK_ON_GA});
  geoms.push_back({9,8,1024,M26_NAIVE_GA_WINDOWED});
  geoms.push_back({9,4,1024,M26_NAIVE_SINK_PER_KV});
  geoms.push_back({9,4,1024,M26_NAIVE_NO_RUNNING_RESCALE});
  for(const Geom& g:geoms) {
    m26_geom geom{64,g.nkv,192,128,g.nkv==8?128:0,1.0};
    int np=std::max(1,(g.S+255)/256),physical=np*256;
    size_t qc=size_t(g.T)*64*192,count=size_t(g.T)*64*128;
    std::vector<float> q(qc),sink(64),a(count),b(count);
    std::vector<uint8_t> k(size_t(physical)*g.nkv*192),v(size_t(physical)*g.nkv*128);
    std::vector<int64_t> qp(g.T),kp(std::max(g.S,1));std::vector<int32_t> pages(np);
    for(int p=0;p<np;++p)pages[p]=np-1-p;
    for(int h=0;h<64;++h)sink[h]=float(std::log(double(1+h%7)));
    for(int t=0;t<g.T;++t)qp[t]=1000+t;
    for(int j=0;j<g.S;++j)kp[j]=1000+j;
    for(size_t i=0;i<q.size();++i)q[i]=float(int(rnd()%2001)-1000)/32.f;
    for(size_t i=0;i<k.size();++i)k[i]=raw_byte();
    for(size_t i=0;i<v.size();++i)v[i]=raw_byte();
    auto* dq=(float*)dev_upload(q.data(),q.size()*4);
    auto* dk=(uint8_t*)dev_upload(k.data(),k.size());auto* dv=(uint8_t*)dev_upload(v.data(),v.size());
    auto* dqp=(int64_t*)dev_upload(qp.data(),qp.size()*8);auto* dkp=(int64_t*)dev_upload(kp.data(),kp.size()*8);
    auto* dp=(int32_t*)dev_upload(pages.data(),pages.size()*4);auto* ds=(float*)dev_upload(sink.data(),sink.size()*4);
    auto* da=(float*)dev_upload(a.data(),a.size()*4);auto* db=(float*)dev_upload(b.data(),b.size()*4);
    CK(m26_attn_prefill_fp8_tc(&geom,dq,dk,dv,dp,256,dqp,dkp,g.T,g.S,g.naive,ds,da,nullptr));
    CK(m26_attn_prefill_fp8_tc_split(&geom,dq,dk,dv,dp,256,dqp,dkp,g.T,g.S,g.naive,ds,db,nullptr));
    CK(cudaMemcpy(a.data(),da,a.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(),db,b.size()*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(a.data(),b.data(),a.size()*4)!=0) {
      size_t at=0;for(;at<a.size()&&a[at]==b[at];++at){}
      std::fprintf(stderr,"RESULT: FAIL P1_SPLIT_BITWISE T=%d nkv=%d S=%d naive=%u first=%zu base=%a split=%a\n",g.T,g.nkv,g.S,g.naive,at,a[at],b[at]);
      return 1;
    }
    outputs+=count;++cases;
    for(void* p:std::vector<void*>{db,da,ds,dp,dkp,dqp,dv,dk,dq})CK(cudaFree(p));
  }
  std::printf("P1_SPLIT_BITWISE PASS cases=%d outputs=%zu baseline=f32q split=f32q\n",cases,outputs);
  return 0;
}

static double median(std::vector<float> v){std::sort(v.begin(),v.end());return v.size()%2?double(v[v.size()/2]):0.5*(v[v.size()/2-1]+v[v.size()/2]);}

static void time_point(int S,int nkv) {
  m26_geom geom{64,nkv,192,128,nkv==8?128:0,1.0};
  constexpr int T=2048,WARMUP=3,SAMPLES=7;
  int np=std::max(1,(S+255)/256),physical=np*256;
  size_t qc=size_t(T)*64*192,count=size_t(T)*64*128;
  std::vector<float> q(qc),sink(64),out(count);
  std::vector<uint8_t> k(size_t(physical)*nkv*192),v(size_t(physical)*nkv*128);
  std::vector<int64_t> qp(T),kp(S);std::vector<int32_t> pages(np);
  for(int p=0;p<np;++p)pages[p]=np-1-p;
  for(int h=0;h<64;++h)sink[h]=float(std::log(double(1+h%7)));
  for(int t=0;t<T;++t)qp[t]=1000+t;
  for(int j=0;j<S;++j)kp[j]=1000+j;
  for(size_t i=0;i<q.size();++i)q[i]=float(int(rnd()%2001)-1000)/32.f;
  for(size_t i=0;i<k.size();++i)k[i]=valid_code();
  for(size_t i=0;i<v.size();++i)v[i]=valid_code();
  auto* dq=(float*)dev_upload(q.data(),q.size()*4);
  auto* dk=(uint8_t*)dev_upload(k.data(),k.size());auto* dv=(uint8_t*)dev_upload(v.data(),v.size());
  auto* dqp=(int64_t*)dev_upload(qp.data(),qp.size()*8);auto* dkp=(int64_t*)dev_upload(kp.data(),kp.size()*8);
  auto* dp=(int32_t*)dev_upload(pages.data(),pages.size()*4);auto* ds=(float*)dev_upload(sink.data(),sink.size()*4);
  auto* dout=(float*)dev_upload(out.data(),out.size()*4);
  int base_regs=0,base_ctas=0,split_regs=0,split_ctas=0;
  CK(m26_attn_prefill_tc_config(&base_regs,&base_ctas));
  CK(m26_attn_prefill_tc_config_split(&split_regs,&split_ctas));
  std::printf("P1_SPLIT_RESOURCE S=%d nkv=%d base_regs=%d base_ctas=%d split_regs=%d split_ctas=%d\n",S,nkv,base_regs,base_ctas,split_regs,split_ctas);
  auto run=[&](bool is_split){
    for(int i=0;i<WARMUP;++i){if(is_split)CK(m26_attn_prefill_fp8_tc_split(&geom,dq,dk,dv,dp,256,dqp,dkp,T,S,0,ds,dout,nullptr));else CK(m26_attn_prefill_fp8_tc(&geom,dq,dk,dv,dp,256,dqp,dkp,T,S,0,ds,dout,nullptr));CK(cudaDeviceSynchronize());}
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<SAMPLES;++i){
      CK(cudaEventRecord(a));
      if(is_split)CK(m26_attn_prefill_fp8_tc_split(&geom,dq,dk,dv,dp,256,dqp,dkp,T,S,0,ds,dout,nullptr));
      else CK(m26_attn_prefill_fp8_tc(&geom,dq,dk,dv,dp,256,dqp,dkp,T,S,0,ds,dout,nullptr));
      CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
      float ms=0;CK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);
      std::printf("P1_SPLIT_SAMPLE S=%d nkv=%d mode=%s index=%d ms=%.6f\n",S,nkv,is_split?"split":"base",i,ms);
    }
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
    double ms=median(times);
    std::printf("P1_SPLIT_METRIC S=%d nkv=%d mode=%s median_ms=%.6f min_ms=%.6f max_ms=%.6f tok_s=%.9f\n",S,nkv,is_split?"split":"base",ms,*std::min_element(times.begin(),times.end()),*std::max_element(times.begin(),times.end()),S/(ms*1e-3));
    return ms;
  };
  double base_ms=run(false),split_ms=run(true);
  std::printf("P1_SPLIT_RATIO S=%d nkv=%d base_tok_s=%.9f split_tok_s=%.9f speedup=%.6f\n",S,nkv,S/(base_ms*1e-3),S/(split_ms*1e-3),base_ms/split_ms);
  for(void* p:std::vector<void*>{dout,ds,dp,dkp,dqp,dv,dk,dq})CK(cudaFree(p));
}

int main(int,char**) {
  cudaDeviceProp props{};CK(cudaGetDeviceProperties(&props,0));
  int arch=props.major*10+props.minor;
  std::printf("P1_SPLIT_HOST gpu=%s arch=sm_%d sms=%d\n",props.name,arch,props.multiProcessorCount);
  if(arch!=120){std::fprintf(stderr,"RESULT: REFUSE trial requires sm_120 (got sm_%d)\n",arch);return 3;}
  { int r=0,c=0; CK(m26_attn_prefill_tc_config(&r,&c)); CK(m26_attn_prefill_tc_config_split(&r,&c)); }
  if(bitwise_arm()!=0){std::fprintf(stderr,"RESULT: FAIL bitwise mismatch stops the trial\n");return 2;}
  for(int S:{8192,32768,65536})for(int nkv:{4,8})time_point(S,nkv);
  std::printf("RESULT: PASS P1 split trial (bitwise identical; timing above, no promotion)\n");
  return 0;
}
