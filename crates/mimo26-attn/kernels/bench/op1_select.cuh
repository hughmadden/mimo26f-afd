// OP1 selection controls only. Retained attention kernels are unchanged.
// Included in attn_bench.cu's namespace; generated data comes from frozen D7.
#include "op1_generated.h"

__global__ void op1_init_q(float* q, size_t n, int prefix, unsigned seed) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    q[i]=query(uint32_t(i)+uint32_t(prefix)*64*192+seed);
}
__global__ void op1_init_kv(uint8_t* p,size_t n,int width,int nk,int pages,int first,int s,unsigned seed) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
    int row=int(i/(nk*width)), logical=(pages-1-row/256)*256+row%256;
    unsigned offset=unsigned(i%(nk*width));
    p[i]=logical<s?code(unsigned(first+logical)*nk*width+offset,seed):0x7f;
  }
}

struct Op1Buffers {
  m26_geom g;
  int t,s,first,prefix,pg,layer;
  size_t nq,no;
  float *q,*out_base,*out,*ref,*tc,*sink;
  double* part;
  uint8_t *k,*v;
  int64_t *qp,*kp;
  int32_t* pages;
  unsigned* bad;
  std::vector<float> sinks;
  static constexpr int guard=64; // 256-byte prefix preserves allocator alignment.
  Op1Buffers(const op1::Case& c,bool swa,int layer_id):
      g{64,swa?8:4,192,128,swa?128:0,1.0},t(c.t),s(swa?c.swa_s:c.s),
      first(swa?c.swa_start:0),prefix(c.prefix),pg((s+255)/256),layer(layer_id),
      nq(size_t(t)*64*192),no(size_t(t)*64*128),sinks(64) {
    q=alloc<float>(nq);k=alloc<uint8_t>(size_t(pg)*256*g.n_kv*192);
    v=alloc<uint8_t>(size_t(pg)*256*g.n_kv*128);
    qp=alloc<int64_t>(t);kp=alloc<int64_t>(s);pages=alloc<int32_t>(pg);
    out_base=alloc<float>(no+2*guard);out=out_base+guard;
    ref=alloc<float>(no);tc=alloc<float>(size_t(t)*64*8*130);
    part=alloc<double>(size_t(t)*64*16*130);bad=alloc<unsigned>(1);
    sink=swa?alloc<float>(64):nullptr;
    for(int h=0;h<64;++h)sinks[h]=(h-32)*.0625f + layer*.00390625f;
    if(sink)CK(cudaMemcpy(sink,sinks.data(),64*4,cudaMemcpyHostToDevice));
    op1_init_q<<<64,256>>>(q,nq,prefix,qseed());CK(cudaGetLastError());
    op1_init_kv<<<64,256>>>(k,size_t(pg)*256*g.n_kv*192,192,g.n_kv,pg,first,s,kseed());CK(cudaGetLastError());
    op1_init_kv<<<64,256>>>(v,size_t(pg)*256*g.n_kv*128,128,g.n_kv,pg,first,s,vseed());CK(cudaGetLastError());
    init_positions<<<4,256>>>(qp,t,prefix);CK(cudaGetLastError());
    init_positions<<<4,256>>>(kp,s,first);CK(cudaGetLastError());
    init_pages<<<1,256>>>(pages,pg);CK(cudaGetLastError());
    std::vector<float> guards(no+2*guard,19.25f);
    CK(cudaMemcpy(out_base,guards.data(),guards.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemset(bad,0,4));p1_query_lattice_probe<<<64,256>>>(q,nq,bad);CK(cudaGetLastError());
    unsigned invalid=0;CK(cudaMemcpy(&invalid,bad,4,cudaMemcpyDeviceToHost));
    if(invalid){fprintf(stderr,"RESULT: FAIL OP1 query lattice proof\n");std::exit(5);}
  }
  unsigned qseed() const{return unsigned(layer)*104729u;}
  unsigned kseed() const{return 23u+unsigned(layer)*104729u;}
  unsigned vseed() const{return 47u+unsigned(layer)*104729u;}
  void reset(cudaStream_t stream=nullptr) {CK(cudaMemsetAsync(out,0xff,no*4,stream));}
  void launch(bool native,int path,cudaStream_t stream=nullptr) {
    if(!path) {
      auto fn=native?m26_attn_prefill_fp8_tc_bf16q:m26_attn_prefill_fp8_tc;
      CK(fn(&g,q,k,v,pages,256,qp,kp,t,s,0,sink,out,stream));
    } else {
      int splits=1<<(path-1);
      auto fn=native?m26_attn_decode_splitkv_fp8_pipe_bf16q:m26_attn_decode_splitkv_fp8_pipe;
      CK(fn(&g,q,k,v,pages,256,qp,kp,t,s,splits,0,8,tc,stream));
      CK(m26_attn_reduce_tc(&g,tc,sink,t,splits,0,out,stream));
    }
  }
  std::vector<float> reference(double& coordinate_error) {
    CK(cudaMemset(ref,0xff,no*4));
    CK(cudaMemset(part,0xff,size_t(t)*64*16*130*sizeof(double)));
    CK(m26_attn_decode_splitkv_fp8(&g,q,k,nullptr,v,nullptr,pages,256,qp,kp,t,s,16,0,part,nullptr));
    CK(m26_attn_reduce(&g,part,sink,t,16,0,ref,nullptr));
    std::vector<float> expected(no);CK(cudaMemcpy(expected.data(),ref,no*4,cudaMemcpyDeviceToHost));
    for(float x:expected)if(!std::isfinite(x)){fprintf(stderr,"RESULT: FAIL OP1 reference finite\n");std::exit(5);}
    coordinate_error=0;
    for(int c=0;c<3;++c) {
      int row=c*(t-1)/2,h=c*31,d=c*53,kh=h/(64/g.n_kv),end=prefix+row;
      int begin=g.window?std::max(first,end-127):first;
      double m=-INFINITY,l=0,o=0;
      auto fold=[&](double score,double value){double next=std::max(m,score),a=std::isfinite(m)?std::exp(m-next):0,b=std::exp(score-next);o=o*a+b*value;l=l*a+b;m=next;};
      for(int j=begin;j<=end;++j) {
        double dot=0;
        for(int z=0;z<192;++z)dot+=query(uint32_t(((prefix+row)*64+h)*192+z)+qseed())*
            decode(code(uint32_t((j*g.n_kv+kh)*192+z),kseed()));
        fold(dot/std::sqrt(192.0),decode(code(uint32_t((j*g.n_kv+kh)*128+d),vseed())));
      }
      if(sink)fold(sinks[h],0);
      double value=o/l;
      if(!std::isfinite(value)){fprintf(stderr,"RESULT: FAIL OP1 independent coordinate finite\n");std::exit(5);}
      coordinate_error=std::max(coordinate_error,std::abs(double(expected[(size_t(row)*64+h)*128+d])-value));
    }
    if(coordinate_error>2e-5){fprintf(stderr,"RESULT: FAIL OP1 independent coordinates\n");std::exit(5);}
    return expected;
  }
  double check(const std::vector<float>& expected) {
    std::vector<float> all(no+2*guard);CK(cudaMemcpy(all.data(),out_base,all.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<guard;++i)if(all[i]!=19.25f||all[guard+no+i]!=19.25f){fprintf(stderr,"RESULT: FAIL OP1 output guard\n");std::exit(5);}
    std::vector<float> result(all.begin()+guard,all.end()-guard);
    auto checked=bench::p1_full_check(result,expected,no);
    if(!checked.ok){fprintf(stderr,"RESULT: FAIL OP1 output index=%zu error=%.9g\n",checked.index,checked.max_error);std::exit(5);}
    return checked.max_error;
  }
};
void op1_configure() {
  for(int native=0;native<2;++native)for(int path=0;path<2;++path) {
    int regs=0,cap=0;
    if(!path){auto f=native?m26_attn_prefill_tc_config_bf16q:m26_attn_prefill_tc_config;CK(f(&regs,&cap));}
    else {auto f=native?m26_attn_decode_pipe_config_bf16q:m26_attn_decode_pipe_config;CK(f(8,&regs,&cap));}
    if(regs!=(path?(native?70:92):(native?124:125))||cap!=(path?2:(native?2:1))) {
      fprintf(stderr,"RESULT: REFUSE OP1 resource drift\n");std::exit(3);
    }
    printf("OP1_RESOURCE mode=%s family=%s registers=%d shared=%d capacity=%d\n",native?"bf16q":"f32q",path?"c3":"p1",regs,path?49536:(native?38976:88128),cap);
  }
}
void run_op1_select(bool proxy) {
  if(proxy){fprintf(stderr,"RESULT: REFUSE OP1 coordinator only\n");std::exit(3);}
  constexpr int repeats=16;
  auto start=std::chrono::steady_clock::now();
  auto budget=[&]{if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>480){fprintf(stderr,"RESULT: INCOMPLETE OP1 selection 480s budget\n");std::exit(4);}};
  reserve_check(uint64_t(2)<<30);op1_configure();
  printf("OP1_SELECT_BEGIN d7_sha=%s cases=36 kinds=2 modes=2 paths=5 repeats=16 warmup=3 samples=7 scope=selection-only-context-proxy gate=UNSET Q=post-RoPE-f32 Q_values=BF16-exact KV=E4M3-unit V=prescaled page=reverse-256 boundary=attention-core\n",op1::d7_sha);
  cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
  int count=0;
  for(const auto& c:op1::cases)if(c.select)for(int swa=0;swa<2;++swa) {
    budget();reserve_check(uint64_t(1)<<30);
    Op1Buffers data(c,swa,swa?1:0);double coord=0;auto expected=data.reference(coord);
    printf("OP1_SELECT_CASE id=%d category=%s step=%d kind=%s T=%d S=%d prefix=%d first=%d query_checked=%zu reference_checked=%zu query_exact=PASS reference_finite=PASS coordinates=3 max_coordinate_diff=%.9g\n",c.id,op1::categories[c.category],c.step,swa?"swa":"ga",data.t,data.s,data.prefix,data.first,data.nq,data.no,coord);
    // All ten complete checks precede any timing in this shape.
    for(int native=0;native<2;++native)for(int path=0;path<5;++path) {
      data.reset();CK(cudaMemset(data.tc,0xff,size_t(data.t)*64*8*130*4));
      data.launch(native,path);double error=data.check(expected);
      printf("OP1_SELECT_CORRECT id=%d kind=%s mode=%s path=%d checked=%zu guard=128 max_error=%.9g\n",c.id,swa?"swa":"ga",native?"bf16q":"f32q",path,data.no,error);
    }
    // Reverse the path order on alternating shapes; precision samples interleave.
    for(int slot=0;slot<5;++slot) {
      int path=(count&1)?4-slot:slot;
      cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
      for(int native=0;native<2;++native) {
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int r=0;r<repeats;++r)data.launch(native,path,stream);
        CK(cudaStreamEndCapture(stream,&graphs[native]));
        size_t nodes=0;CK(cudaGraphGetNodes(graphs[native],nullptr,&nodes));
        if(nodes!=size_t(repeats*(path?2:1))){fprintf(stderr,"RESULT: FAIL OP1 graph node count\n");std::exit(5);}
        printf("OP1_SELECT_GRAPH id=%d kind=%s mode=%s path=%d nodes=%zu\n",c.id,swa?"swa":"ga",native?"bf16q":"f32q",path,nodes);
        CK(cudaGraphInstantiate(&execs[native],graphs[native],nullptr,nullptr,0));
        for(int r=0;r<3;++r)CK(cudaGraphLaunch(execs[native],stream));
        CK(cudaStreamSynchronize(stream));data.check(expected);
      }
      for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
        int native=order^(sample&1);budget();data.reset(stream);
        CK(cudaEventRecord(a,stream));CK(cudaGraphLaunch(execs[native],stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));
        float ms=0;CK(cudaEventElapsedTime(&ms,a,b));double error=data.check(expected);
        printf("OP1_SELECT_SAMPLE id=%d kind=%s mode=%s path=%d index=%d graph_ms=%.9f per_call_ms=%.9f checked=%zu guard=128 max_error=%.9g\n",c.id,swa?"swa":"ga",native?"bf16q":"f32q",path,sample,ms,ms/repeats,data.no,error);
      }
      for(int native=0;native<2;++native){CK(cudaGraphExecDestroy(execs[native]));CK(cudaGraphDestroy(graphs[native]));}
    }
    for(void* p:allocations)CK(cudaFree(p));allocations.clear();++count;
  }
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaStreamDestroy(stream));reserve_check(0);
  printf("OP1_SELECT_COMPLETE shape_kinds=%d candidates=720 samples=5040 all_layer_step_measured=0\n",count);
  puts("RESULT: PASS OP1 selection harness; no operating-point gate or promotion");
}
