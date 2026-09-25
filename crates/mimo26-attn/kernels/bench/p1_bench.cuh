// Paired P1 benchmark. Included in attn_bench.cu's anonymous namespace.
// Qualified attention kernels are unchanged; scalar AoS FP64 reference remains
// separate. Shared reference is legal ONLY after every Q is BF16-exact.
__global__ void p1_query_lattice_probe(const float* q,size_t n,unsigned* bad) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
    if(!bench::p1_query_bits_ok(__float_as_uint(q[i])))atomicOr(bad,1u);
  }
}
void run_p1_pair(const std::string& cell,bool proxy) {
  int s=cell=="p1-2k"?2048:cell=="p1-swa"?32768:cell=="p1-128k"?131072:cell=="p1-1m"?1048576:0;
  if(!s){fprintf(stderr,"RESULT: REFUSE unknown P1 context\n");std::exit(3);}
  constexpr int t=2048,splits=256,warmup=3,samples=7;
  const int batch=bench::p1_reference_batch(s);
  constexpr size_t count=size_t(t)*64*128,qcount=size_t(t)*64*192;
  bool swa=cell=="p1-swa";m26_geom g{64,swa?8:4,192,128,swa?128:0,1.0};
  auto start=std::chrono::steady_clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
  auto budget=[&]{if(elapsed()>480){fprintf(stderr,"RESULT: INCOMPLETE P1 pair exceeded 480s\n");std::exit(4);}reserve_check(0);};
  uint64_t bytes=bench::kv_bytes(s,g.n_kv),scratch=uint64_t(batch)*64*splits*130*sizeof(double);
  reserve_check(bytes+qcount*4+count*8+uint64_t(s+t)*8+uint64_t(s/256)*4+scratch+(uint64_t(2)<<30));
  auto* q=alloc<float>(qcount);auto* k=alloc<uint8_t>(size_t(s)*g.n_kv*192);auto* v=alloc<uint8_t>(size_t(s)*g.n_kv*128);
  auto* qp=alloc<int64_t>(t);auto* kp=alloc<int64_t>(s);auto* pages=alloc<int32_t>(s/256);
  auto* out=alloc<float>(count);auto* reference_out=alloc<float>(count);auto* partial=alloc<double>(size_t(batch)*64*splits*130);
  auto* bad=alloc<unsigned>(1);auto* sink=swa?alloc<float>(64):nullptr;
  std::vector<float> sinks(64);for(int h=0;h<64;++h)sinks[h]=(h-32)*.0625f;
  if(sink)CK(cudaMemcpy(sink,sinks.data(),64*4,cudaMemcpyHostToDevice));
  init_queries<<<256,256>>>(q,qcount);CK(cudaGetLastError());
  init_codes<<<256,256>>>(k,size_t(s)*g.n_kv*192,23);CK(cudaGetLastError());
  init_codes<<<256,256>>>(v,size_t(s)*g.n_kv*128,47);CK(cudaGetLastError());
  init_positions<<<256,256>>>(qp,t,s-t);CK(cudaGetLastError());
  init_positions<<<256,256>>>(kp,s,0);CK(cudaGetLastError());
  init_pages<<<256,256>>>(pages,s/256);CK(cudaGetLastError());
  CK(cudaMemset(bad,0,sizeof(unsigned)));
  p1_query_lattice_probe<<<256,256>>>(q,qcount,bad);CK(cudaGetLastError());
  unsigned invalid=0;CK(cudaMemcpy(&invalid,bad,sizeof(unsigned),cudaMemcpyDeviceToHost));
  if(invalid){fprintf(stderr,"RESULT: REFUSE shared P1 reference requires finite normal BF16-exact Q\n");std::exit(5);}
  printf("P1_CONTEXT cell=%s label=%s T=2048 S=%d n_q=64 n_kv=%d QK=192 V=128 window=%lld sink=%s page_tokens=256 paged=reverse-256 Q_storage=f32 Q_values=BF16-exact K=E4M3-unit V_dtype=E4M3-unit cached_V=prescaled KV_abs_max=1.875 scope=bounded-synthetic mma_m=64 mma_n=16 warmup=3 samples=7 budget_s=480 unique_KV_bytes=%llu\n",
      cell.c_str(),proxy?"PROXY":"TARGET",s,g.n_kv,(long long)g.window,swa?"per-Q-head":"absent",(unsigned long long)bytes);
  printf("P1_QUERY checked=%zu/%zu exact=PASS finite=PASS\n",qcount,qcount);
  CK(cudaMemset(reference_out,0xff,count*4));double ref_start=elapsed();
  for(int first=0;first<t;first+=batch) {
    int n=std::min(batch,t-first);budget();
    CK(cudaMemset(partial,0xff,size_t(n)*64*splits*130*sizeof(double)));
    CK(m26_attn_decode_splitkv_fp8(&g,q+size_t(first)*64*192,k,nullptr,v,nullptr,pages,256,
        qp+first,kp,n,s,splits,0,partial,nullptr));
    CK(m26_attn_reduce(&g,partial,sink,n,splits,0,reference_out+size_t(first)*64*128,nullptr));
    CK(cudaDeviceSynchronize());budget();
    if((first+batch)%256==0)printf("P1_REFERENCE_PROGRESS queries=%d/2048 elapsed_s=%.3f\n",first+batch,elapsed()-ref_start);
  }
  std::vector<float> reference(count),result(count);
  CK(cudaMemcpy(reference.data(),reference_out,count*4,cudaMemcpyDeviceToHost));
  for(float x:reference)if(!std::isfinite(x)){fprintf(stderr,"RESULT: FAIL nonfinite P1 full reference\n");std::exit(5);}
  double coordinates[3]={},reference_error=0;
  for(int c=0;c<3;++c) {
    int row=c*(t-1)/2,h=c*31,d=c*53,kh=h/(64/g.n_kv),end=s-t+row+1,begin=swa?std::max(0,end-128):0;
    double m=-INFINITY,l=0,o=0;
    auto fold=[&](double score,double value){double next=std::max(m,score),a=std::isfinite(m)?std::exp(m-next):0,b=std::exp(score-next);o=o*a+b*value;l=l*a+b;m=next;};
    for(int j=begin;j<end;++j) {
      int phys=(s/256-1-j/256)*256+j%256;uint32_t ki=(uint32_t(phys)*g.n_kv+kh)*192;double dot=0;
      for(int z=0;z<192;++z)dot+=query((row*64+h)*192+z)*decode(code(ki+z,23));
      fold(dot/std::sqrt(192.0),decode(code((uint32_t(phys)*g.n_kv+kh)*128+d,47)));
    }
    if(swa)fold(sinks[h],0);
    coordinates[c]=l>0?o/l:0;
    if(!std::isfinite(coordinates[c])){fprintf(stderr,"RESULT: FAIL nonfinite independent P1 coordinate\n");std::exit(5);}
    reference_error=std::max(reference_error,std::abs(double(reference[(size_t(row)*64+h)*128+d])-coordinates[c]));
  }
  if(reference_error>2e-5){fprintf(stderr,"RESULT: FAIL P1 reference coordinate mismatch\n");std::exit(5);}
  budget();
  printf("P1_REFERENCE outputs=%zu/%zu finite=PASS baseline=scalar-f64-splitkv-reduce slab_queries=%d splits=256 reuse=identical-BF16-exact-inputs elapsed_ms=%.3f coordinates=3/3 max_coordinate_diff=%.9g\n",count,count,batch,(elapsed()-ref_start)*1000,reference_error);
  auto launch=[&](bool native){auto fn=native?m26_attn_prefill_fp8_tc_bf16q:m26_attn_prefill_fp8_tc;CK(fn(&g,q,k,v,pages,256,qp,kp,t,s,0,sink,out,nullptr));};
  // Verify BOTH complete output tensors before emitting ANY timing sample.
  for(bool native:{false,true}) {
    auto config=native?m26_attn_prefill_tc_config_bf16q:m26_attn_prefill_tc_config;
    int regs=0,capacity=0;CK(config(&regs,&capacity));
    if(regs>128||capacity!=(native?2:1)){fprintf(stderr,"RESULT: REFUSE P1 capacity/register drift\n");std::exit(3);}
    CK(cudaMemset(out,0xff,count*4));launch(native);
    CK(cudaMemcpy(result.data(),out,count*4,cudaMemcpyDeviceToHost));
    auto checked=bench::p1_full_check(result,reference,count);
    if(!checked.ok){fprintf(stderr,"RESULT: FAIL P1 full comparison mode=%s index=%zu diff=%.9g\n",native?"bf16q":"f32q",checked.index,checked.max_error);std::exit(5);}
    double worst=checked.max_error,coord=0;
    for(int c=0;c<3;++c)coord=std::max(coord,std::abs(double(result[(size_t(c*(t-1)/2)*64+c*31)*128+c*53])-coordinates[c]));
    if(coord>2e-5){fprintf(stderr,"RESULT: FAIL P1 candidate coordinates\n");std::exit(5);}
    printf("P1_CORRECT mode=%s reference=%s checked=%zu/%zu finite=PASS coordinates=3/3 max_baseline_diff=%.9g max_coordinate_diff=%.9g registers=%d shared_bytes=%d capacity_ctas=%d\n",native?"bf16q":"f32q",native?"bf16q-lattice-local":"f32q",count,count,worst,coord,regs,native?38976:88128,capacity);
    budget();
  }
  for(bool native:{false,true}) {
    for(int i=0;i<warmup;++i){launch(native);CK(cudaDeviceSynchronize());budget();}
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<samples;++i) {
      budget();CK(cudaEventRecord(a));launch(native);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
      float ms=0;CK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);
      printf("P1_SAMPLE mode=%s index=%d ms=%.6f\n",native?"bf16q":"f32q",i,ms);
    }
    double ms=bench::median(times),useful=bench::flops(t,s,int(g.window)),executed=bench::p1_mma_flops(t,s,g.n_kv,int(g.window),!native);
    double utf=bench::tflops(useful,ms),etf=bench::tflops(executed,ms),target=native?100:125.7;
    if(!proxy&&etf>209.5){fprintf(stderr,"RESULT: REFUSE P1 exceeds nominal dense BF16 peak; accounting review required\n");std::exit(5);}
    printf("P1_METRIC mode=%s median_ms=%.6f min_ms=%.6f max_ms=%.6f useful_flops=%.0f executed_mma_flops=%.0f mma_work_factor=%.9f useful_TFLOPS=%.9f executed_TFLOPS=%.9f target_domain=%s target_TFLOPS=%.1f verdict=%s\n",native?"bf16q":"f32q",ms,*std::min_element(times.begin(),times.end()),*std::max_element(times.begin(),times.end()),useful,executed,executed/useful,utf,etf,native?"useful":"executed",target,(native?utf:etf)>=target?"PASS":"MISS");
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));budget();
  }
  printf("P1_COMPLETE modes=2 checked_per_mode=%zu samples_per_mode=7\n",count);
  for(void* p:allocations)CK(cudaFree(p));allocations.clear();
  puts("RESULT: PASS P1 pair harness (performance verdicts separate, no promotion)");
}
