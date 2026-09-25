// Frozen B1 math bandwidth cell. Correctness before timing; no tuning loop.
#include <cuda_profiler_api.h>
namespace {
void bench_connected_host(){
    proof::bench_selftest();const auto ids=proof::bench_experts(256);constexpr int cap=2048;
    std::vector<uint8_t> resident(256,1);std::vector<int32_t> inv(cap),original(cap),route_ids(cap);std::vector<float> weights(cap,1),gw(cap);std::vector<m26b1::Group> groups(cap);uint32_t count=0,fault=0;
    for(int m=1;m<=8;++m){const int rows=32*m;
        for(int t=0;t<rows;++t)for(int s=0;s<8;++s)route_ids[t*8+s]=ids[t/m*8+s];
        need(m26b1::plan_host({256,uint32_t(rows),route_ids.data(),weights.data(),uint64_t(rows)*8,resident.data(),256},{inv.data(),original.data(),gw.data(),groups.data(),&count,&fault,cap,cap})==m26b1::Status::ok && count==256 && !fault,"benchmark host plan");
        for(int s=0;s<256;++s){const auto& g=groups[ids[s]];need(g.expert==ids[s] && g.rows==m,"benchmark group shape");
            for(int j=0;j<m;++j){const int route=(s/8*m+j)*8+s%8;need(g.input_rows[j]==s/8*m+j && original[g.route_base+j]==route && inv[route]==g.route_base+j,"benchmark original-route/golden axes");}}
    }
    std::puts("HOST PASS B1 benchmark plan: all256 experts, M1..8, original-route golden axes, fixed thresholds");
}
struct B1Event {
    cudaEvent_t value=nullptr;
    B1Event(){ck(cudaEventCreate(&value),"B1 event create");}
    ~B1Event(){if(value && cudaEventDestroy(value)!=cudaSuccess)cleanup_failed=true;}
    B1Event(const B1Event&)=delete;B1Event& operator=(const B1Event&)=delete;
};
int bench_connected(const std::string& root,const std::string& input,int profile_m=0){
    need(profile_m==0 || profile_m==1 || profile_m==8,"B1 profile M must be 1 or 8");
    gpu_ready();const auto source=proof::read_json(input+"/b1-bench-source.json");
    const auto index=proof::read_json(root+"/model.safetensors.index.json");
    const auto ids=proof::bench_experts(256);
    const auto xp=proof::read(input+"/b1-x-payload.u8"),xs=proof::read(input+"/b1-x-scales.u8");
    const auto raw=proof::read(input+"/b1-bench-partial.f64");
    need(xp.size()==8*4096 && xs.size()==8*128 && raw.size()==size_t(256)*8*4096*8,"B1 benchmark reference extents");
    for(const char* file:{"b1-x-payload.u8","b1-x-scales.u8","b1-bench-partial.f64"}){
        const auto bytes=proof::read(input+"/"+file);need(proof::sha256(bytes)==source.at("artifacts").at(file).at("sha256").str(),"B1 benchmark artifact hash");}
    std::vector<double> golden(raw.size()/8);std::memcpy(golden.data(),raw.data(),raw.size());
    for(double v:golden)need(std::isfinite(v),"B1 benchmark nonfinite reference");
    const size_t weight_bytes=size_t(256)*m26b1::image_bytes;
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"B1 benchmark properties");
    need(prop.major*10+prop.minor==121 && prop.multiProcessorCount==48 && prop.l2CacheSize>0 && weight_bytes>size_t(prop.l2CacheSize)*2,"GB10/L2 benchmark guard");
    Buffer canonical(m26b1::image_bytes),prepared(weight_bytes),dslots(256*4),resident(256);
    std::vector<int32_t> slots(256);std::vector<uint8_t> present(256,1);m26b1::PreparedInfo info{};
    for(int s=0;s<256;++s){slots[ids[s]]=s;const auto image=proof::bench_image(root,index,source,ids[s]);canonical.upload(image.data());
        ck(m26b1::prepare_async({2,4096,512,0,m26b1::image_bytes},ptr<uint8_t>(canonical),canonical.bytes,ptr<uint8_t>(prepared)+size_t(s)*m26b1::image_bytes,m26b1::image_bytes,&info,0),"benchmark prepare");ck(cudaDeviceSynchronize(),"prepare reuse");}
    dslots.upload(slots.data());resident.upload(present.data());const m26b1::Pool pool{info,prepared.bytes,256};
    constexpr int cap=256*8,ng=256;const size_t mid_n=size_t(ng)*16*512;
    if(profile_m)std::printf("PROFILE CONTRACT family=B1 M=%d residents=256 bytes=%zu warmups=10 captured_launches=1; no event bandwidth samples\n",profile_m,weight_bytes);
    else std::printf("B1 BENCH CONTRACT layer=1 rank=0 residents=256 unique_weight_scale_bytes=%zu l2_bytes=%d lattice=E-W4A8-v1 warmups=10 samples=31 timed_scope=FC1_SiLU_quantizer_FC2 excludes=prepare_plan_route_reduce_upload metric=effective_unique_weight_GBps clocks=unlocked\n",weight_bytes,prop.l2CacheSize);
    int verdict=0;
    for(int m=1;m<=8;++m){
        if(profile_m && m!=profile_m)continue;
        const int rows=32*m,routes=rows*8;
        std::vector<uint8_t> x(rows*4096),sc(rows*128);std::vector<int32_t> route_ids(routes);std::vector<float> weights(routes);
        for(int t=0;t<rows;++t){std::copy_n(xp.data()+(t%m)*4096,4096,x.data()+t*4096);std::copy_n(xs.data()+(t%m)*128,128,sc.data()+t*128);
            for(int s=0;s<8;++s){route_ids[t*8+s]=ids[(t/m)*8+s];weights[t*8+s]=float(s+1)/36;}}
        Buffer dx(x.size()),ds(sc.size()),dw(routes*4),di(routes*4),inv(cap*4),original(cap*4),gw(cap*4),groups(cap*sizeof(m26b1::Group)),count(4),fault(4);
        dx.upload(x.data());ds.upload(sc.data());dw.upload(weights.data());di.upload(route_ids.data());
        m26b1::PlanStorage ps{ptr<int32_t>(inv),ptr<int32_t>(original),ptr<float>(gw),ptr<m26b1::Group>(groups),ptr<uint32_t>(count),ptr<uint32_t>(fault),cap,cap};
        ck(m26b1::plan_async({256,uint32_t(rows),ptr<int32_t>(di),ptr<float>(dw),uint64_t(routes),ptr<uint8_t>(resident),256},ps,0),"benchmark plan");
        ck(cudaDeviceSynchronize(),"benchmark plan sync");zero_faults(fault,"benchmark plan fault");uint32_t got_groups=0;count.download(&got_groups);need(got_groups==ng,"benchmark groups");
        Buffer dh(mid_n*4),dq(mid_n),dqs(mid_n/32),qfault(mid_n/32*4),f1(ng*4*4),f2(ng*32*4),dy(size_t(routes)*4096*4);
        auto launch=[&](unsigned naive){
            m26b1::connected_fc1<128><<<dim3(4,ng),128>>>(pool,ptr<uint8_t>(prepared),ptr<int32_t>(dslots),ps,ptr<uint32_t>(dx),ptr<uint8_t>(ds),ptr<float>(dh),nullptr,nullptr,ptr<uint32_t>(f1));ck(cudaGetLastError(),"benchmark FC1");
            ck(m26b1::quantize_async({ptr<float>(dh),mid_n},{ptr<uint8_t>(dq),ptr<uint8_t>(dqs),ptr<uint32_t>(qfault),mid_n,mid_n/32,mid_n/32},0),"benchmark quantizer");
            m26b1::connected_fc2_fp8<128><<<dim3(32,ng),128>>>(pool,ptr<uint8_t>(prepared),ptr<int32_t>(dslots),ps,ptr<uint8_t>(dq),ptr<uint8_t>(dqs),ptr<uint32_t>(qfault),ptr<float>(dy),routes,ptr<uint32_t>(f2),ptr<float>(dw),naive);ck(cudaGetLastError(),"benchmark FC2");
        };
        auto validate=[&](){
            ck(cudaDeviceSynchronize(),"benchmark completion");zero_faults(f1,"benchmark FC1 fault");zero_faults(qfault,"benchmark quantizer fault");zero_faults(f2,"benchmark FC2 fault");
            std::vector<float> y(size_t(routes)*4096);dy.download(y.data());size_t bad=0;double max_abs=0;
            for(int s=0;s<256;++s)for(int j=0;j<m;++j)for(int n=0;n<4096;++n){
                const size_t at=(size_t(s/8)*m*8+j*8+s%8)*4096+n;const double want=golden[(size_t(s)*8+j)*4096+n];
                const double err=std::abs(double(y[at])-want);bad+=!std::isfinite(y[at]) || err>1e-5+1e-5*std::abs(want);max_abs=std::max(max_abs,err);}
            for(Buffer* b:{&prepared,&dslots,&resident,&dx,&ds,&dw,&di,&inv,&original,&gw,&groups,&count,&fault,&dh,&dq,&dqs,&dy})b->guard();
            std::printf("B1 BENCH CHECK M=%d bad=%zu coordinates=%zu maxabs=%.12g\n",m,bad,y.size(),max_abs);return bad;
        };
        launch(0);if(validate())throw Numerical("B1 benchmark real reference mismatch; no timing for this row");
        if(m==1){launch(1);need(validate()>0,"B1 benchmark missed premature weighting");std::puts("B1 BENCH NEGATIVE PASS actual premature weighting");}
        for(int warm=0;warm<10;++warm)launch(0);ck(cudaDeviceSynchronize(),"B1 warmup");
        if(profile_m){
            ck(cudaProfilerStart(),"B1 profiler start");launch(0);ck(cudaDeviceSynchronize(),"B1 profile sync");ck(cudaProfilerStop(),"B1 profiler stop");
            if(validate())throw Numerical("B1 post-profile reference mismatch");
            std::printf("PROFILE PASS family=B1 M=%d kernels=3; diagnostic counters only, NOT a bandwidth gate\n",m);
            continue;
        }
        B1Event start,stop;std::vector<float> samples;
        for(int sample=0;sample<31;++sample){ck(cudaEventRecord(start.value),"B1 event start");launch(0);ck(cudaEventRecord(stop.value),"B1 event stop");ck(cudaEventSynchronize(stop.value),"B1 event sync");float ms=0;ck(cudaEventElapsedTime(&ms,start.value,stop.value),"B1 elapsed");need(std::isfinite(ms) && ms>0,"B1 timer");samples.push_back(ms);std::printf("B1 BENCH SAMPLE M=%d sample=%d ms=%.9f\n",m,sample,double(ms));}
        if(validate())throw Numerical("B1 post-timing reference mismatch");std::sort(samples.begin(),samples.end());
        const double gbps=proof::weight_gbps(weight_bytes,samples[15]);const int row=proof::bandwidth_verdict(gbps);
        std::printf("B1 BENCH ROW M=%d bytes=%zu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f verdict=%s\n",m,weight_bytes,double(samples[15]),double(samples.front()),double(samples.back()),gbps,row==0?"TARGET":row==6?"STOP":row==7?"PIVOT":"INVALID");
        need(row!=2,"B1 impossible bandwidth");if(row==6)verdict=6;else if(row==7 && !verdict)verdict=7;
    }
    canonical.guard();if(profile_m)return 0;
    std::printf("B1 BENCH FINAL verdict=%d; fixed gate, no tuning or promotion on failure\n",verdict);return verdict;
}
} // namespace
