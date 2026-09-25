// F5 B1-v1 native-wide prefill diagnostic (M16/M64), report-only. The frozen
// B1 kernel supports16-row groups natively, so M16 is one16-row group per expert
// and M64 is four16-row groups per expert. Input is the frozen8-row benchmark
// prefix cycled over the M rows (cycle_rows=yes), matching the B2 F5 cell.
#define main frozen_b1_driver_main
#include "primitive.cu"
#undef main
static_assert(M26B1_ASYNC_SCALE_FC1==1 && M26B1_ASYNC_SCALE_FC2==1 && M26B1_FC1_STAGES==1 && M26B1_BENCH_I16==0,"F5 requires pinned v1 scale-both");
#define M26_STEP_REDZONE 16
#include "step_gpu_common.cuh"
#include "step_prefill_plan.h"
static_assert(step_bench::redzone_bytes==16,"frozen B1 allocation alignment");
namespace step_bench {
struct PrefillB1 {
    int m,gpe,ngroups,routes;size_t mid;
    Allocation canonical{expert_bytes},prepared{256*expert_bytes},slots_device{256*4};
    m26b1::PreparedInfo info{};m26b1::Pool pool{};std::vector<int> pool_ids,slots;
    std::vector<uint8_t> xp,xs;std::vector<double> golden;std::vector<int> real;
    Allocation x,sc,inverse,original,gw,groups,ng,fault,h,q,qs,qfault,f1,f2,y;m26b1::PlanStorage ps{};
    PrefillB1(const std::string& r,const std::string& o,const Step& s):m(s.hist.begin()->first),gpe((m+15)/16),ngroups(256*gpe),routes(m*256),mid(size_t(ngroups)*16*512),
      x(size_t(routes)*4096),sc(size_t(routes)*128),inverse(routes*4),original(routes*4),gw(routes*4),groups(size_t(ngroups)*sizeof(m26b1::Group)),ng(4),fault(4),h(mid*4),q(mid),qs(mid/32),qfault(mid/32*4),f1(ngroups*4*4),f2(ngroups*32*4),y(size_t(routes)*4096*4){
        prefill_shape(s);proof::need(m==16||m==64,"F5 B1 native rows are16 or64");
        const auto source=proof::read_json(o+"/source.json"),index=proof::read_json(r+"/model.safetensors.index.json");
        proof::need(source.at("activation").str()=="E-ACT-CR32-v1","R17 reference required");
        pool_ids=proof::bench_experts(256);slots.resize(256);for(int i=0;i<256;++i)slots[pool_ids[i]]=i;slots_device.upload(slots.data());
        for(int slot=0;slot<256;++slot){const auto packed=image(r,index,source,1,pool_ids[slot]);canonical.upload(packed.data());ck(m26b1::prepare_async({2,4096,512,0,m26b1::image_bytes},canonical.ptr<uint8_t>(),canonical.bytes,prepared.ptr<uint8_t>()+size_t(slot)*expert_bytes,expert_bytes,&info,0),"F5 B1 prepare");}
        ck(cudaDeviceSynchronize(),"F5 B1 prepare sync");pool={info,prepared.bytes,256};
        xp=artifact(o,source,"x-payload.u8");xs=artifact(o,source,"x-scales.u8");proof::need(xp.size()==8*4096&&xs.size()==8*128,"F5 B1 input extent");
        const auto raw=artifact(o,source,"b1-L1.f64");proof::need(raw.size()==size_t(256)*8*4096*8,"F5 B1 oracle extent");golden.resize(raw.size()/8);std::memcpy(golden.data(),raw.data(),raw.size());
        // Cycled8-row prefix input, e-major then16-row sub-group.
        std::vector<uint8_t> px(size_t(routes)*4096),scv(size_t(routes)*128);
        for(int e=0;e<256;++e)for(int rr=0;rr<m;++rr){std::copy_n(xp.begin()+size_t(rr%8)*4096,4096,px.begin()+(size_t(e)*m+rr)*4096);std::copy_n(xs.begin()+size_t(rr%8)*128,128,scv.begin()+(size_t(e)*m+rr)*128);}
        x.upload(px.data());sc.upload(scv.data());
        std::vector<int32_t> route(routes);std::iota(route.begin(),route.end(),0);inverse.upload(route.data());original.upload(route.data());std::vector<float> w(routes,.5f);gw.upload(w.data());
        std::vector<m26b1::Group> gs(ngroups);real.clear();
        for(int e=0;e<256;++e)for(int sg=0;sg<gpe;++sg){const int g=e*gpe+sg;auto& item=gs[g];item.expert=e;item.rows=16;item.route_base=g*16;for(int j=0;j<16;++j)item.input_rows[j]=g*16+j;real.push_back(e);}
        groups.upload(gs.data());uint32_t num=ngroups;ng.upload(&num);fault.zero();
        ps={inverse.ptr<int32_t>(),original.ptr<int32_t>(),gw.ptr<float>(),groups.ptr<m26b1::Group>(),ng.ptr<uint32_t>(),fault.ptr<uint32_t>(),uint64_t(routes),uint64_t(ngroups)};
    }
    void launch(unsigned mutation){
        fault.zero();
        m26b1::connected_fc1<128,true><<<dim3(4,ngroups),128>>>(pool,prepared.ptr<uint8_t>(),slots_device.ptr<int32_t>(),ps,x.ptr<uint32_t>(),sc.ptr<uint8_t>(),h.ptr<float>(),nullptr,nullptr,f1.ptr<uint32_t>());ck(cudaGetLastError(),"F5 B1 FC1");
        ck(m26b1::quantize_async({h.ptr<float>(),mid},{q.ptr<uint8_t>(),qs.ptr<uint8_t>(),qfault.ptr<uint32_t>(),mid,mid/32,mid/32},0),"F5 B1 quantizer");
        m26b1::connected_fc2_fp8<128,true><<<dim3(32,ngroups),128>>>(pool,prepared.ptr<uint8_t>(),slots_device.ptr<int32_t>(),ps,q.ptr<uint8_t>(),qs.ptr<uint8_t>(),qfault.ptr<uint32_t>(),y.ptr<float>(),routes,f2.ptr<uint32_t>(),gw.ptr<float>(),mutation);ck(cudaGetLastError(),"F5 B1 FC2");
    }
    uint64_t check(){
        ck(cudaDeviceSynchronize(),"F5 B1 verify sync");for(auto* f:{&fault,&f1,&qfault,&f2}){std::vector<uint32_t> v(f->bytes/4);f->download(v.data());for(auto z:v)proof::need(!z,"F5 B1 device fault");}
        std::vector<float> got(y.bytes/4);y.download(got.data());uint64_t bad=0;
        for(int e=0;e<256;++e)for(int rr=0;rr<m;++rr)for(int k=0;k<4096;++k){const float v=got[(size_t(e)*m+rr)*4096+k],want=float(golden[(size_t(e)*8+rr%8)*4096+k]);bad+=!std::isfinite(v)||!std::isfinite(want)||std::abs(double(v)-want)>1e-5+1e-5*std::abs(double(want));}
        canonical.guard();prepared.guard();slots_device.guard();for(auto* a:{&x,&sc,&inverse,&original,&gw,&groups,&ng,&fault,&h,&q,&qs,&qfault,&f1,&f2,&y})a->guard();
        std::printf("STEP CORRECT family=B1-scale-both coordinates=%llu bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)(uint64_t(routes)*4096),(unsigned long long)bad);return bad;
    }
};
int prefill_run(const std::string& root,const std::string& oracle,const Workload& w){
    proof::need(w.steps.size()==1,"F5 B1 one shape");const auto& s=w.steps[0];prefill_shape(s);const int m=s.hist.begin()->first;proof::need(w.name=="prefill-M"+std::to_string(m),"F5 B1 workload identity");
    std::printf("F5 CONTRACT native_max_M=%d host_passes=1 row_tile=16 grid_z=%d activation_prefix_rows=8 cycle_rows=yes report_only=yes\n",m,m==16?1:4);
    std::puts("STEP B1 PIN source=62d1ca5797845f65e3ebc41ff180f499fff366cc variant=scale-both FC1_async_scales=true FC2_async_scales=true stages=1 lattice=v1 bitwise_CR32_claim=no diagnostic_only=yes");
    std::puts("STEP ALLOCATION frozen_redzone_bytes=16 payload_alignment=16");
    std::printf("STEP SCOPE workload=%s ordinal=0 weight_layer=1 route_layer=1 frozen_order=0 timing=expert_compute_only\n",w.name.c_str());
    PrefillB1 b(root,oracle,s);b.launch(0);proof::need(b.check()==0,"F5 B1 pre-reference");b.launch(1);proof::need(b.check()>0,"F5 B1 powerless negative");std::puts("STEP NEGATIVE PASS actual backend mutation");
    for(int i=0;i<10;++i)b.launch(0);ck(cudaDeviceSynchronize(),"F5 B1 warm");Event start,end;std::vector<float> times;
    for(int i=0;i<31;++i){ck(cudaEventRecord(start.value),"F5 B1 begin");b.launch(0);ck(cudaEventRecord(end.value),"F5 B1 end");const auto ms=elapsed(start,end);times.push_back(ms);
        std::printf("STEP SAMPLE family=B1-scale-both workload=%s ordinal=0 layer=1 sample=%d experts=256 routes=%d bytes=%llu scheduled_bytes=%llu loads=1 ms=%.9f schedule=%s\n",w.name.c_str(),i,s.routes,(unsigned long long)distinct_bytes(s),(unsigned long long)distinct_bytes(s),double(ms),digest(schedule_prefill(s,7)).c_str());}
    proof::need(b.check()==0,"F5 B1 post-reference");std::sort(times.begin(),times.end());const double ms=times[15],gbps=double(distinct_bytes(s))/(ms*1e6);
    std::printf("STEP ROW family=B1-scale-both workload=%s ordinal=0 layer=1 experts=256 routes=%d bytes=%llu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f above_dram_peak=%d diagnostic_only=yes\n",w.name.c_str(),s.routes,(unsigned long long)distinct_bytes(s),ms,double(times.front()),double(times.back()),gbps,int(gbps>273));
    ck(cudaEventRecord(start.value),"F5 B1 replay begin");b.launch(0);ck(cudaEventRecord(end.value),"F5 B1 replay end");const auto replay=elapsed(start,end);
    std::printf("STEP BREAKDOWN family=B1-scale-both workload=%s ordinal=0 original_M=%d native_M=16 pass=0 experts=256 ms=%.9f scope=separate_one_shot_replay_not_additive\n",w.name.c_str(),m,double(replay));proof::need(b.check()==0,"F5 B1 replay reference");
    std::printf("STEP FINAL family=B1-scale-both workload=%s steps=1 sum_median_ms=%.9f distinct_bytes=%llu aggregate_effective_GBps=%.6f m1_sanity=NOT_APPLICABLE no_promotion=yes\n",w.name.c_str(),ms,(unsigned long long)distinct_bytes(s),gbps);
    std::printf("F5 ROW M=%d line_GBps=163.8 at_or_above=%d gate=no native_wide_tile=no\n",m,int(gbps>=163.8));return 0;
}
}
int main(int argc,char** argv){int status=0;try{proof::need(argc==6&&std::string(argv[5])=="diagnostic","F5 B1 weights oracle histogram workload diagnostic");const auto work=step_bench::parse(proof::read_json(argv[3]),argv[4]);gpu_ready();step_bench::identity();status=step_bench::prefill_run(argv[1],argv[2],work);}catch(const std::exception& e){std::fprintf(stderr,"F5 B1 FAILURE %s\n",e.what());status=2;}return cleanup_failed?4:status;}
