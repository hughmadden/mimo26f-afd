// F5-only adapter. No edits to the qualified routing adapters or frozen kernel.
#define main frozen_b2_driver_main
#include "gemm_parity.cu"
#undef main
#if !defined(M26X_EXACT_M) || !M26X_EXACT_M
#error F5 requires frozen exact-M B2
#endif
#define M26_STEP_REDZONE 128
#include "step_gpu_common.cuh"
#include "step_prefill_plan.h"
static_assert(M26X_CAPACITY_CLASS==2048&&step_bench::redzone_bytes==128,"F5 frozen capacity/alignment");
namespace step_bench {
struct PrefillB2 {
    int m;std::vector<int> pool,slots,real;Allocation weights{256*expert_bytes},x,y,scratch,ids{256*4},offsets{257*4},fault{4};m26x_plan p{};std::vector<float> golden;
    PrefillB2(const std::string& root,const std::string& oracle,const Step& s):m(s.hist.begin()->first),pool(proof::bench_experts(256)),slots(256),x(size_t(m)*256*4096*4),y(x.bytes),scratch(size_t(m)*256*1024*4){
        prefill_shape(s);const auto source=proof::read_json(oracle+"/source.json"),index=proof::read_json(root+"/model.safetensors.index.json");
        for(int i=0;i<256;++i){slots[pool[i]]=i;const auto packed=image(root,index,source,1,pool[i]);ck(cudaMemcpy(weights.ptr<uint8_t>()+size_t(i)*expert_bytes,packed.data(),packed.size(),cudaMemcpyHostToDevice),"F5 weights");}
        auto raw=artifact(oracle,source,"x.f32");proof::need(raw.size()==8*4096*4,"F5 input extent");std::vector<float> input(raw.size()/4);std::memcpy(input.data(),raw.data(),raw.size());
        std::vector<float> values(size_t(m)*256*4096);for(int g=0;g<256;++g)for(int r=0;r<m;++r)std::copy_n(input.begin()+size_t(r%8)*4096,4096,values.begin()+(size_t(g)*m+r)*4096);x.upload(values.data());
        raw=artifact(oracle,source,"b2-L1.f32");proof::need(raw.size()==size_t(256)*8*4096*4,"F5 oracle extent");golden.resize(raw.size()/4);std::memcpy(golden.data(),raw.data(),raw.size());
        std::vector<int32_t> off(257);for(int i=0;i<=256;++i)off[i]=i*m;offsets.upload(off.data());
        p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;p.capacity_class=2048;p.resident_experts=256;p.n_groups=256;p.padded_groups=3;p.total_tokens=m*256;p.max_m=m;
        p.grouped_bytes=weights.bytes;p.x_bytes=x.bytes;p.out_bytes=y.bytes;p.scratch_bytes=scratch.bytes;p.expert_ids=ids.ptr<int32_t>();p.group_offsets=offsets.ptr<int32_t>();p.fault=fault.ptr<uint32_t>();
    }
    void prepare(const Step& s,const std::vector<Load>& plan){validate_prefill(s,plan);real=plan[0].experts;std::vector<int32_t> mapped;for(int e:real)mapped.push_back(slots[e]);ids.upload(mapped.data());fault.zero();std::vector<int32_t> off(257);for(int i=0;i<=256;++i)off[i]=i*m;proof::need(m26x_validate_host_plan(&p,mapped.data(),off.data())==0,"F5 native host plan");}
    void launch(unsigned mutation){ck(m26x_expert_ffn_v2(&p,weights.ptr<uint8_t>(),x.ptr<float>(),0,mutation?M26X_NAIVE_NIBBLE_SWAP:0,scratch.ptr<float>(),y.ptr<void>(),nullptr),"F5 frozen FFN");}
    uint64_t check(){ck(cudaDeviceSynchronize(),"F5 verify sync");uint32_t flag=0;fault.download(&flag);proof::need(!flag,"F5 device metadata fault");std::vector<float> got(y.bytes/4);y.download(got.data());uint64_t bad=0;
        for(int g=0;g<256;++g)for(int r=0;r<m;++r)for(int c=0;c<4096;++c){const float v=got[(size_t(g)*m+r)*4096+c],want=golden[(size_t(real[g])*8+r%8)*4096+c];bad+=!std::isfinite(v)||!std::isfinite(want)||std::abs(double(v)-want)>1e-5+1e-5*std::abs(double(want));}
        weights.guard();x.guard();y.guard();scratch.guard();ids.guard();offsets.guard();fault.guard();std::printf("STEP CORRECT family=B2 coordinates=%llu bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)(uint64_t(m)*256*4096),(unsigned long long)bad);return bad;
    }
};
int prefill_run(const std::string& root,const std::string& oracle,const Workload& w){
    proof::need(w.steps.size()==1,"F5 one shape per program");const auto& s=w.steps[0];prefill_shape(s);const int m=s.hist.begin()->first;proof::need(w.name=="prefill-M"+std::to_string(m),"F5 workload identity");
    std::printf("F5 CONTRACT native_max_M=%d host_passes=1 row_tile=%d grid_z=%d activation_prefix_rows=8 cycle_rows=yes report_only=yes\n",m,m==1?1:8,m==1?1:m/8);
    std::puts("STEP ALLOCATION frozen_redzone_bytes=128 payload_alignment=128");std::printf("STEP SCOPE workload=%s ordinal=0 weight_layer=1 route_layer=1 frozen_order=0 timing=expert_compute_only\n",w.name.c_str());
    PrefillB2 b(root,oracle,s);auto plan=schedule_prefill(s,sample_seed(w.seed,0,41));b.prepare(s,plan);b.launch(0);proof::need(b.check()==0,"F5 pre-reference");b.launch(1);proof::need(b.check()>0,"F5 powerless negative");std::puts("STEP NEGATIVE PASS actual backend mutation");
    for(int i=0;i<10;++i){plan=schedule_prefill(s,sample_seed(w.seed,0,i));b.prepare(s,plan);b.launch(0);}ck(cudaDeviceSynchronize(),"F5 warm completion");Event start,end;std::vector<float> times;
    for(int i=0;i<31;++i){plan=schedule_prefill(s,sample_seed(w.seed,0,10+i));b.prepare(s,plan);ck(cudaEventRecord(start.value),"F5 begin");b.launch(0);ck(cudaEventRecord(end.value),"F5 end");const auto ms=elapsed(start,end);times.push_back(ms);
        std::printf("STEP SAMPLE family=B2 workload=%s ordinal=0 layer=1 sample=%d experts=256 routes=%d bytes=%llu scheduled_bytes=%llu loads=1 ms=%.9f schedule=%s\n",w.name.c_str(),i,s.routes,(unsigned long long)distinct_bytes(s),(unsigned long long)distinct_bytes(s),double(ms),digest(plan).c_str());}
    proof::need(b.check()==0,"F5 post-reference");std::sort(times.begin(),times.end());const double ms=times[15],gbps=double(distinct_bytes(s))/(ms*1e6);const bool sanity=m!=1||(gbps/216.742768>=.95&&gbps/216.742768<=1.05);
    std::printf("STEP ROW family=B2 workload=%s ordinal=0 layer=1 experts=256 routes=%d bytes=%llu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f above_dram_peak=%d diagnostic_only=yes\n",w.name.c_str(),s.routes,(unsigned long long)distinct_bytes(s),ms,double(times.front()),double(times.back()),gbps,int(gbps>273));
    plan=schedule_prefill(s,sample_seed(w.seed,0,42));b.prepare(s,plan);ck(cudaEventRecord(start.value),"F5 replay begin");b.launch(0);ck(cudaEventRecord(end.value),"F5 replay end");const auto replay=elapsed(start,end);
    std::printf("STEP BREAKDOWN family=B2 workload=%s ordinal=0 original_M=%d native_M=%d pass=0 experts=256 ms=%.9f scope=separate_one_shot_replay_not_additive\n",w.name.c_str(),m,m,double(replay));proof::need(b.check()==0,"F5 replay reference");
    std::printf("STEP FINAL family=B2 workload=%s steps=1 sum_median_ms=%.9f distinct_bytes=%llu aggregate_effective_GBps=%.6f m1_sanity=%s no_promotion=yes\n",w.name.c_str(),ms,(unsigned long long)distinct_bytes(s),gbps,m==1?(sanity?"PASS":"FAIL"):"NOT_APPLICABLE");
    std::printf("F5 ROW M=%d line_GBps=163.8 at_or_above=%d gate=no native_wide_tile=no\n",m,int(gbps>=163.8));return sanity?0:6;
}
}
int main(int argc,char** argv){int status=0;try{proof::need(argc==6&&std::string(argv[5])=="diagnostic","F5 weights oracle histogram workload diagnostic");const auto work=step_bench::parse(proof::read_json(argv[3]),argv[4]);ready();step_bench::identity();status=step_bench::prefill_run(argv[1],argv[2],work);}catch(const std::exception& e){std::fprintf(stderr,"F5 FAILURE %s\n",e.what());status=2;}return cleanup_failed?4:status;}
