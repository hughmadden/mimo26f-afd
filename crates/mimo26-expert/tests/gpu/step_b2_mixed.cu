// R18c adapter: bitwise-verifies the phase-wise mixed-M FFN against the frozen
// per-M FFN on every step, then times the mixed FFN whole-step. Frozen source
// is untouched; m26x_expert_ffn_v2 (per-M reference) and the new
// m26x_expert_ffn_mixed_v2 (subject) run side by side.
#define main frozen_b2_driver_main
#include "gemm_parity.cu"
#undef main
#if !defined(M26X_EXACT_M) || !M26X_EXACT_M
#error R18c requires frozen exact-M B2
#endif
#define M26_STEP_REDZONE 128
#include "step_gpu_common.cuh"
static_assert(step_bench::redzone_bytes==128,"frozen B2 allocation alignment");
#include "mixed_dispatch.cuh"
static_assert(M26X_CAPACITY_CLASS==2048,"frozen B2 capacity class");

namespace step_bench {
// Frozen per-M reference load, mirroring B2Load in step_b2.cu (unchanged).
struct FrozenLoad {
    int rows,count;Allocation x,y,scratch,ids,offsets,fault;m26x_plan p{};std::vector<int> real;
    FrozenLoad(int m,int n,const std::vector<float>& input):rows(m),count(n),x(size_t(m)*n*4096*4),y(x.bytes),scratch(size_t(m)*n*1024*4),ids(n*4),offsets((n+1)*4),fault(4){
        std::vector<float> values(size_t(m)*n*4096);for(int g=0;g<n;++g)std::copy_n(input.begin(),size_t(m)*4096,values.begin()+size_t(g)*m*4096);x.upload(values.data());
        std::vector<int32_t> off(n+1);for(int g=0;g<=n;++g)off[g]=g*m;offsets.upload(off.data());
        p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;p.capacity_class=M26X_CAPACITY_CLASS;
        p.resident_experts=256;p.n_groups=n;p.padded_groups=3;p.total_tokens=m*n;p.max_m=m;p.grouped_bytes=256*expert_bytes;
        p.x_bytes=x.bytes;p.out_bytes=y.bytes;p.scratch_bytes=scratch.bytes;p.expert_ids=ids.ptr<int32_t>();p.group_offsets=offsets.ptr<int32_t>();p.fault=fault.ptr<uint32_t>();
    }
};
struct MixedB2 {
    std::string root,oracle;proof::Json source,index;Allocation weights{256*expert_bytes};std::vector<float> input,golden;std::vector<int> pool_ids,slots;int layer=-1;
    // Mixed subject buffers (unique_ptr: Allocation has no default constructor).
    std::unique_ptr<Allocation> m_x,m_out,m_scratch,m_ids,m_offsets,m_fault;m26x_plan mp{};std::vector<int> real;std::vector<float> host_x,host_out;
    // O1 host bucketing input: the host copy of the group offsets (n_groups+1).
    std::vector<int32_t> host_offsets;
    // Frozen per-M reference loads.
    std::vector<std::unique_ptr<FrozenLoad>> ref;std::vector<int> ref_off; // token offset per load
    int last_mutation=0;
    MixedB2(const std::string& r,const std::string& o):root(r),oracle(o),source(proof::read_json(o+"/source.json")),index(proof::read_json(r+"/model.safetensors.index.json")),pool_ids(proof::bench_experts(256)),slots(256){
        for(int i=0;i<256;++i)slots[pool_ids[i]]=i;const auto raw=artifact(oracle,source,"x.f32");proof::need(raw.size()==8*4096*4,"mixed input extent");input.resize(raw.size()/4);std::memcpy(input.data(),raw.data(),raw.size());
    }
    const char* family()const{return "B2";}
    void configure(const Step& s){
        ref.clear();ref_off.clear();real.clear();host_x.clear();host_out.clear();
        if(layer!=s.layer){
            for(int slot=0;slot<256;++slot){const auto packed=image(root,index,source,s.layer,pool_ids[slot]);ck(cudaMemcpy(weights.ptr<uint8_t>()+size_t(slot)*expert_bytes,packed.data(),packed.size(),cudaMemcpyHostToDevice),"mixed weight upload");}
            const auto raw=artifact(oracle,source,"b2-L"+std::to_string(s.layer)+".f32");proof::need(raw.size()==size_t(256)*8*4096*4,"mixed reference extent");golden.resize(raw.size()/4);std::memcpy(golden.data(),raw.data(),raw.size());layer=s.layer;
        }
        const auto plan=schedule(s,0);
        // Frozen per-M reference loads + token offsets (ascending M order).
        int off=0;
        for(const auto& l:plan){
            ref.emplace_back(new FrozenLoad(l.rows,int(l.experts.size()),input));
            for(int e:l.experts){auto& b=*ref.back();b.real.push_back(e);std::vector<int32_t> ids;for(int q:l.experts)ids.push_back(slots[q]);b.ids.upload(ids.data());b.fault.zero();std::vector<int32_t> offsets(b.count+1);for(int g=0;g<=b.count;++g)offsets[g]=g*b.rows;proof::need(m26x_validate_host_plan(&b.p,ids.data(),offsets.data())==0,"frozen reference host plan");}
            ref_off.push_back(off);off+=l.rows*int(l.experts.size());
        }
        // Mixed plan: concatenate groups in the same schedule order.
        std::vector<int32_t> ids,offsets(1,0);
        for(const auto& l:plan){
            for(int e:l.experts){ids.push_back(slots[e]);real.push_back(e);offsets.push_back(offsets.back()+l.rows);for(int r=0;r<l.rows;++r)host_x.insert(host_x.end(),input.begin()+size_t(r)*4096,input.begin()+size_t(r+1)*4096);}
        }
        const int T=offsets.back();proof::need(T==s.routes,"mixed token count");int maxm=0;for(size_t i=0;i+1<offsets.size();++i)maxm=std::max(maxm,offsets[i+1]-offsets[i]);
        proof::need(maxm<=8,"mixed kernel decode widths only");proof::need(int(ids.size())==s.experts,"mixed group count");
        mp.layout_version=2;mp.manifest_arch=M26X_BAKED_ARCH;mp.manifest_sms=M26X_BAKED_SMS;mp.capacity_class=2048;mp.resident_experts=256;
        mp.n_groups=int(ids.size());mp.padded_groups=3;mp.total_tokens=T;mp.max_m=maxm;mp.grouped_bytes=256*expert_bytes;
        mp.x_bytes=size_t(T)*4096*4;mp.out_bytes=size_t(T)*4096*4;mp.scratch_bytes=size_t(T)*512*4*2;
        m_x.reset(new Allocation(mp.x_bytes));m_out.reset(new Allocation(mp.out_bytes));m_scratch.reset(new Allocation(mp.scratch_bytes));m_ids.reset(new Allocation(size_t(ids.size())*4));m_offsets.reset(new Allocation(size_t(offsets.size())*4));m_fault.reset(new Allocation(4));
        mp.expert_ids=m_ids->ptr<int32_t>();mp.group_offsets=m_offsets->ptr<int32_t>();mp.fault=m_fault->ptr<uint32_t>();
        m_ids->upload(ids.data());m_offsets->upload(offsets.data());m_x->upload(host_x.data());m_fault->zero();
        std::vector<int32_t> host_ids=ids;proof::need(m26x_validate_host_plan(&mp,host_ids.data(),offsets.data())==0,"mixed host plan");
        host_offsets=offsets;
        host_out.resize(size_t(T)*4096);
    }
    void launch_frozen_reference(){
        for(auto& item:ref){auto& b=*item;b.fault.zero();ck(m26x_expert_ffn_v2(&b.p,weights.ptr<uint8_t>(),b.x.ptr<float>(),0,0,b.scratch.ptr<float>(),b.y.ptr<void>(),nullptr),"frozen reference FFN");}
    }
    void launch_mixed(unsigned mutation){
        last_mutation=int(mutation);m_fault->zero();
        ck(m26x_expert_ffn_mixed_v2(&mp,weights.ptr<uint8_t>(),m_x->ptr<float>(),0,mutation?M26X_NAIVE_NIBBLE_SWAP:0,m_scratch->ptr<float>(),m_out->ptr<void>(),nullptr),"mixed FFN");
    }
    void launch_o1(unsigned mutation){
        last_mutation=int(mutation);m_fault->zero();
        ck(m26x_expert_ffn_mixed_o1_v2(&mp,host_offsets.data(),weights.ptr<uint8_t>(),m_x->ptr<float>(),0,mutation?M26X_NAIVE_NIBBLE_SWAP:0,m_scratch->ptr<float>(),m_out->ptr<void>(),nullptr),"O1 FFN");
    }
    // Compare mixed output against frozen per-M reference bitwise + golden (1e-5).
    // Returns bitwise mismatch count; fills golden_bad via out param.
    static uint32_t fbits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
    uint64_t check(uint64_t& golden_bad){
        ck(cudaDeviceSynchronize(),"mixed verify sync");uint32_t flag=0;m_fault->download(&flag);proof::need(!flag,"mixed metadata fault");m_out->download(host_out.data());
        uint64_t bitwise=0;golden_bad=0;
        for(size_t li=0;li<ref.size();++li){auto& b=*ref[li];uint32_t f=0;b.fault.download(&f);proof::need(!f,"frozen reference fault");std::vector<float> fr(b.y.bytes/4);b.y.download(fr.data());const int off=ref_off[li];
            for(int t=0;t<b.count*b.rows;++t)for(int c=0;c<4096;++c){
                const float got=host_out[size_t(off+t)*4096+c],want=fr[size_t(t)*4096+c];
                bitwise+=fbits(got)!=fbits(want);
                const int expert=b.real[t/b.rows],row=t%b.rows;const float gold=golden[(size_t(expert)*8+row)*4096+c];
                golden_bad+=!std::isfinite(got)||!std::isfinite(gold)||std::abs(double(got)-gold)>1e-5+1e-5*std::abs(double(gold));
            }
        }
        m_x->guard();m_out->guard();m_scratch->guard();m_ids->guard();m_offsets->guard();m_fault->guard();weights.guard();
        return bitwise;
    }
    void reference_check(const Step& s){
        launch_frozen_reference();launch_mixed(0);uint64_t gb=0;const auto bw=check(gb);
        proof::need(bw==0&&gb==0,"mixed bitwise/reference failure");
        std::printf("STEP CORRECT family=B2 coordinates=%llu bitwise_bad=%llu golden_bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)(uint64_t(s.routes)*4096),(unsigned long long)bw,(unsigned long long)gb);
    }
};
int mixed_run(const std::string& root,const std::string& oracle,const Workload& w){
    std::printf("MIXED CONTRACT kernels=4 max_m=8 policy=phase_wise_coalesced exact_M=1 grid_z=1 correct_entry=mixed_gemm<true> separate_naive_entry=mixed_gemm<false>\n");
    std::puts("STEP ALLOCATION frozen_redzone_bytes=128 payload_alignment=128");
    MixedB2 b(root,oracle);Event begin,end;double total_ms=0;uint64_t total_bytes=0;
    for(size_t ordinal=0;ordinal<w.steps.size();++ordinal){const auto& s=w.steps[ordinal];
        std::printf("STEP SCOPE workload=%s ordinal=%zu weight_layer=%d route_layer=%d frozen_order=0 timing=expert_compute_only\n",w.name.c_str(),ordinal,s.layer,s.route_layer);
        b.configure(s);b.reference_check(s);
        // O1 (per-M sibling) bitwise proof against the same frozen reference.
        b.launch_o1(0);uint64_t gb1=0;const auto bw1=b.check(gb1);
        proof::need(bw1==0&&gb1==0,"O1 bitwise/reference failure");
        std::printf("O1 CORRECT family=B2 coordinates=%llu bitwise_bad=%llu golden_bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)(uint64_t(s.routes)*4096),(unsigned long long)bw1,(unsigned long long)gb1);
        if(ordinal==0){b.launch_mixed(1);uint64_t gb=0;const auto bw=b.check(gb);proof::need(bw>0,"mixed naive negative powerless");std::puts("STEP NEGATIVE PASS actual backend mutation");}
        if(ordinal==0){b.launch_o1(1);uint64_t gb2=0;const auto bw2=b.check(gb2);proof::need(bw2>0,"O1 naive negative powerless");std::puts("O1 NEGATIVE PASS actual backend mutation");}
        for(int warm=0;warm<10;++warm)b.launch_mixed(0);ck(cudaDeviceSynchronize(),"mixed warm completion");
        for(int warm=0;warm<10;++warm)b.launch_o1(0);ck(cudaDeviceSynchronize(),"O1 warm completion");
        std::vector<float> samples,o1_samples;
        for(int sample=0;sample<31;++sample){
            ck(cudaEventRecord(begin.value),"mixed begin");b.launch_mixed(0);ck(cudaEventRecord(end.value),"mixed end");const auto ms=elapsed(begin,end);samples.push_back(ms);
            ck(cudaEventRecord(begin.value),"O1 begin");b.launch_o1(0);ck(cudaEventRecord(end.value),"O1 end");const auto o1ms=elapsed(begin,end);o1_samples.push_back(o1ms);
            std::printf("STEP SAMPLE family=B2 workload=%s ordinal=%zu layer=%d sample=%d experts=%d routes=%d bytes=%llu scheduled_bytes=%llu groups=%d ms=%.9f o1_ms=%.9f schedule=%s\n",w.name.c_str(),ordinal,s.layer,sample,s.experts,s.routes,(unsigned long long)distinct_bytes(s),(unsigned long long)distinct_bytes(s),s.experts,double(ms),double(o1ms),digest(schedule(s,sample_seed(w.seed,ordinal,10+sample))).c_str());
        }
        b.reference_check(s);b.launch_o1(0);{uint64_t gb2=0;const auto bw2=b.check(gb2);proof::need(bw2==0&&gb2==0,"O1 post-timing bitwise/reference failure");}
        std::sort(samples.begin(),samples.end());const double ms=samples[15],gbps=double(distinct_bytes(s))/(ms*1e6);
        std::sort(o1_samples.begin(),o1_samples.end());const double o1ms=o1_samples[15],o1gbps=double(distinct_bytes(s))/(o1ms*1e6);
        std::printf("STEP ROW family=B2 workload=%s ordinal=%zu layer=%d experts=%d routes=%d bytes=%llu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f above_dram_peak=%d diagnostic_only=yes\n",w.name.c_str(),ordinal,s.layer,s.experts,s.routes,(unsigned long long)distinct_bytes(s),ms,double(samples.front()),double(samples.back()),gbps,int(gbps>273));
        std::printf("O1 ROW family=B2 workload=%s ordinal=%zu layer=%d experts=%d routes=%d bytes=%llu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f above_dram_peak=%d trial=pre_registered_184_190 central=187.0 no_promotion=yes\n",w.name.c_str(),ordinal,s.layer,s.experts,s.routes,(unsigned long long)distinct_bytes(s),o1ms,double(o1_samples.front()),double(o1_samples.back()),o1gbps,int(o1gbps>273));
        total_ms+=ms;total_bytes+=distinct_bytes(s);
    }
    std::printf("STEP FINAL family=B2 workload=%s steps=%zu sum_median_ms=%.9f distinct_bytes=%llu aggregate_effective_GBps=%.6f m1_sanity=NOT_APPLICABLE no_promotion=yes\n",w.name.c_str(),w.steps.size(),total_ms,(unsigned long long)total_bytes,double(total_bytes)/(total_ms*1e6));
    std::printf("R18C LINE gbps=%.6f pass=191.1 stop_below=163.8 verdict=%s gate=conditional_on_F1 no_promotion=yes\n",double(total_bytes)/(total_ms*1e6),double(total_bytes)/(total_ms*1e6)>=191.1?"PASS":double(total_bytes)/(total_ms*1e6)>=163.8?"BELOW_TARGET_RECORDED":"STOP");
    return 0;
}
}
int main(int argc,char** argv){int status=0;try{
    proof::need(argc==6,"step-b2-mixed weights oracle histogram workload diagnostic");const auto work=step_bench::parse(proof::read_json(argv[3]),argv[4]);
    proof::need(std::string(argv[5])=="diagnostic","diagnostic marker required");
    ready();step_bench::identity();status=step_bench::mixed_run(argv[1],argv[2],work);
}catch(const std::exception& e){std::fprintf(stderr,"STEP FAILURE %s\n",e.what());status=2;}return cleanup_failed?4:status;}
