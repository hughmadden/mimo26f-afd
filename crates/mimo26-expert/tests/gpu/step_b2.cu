// Frozen driver supplies ABI/proof utilities; its original main is not invoked.
#define main frozen_b2_driver_main
#include "gemm_parity.cu"
#undef main
#if !defined(M26X_EXACT_M) || !M26X_EXACT_M
#error R18 requires the frozen exact-M policy
#endif
// Match frozen Guarded: preserve128B row-load alignment, including canaries.
#define M26_STEP_REDZONE 128
#include "step_gpu_common.cuh"
static_assert(step_bench::redzone_bytes==128,"frozen B2 allocation alignment");
namespace step_bench {
struct B2Load {
    int rows,count;Allocation x,y,scratch,ids,offsets,fault; m26x_plan p{};std::vector<int> real;
    B2Load(int m,int n,const std::vector<float>& input):rows(m),count(n),x(size_t(m)*n*4096*4),y(x.bytes),scratch(size_t(m)*n*1024*4),ids(n*4),offsets((n+1)*4),fault(4){
        std::vector<float> values(size_t(m)*n*4096);for(int g=0;g<n;++g)std::copy_n(input.begin(),size_t(m)*4096,values.begin()+size_t(g)*m*4096);x.upload(values.data());
        std::vector<int32_t> off(n+1);for(int g=0;g<=n;++g)off[g]=g*m;offsets.upload(off.data());
        p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;p.capacity_class=M26X_CAPACITY_CLASS;
        p.resident_experts=256;p.n_groups=n;p.padded_groups=3;p.total_tokens=m*n;p.max_m=m;p.grouped_bytes=256*expert_bytes;
        p.x_bytes=x.bytes;p.out_bytes=y.bytes;p.scratch_bytes=scratch.bytes;p.expert_ids=ids.ptr<int32_t>();p.group_offsets=offsets.ptr<int32_t>();p.fault=fault.ptr<uint32_t>();
    }
};
struct B2 {
    std::string root,oracle;proof::Json source,index;Allocation weights{256*expert_bytes};std::vector<float> input,golden;std::vector<int> pool_ids,slots;int layer=-1;std::vector<std::unique_ptr<B2Load>> loads;
    B2(const std::string& r,const std::string& o):root(r),oracle(o),source(proof::read_json(o+"/source.json")),index(proof::read_json(r+"/model.safetensors.index.json")),pool_ids(proof::bench_experts(256)),slots(256){
        for(int i=0;i<256;++i)slots[pool_ids[i]]=i;const auto raw=artifact(oracle,source,"x.f32");proof::need(raw.size()==8*4096*4,"step B2 input extent");input.resize(raw.size()/4);std::memcpy(input.data(),raw.data(),raw.size());
    }
    const char* family()const{return "B2";}
    void configure(const Step& s){
        loads.clear();if(layer!=s.layer){
            for(int slot=0;slot<256;++slot){const auto packed=image(root,index,source,s.layer,pool_ids[slot]);ck(cudaMemcpy(weights.ptr<uint8_t>()+size_t(slot)*expert_bytes,packed.data(),packed.size(),cudaMemcpyHostToDevice),"step B2 weight upload");}
            const auto raw=artifact(oracle,source,"b2-L"+std::to_string(s.layer)+".f32");proof::need(raw.size()==size_t(256)*8*4096*4,"step B2 reference extent");golden.resize(raw.size()/4);std::memcpy(golden.data(),raw.data(),raw.size());layer=s.layer;
        }
        const auto plan=schedule(s,0);for(const auto& l:plan)loads.emplace_back(new B2Load(l.rows,int(l.experts.size()),input));
    }
    void prepare(const std::vector<Load>& plan){proof::need(plan.size()==loads.size(),"B2 plan load count");
        for(size_t i=0;i<loads.size();++i){auto& b=*loads[i];b.real=plan[i].experts;std::vector<int32_t> ids;for(int e:b.real)ids.push_back(slots[e]);b.ids.upload(ids.data());b.fault.zero();std::vector<int32_t> offsets(b.count+1);for(int g=0;g<=b.count;++g)offsets[g]=g*b.rows;proof::need(m26x_validate_host_plan(&b.p,ids.data(),offsets.data())==0,"step B2 host plan");}
    }
    void launch_one(size_t i,unsigned mutation){auto& b=*loads.at(i);ck(m26x_expert_ffn_v2(&b.p,weights.ptr<uint8_t>(),b.x.ptr<float>(),0,mutation?M26X_NAIVE_NIBBLE_SWAP:0,b.scratch.ptr<float>(),b.y.ptr<void>(),nullptr),"step B2 FFN");}
    void launch_all(unsigned mutation){for(size_t i=0;i<loads.size();++i)launch_one(i,mutation);}
    uint64_t check(){ck(cudaDeviceSynchronize(),"step B2 verify sync");uint64_t bad=0,coordinates=0;
        for(auto& item:loads){auto& b=*item;uint32_t fault=0;b.fault.download(&fault);proof::need(!fault,"step B2 metadata fault");std::vector<float> got(b.y.bytes/4);b.y.download(got.data());
            for(int group=0;group<b.count;++group)for(int row=0;row<b.rows;++row)for(int col=0;col<4096;++col){const float v=got[(size_t(group)*b.rows+row)*4096+col],want=golden[(size_t(b.real[group])*8+row)*4096+col];++coordinates;bad+=!std::isfinite(v)||!std::isfinite(want)||std::abs(double(v)-want)>1e-5+1e-5*std::abs(double(want));}
            b.x.guard();b.y.guard();b.scratch.guard();b.ids.guard();b.offsets.guard();b.fault.guard();
        }weights.guard();proof::need(coordinates>0,"empty B2 oracle intersection");std::printf("STEP CORRECT family=B2 coordinates=%llu bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)coordinates,(unsigned long long)bad);return bad;
    }
};
}
int main(int argc,char** argv){int status=0;try{
    proof::need(argc==6||argc==7,"step-b2 weights oracle histogram workload [frozen-order]");const auto work=step_bench::parse(proof::read_json(argv[3]),argv[4]);
    // argv5 is an explicit protocol marker; prevents accidental legacy invocation.
    proof::need(std::string(argv[5])=="diagnostic","diagnostic marker required");
    const bool frozen=argc==7;proof::need(!frozen||std::string(argv[6])=="frozen-order","unknown ordering");
    ready();step_bench::identity();{step_bench::B2 adapter(argv[1],argv[2]);status=step_bench::measure(adapter,work,frozen);}
}catch(const std::exception& e){std::fprintf(stderr,"STEP FAILURE %s\n",e.what());status=2;}return cleanup_failed?4:status;}
