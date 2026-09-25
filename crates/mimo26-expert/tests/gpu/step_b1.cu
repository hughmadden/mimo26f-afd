// Diagnostic-only wrapper: compile unchanged62d1ca5 kernels, not current lead.
#define main frozen_b1_driver_main
#include "primitive.cu"
#undef main
static_assert(M26B1_ASYNC_SCALE_FC1==1 && M26B1_ASYNC_SCALE_FC2==1 && M26B1_FC1_STAGES==1 && M26B1_BENCH_I16==0,"R18 requires pinned v1 scale-both, not base/pipe/i16");
// Match the lead's frozen Buffer+16 layout; do not optimize the stopped record.
#define M26_STEP_REDZONE 16
#include "step_gpu_common.cuh"
static_assert(step_bench::redzone_bytes==16,"frozen B1 allocation alignment");
namespace step_bench {
struct B1Load {
    int rows,count,routes;size_t mid;
    Allocation x,sc,weights,inverse,original,gw,groups,ng,fault,h,q,qs,qfault,f1,f2,y;
    m26b1::PlanStorage ps;std::vector<int> real;
    B1Load(int m,int n,const std::vector<uint8_t>& xp,const std::vector<uint8_t>& xs):rows(m),count(n),routes(m*n),mid(size_t(n)*16*512),
      x(size_t(routes)*4096),sc(size_t(routes)*128),weights(routes*4),inverse(routes*4),original(routes*4),gw(routes*4),groups(n*sizeof(m26b1::Group)),ng(4),fault(4),h(mid*4),q(mid),qs(mid/32),qfault(mid/32*4),f1(n*4*4),f2(n*32*4),y(size_t(routes)*4096*4),
      ps{inverse.ptr<int32_t>(),original.ptr<int32_t>(),gw.ptr<float>(),groups.ptr<m26b1::Group>(),ng.ptr<uint32_t>(),fault.ptr<uint32_t>(),uint64_t(routes),uint64_t(n)}{
        std::vector<uint8_t> input(x.bytes),scale(sc.bytes);for(int g=0;g<n;++g){std::copy_n(xp.begin(),size_t(m)*4096,input.begin()+size_t(g)*m*4096);std::copy_n(xs.begin(),size_t(m)*128,scale.begin()+size_t(g)*m*128);}x.upload(input.data());sc.upload(scale.data());
        std::vector<int32_t> route(routes);std::iota(route.begin(),route.end(),0);inverse.upload(route.data());original.upload(route.data());std::vector<float> w(routes,.5f);weights.upload(w.data());gw.upload(w.data());uint32_t num=n;ng.upload(&num);fault.zero();
    }
};
struct B1 {
    std::string root,oracle;proof::Json source,index;Allocation canonical{expert_bytes},prepared{256*expert_bytes},slots_device{256*4};m26b1::PreparedInfo info{};m26b1::Pool pool{};
    std::vector<uint8_t> xp,xs;std::vector<double> golden;std::vector<int> pool_ids,slots;int layer=-1;std::vector<std::unique_ptr<B1Load>> loads;
    B1(const std::string& r,const std::string& o):root(r),oracle(o),source(proof::read_json(o+"/source.json")),index(proof::read_json(r+"/model.safetensors.index.json")),pool_ids(proof::bench_experts(256)),slots(256){
        static_assert(m26b1::image_bytes==expert_bytes,"B1 byte basis");proof::need(source.at("activation").str()=="E-ACT-CR32-v1","R17 reference required");
        xp=artifact(oracle,source,"x-payload.u8");xs=artifact(oracle,source,"x-scales.u8");proof::need(xp.size()==8*4096&&xs.size()==8*128,"B1 input extent");for(int i=0;i<256;++i)slots[pool_ids[i]]=i;slots_device.upload(slots.data());
        std::puts("STEP B1 PIN source=62d1ca5797845f65e3ebc41ff180f499fff366cc variant=scale-both FC1_async_scales=true FC2_async_scales=true stages=1 lattice=v1 bitwise_CR32_claim=no diagnostic_only=yes");
    }
    const char* family()const{return "B1-scale-both";}
    void configure(const Step& s){loads.clear();if(layer!=s.layer){
        for(int slot=0;slot<256;++slot){const auto packed=image(root,index,source,s.layer,pool_ids[slot]);canonical.upload(packed.data());ck(m26b1::prepare_async({2,4096,512,0,m26b1::image_bytes},canonical.ptr<uint8_t>(),canonical.bytes,prepared.ptr<uint8_t>()+size_t(slot)*expert_bytes,expert_bytes,&info,0),"step B1 prepare");ck(cudaDeviceSynchronize(),"step prepare reuse");}
        pool={info,prepared.bytes,256};const auto raw=artifact(oracle,source,"b1-L"+std::to_string(s.layer)+".f64");proof::need(raw.size()==size_t(256)*8*4096*8,"B1 reference extent");golden.resize(raw.size()/8);std::memcpy(golden.data(),raw.data(),raw.size());layer=s.layer;
    }for(const auto& l:schedule(s,0))loads.emplace_back(new B1Load(l.rows,int(l.experts.size()),xp,xs));}
    void prepare(const std::vector<Load>& plan){proof::need(plan.size()==loads.size(),"B1 plan load count");for(size_t i=0;i<loads.size();++i){auto& b=*loads[i];b.real=plan[i].experts;std::vector<m26b1::Group> groups(b.count);
        for(int g=0;g<b.count;++g){auto& item=groups[g];item.expert=b.real[g];item.rows=b.rows;item.route_base=g*b.rows;for(int j=0;j<16;++j)item.input_rows[j]=j<b.rows?g*b.rows+j:-1;proof::need(item.expert>=0&&item.expert<256&&item.rows>=1&&item.rows<=8&&item.route_base+item.rows<=b.routes,"B1 grouped metadata");}b.groups.upload(groups.data());b.fault.zero();
    }}
    void launch_one(size_t i,unsigned mutation){auto& b=*loads.at(i);
        m26b1::connected_fc1<128,true><<<dim3(4,b.count),128>>>(pool,prepared.ptr<uint8_t>(),slots_device.ptr<int32_t>(),b.ps,b.x.ptr<uint32_t>(),b.sc.ptr<uint8_t>(),b.h.ptr<float>(),nullptr,nullptr,b.f1.ptr<uint32_t>());ck(cudaGetLastError(),"step scale-both FC1");
        ck(m26b1::quantize_async({b.h.ptr<float>(),b.mid},{b.q.ptr<uint8_t>(),b.qs.ptr<uint8_t>(),b.qfault.ptr<uint32_t>(),b.mid,b.mid/32,b.mid/32},0),"step B1 quantizer");
        m26b1::connected_fc2_fp8<128,true><<<dim3(32,b.count),128>>>(pool,prepared.ptr<uint8_t>(),slots_device.ptr<int32_t>(),b.ps,b.q.ptr<uint8_t>(),b.qs.ptr<uint8_t>(),b.qfault.ptr<uint32_t>(),b.y.ptr<float>(),b.routes,b.f2.ptr<uint32_t>(),b.weights.ptr<float>(),mutation);ck(cudaGetLastError(),"step scale-both FC2");
    }
    void launch_all(unsigned mutation){for(size_t i=0;i<loads.size();++i)launch_one(i,mutation);}
    uint64_t check(){ck(cudaDeviceSynchronize(),"step B1 verify sync");uint64_t bad=0,coordinates=0;
        for(auto& item:loads){auto& b=*item;for(auto* fault:{&b.fault,&b.f1,&b.qfault,&b.f2}){std::vector<uint32_t> f(fault->bytes/4);fault->download(f.data());for(auto v:f)proof::need(!v,"step B1 device fault");}
            std::vector<float> got(b.y.bytes/4);b.y.download(got.data());for(int g=0;g<b.count;++g)for(int j=0;j<b.rows;++j)for(int k=0;k<4096;++k){const auto v=got[(size_t(g)*b.rows+j)*4096+k];const auto want=golden[(size_t(b.real[g])*8+j)*4096+k];++coordinates;bad+=!std::isfinite(v)||!std::isfinite(want)||std::abs(double(v)-want)>1e-5+1e-5*std::abs(want);}
            for(auto* a:{&b.x,&b.sc,&b.weights,&b.inverse,&b.original,&b.gw,&b.groups,&b.ng,&b.fault,&b.h,&b.q,&b.qs,&b.qfault,&b.f1,&b.f2,&b.y})a->guard();
        }canonical.guard();prepared.guard();slots_device.guard();proof::need(coordinates>0,"empty B1 oracle intersection");std::printf("STEP CORRECT family=B1-scale-both coordinates=%llu bad=%llu atol=1e-5 rtol=1e-5\n",(unsigned long long)coordinates,(unsigned long long)bad);return bad;
    }
};
}
int main(int argc,char** argv){int status=0;try{
    proof::need(argc==6,"step-b1 weights oracle histogram workload diagnostic");const auto work=step_bench::parse(proof::read_json(argv[3]),argv[4]);proof::need(std::string(argv[5])=="diagnostic","diagnostic marker required");
    gpu_ready();step_bench::identity();{step_bench::B1 adapter(argv[1],argv[2]);status=step_bench::measure(adapter,work);}
}catch(const std::exception& e){std::fprintf(stderr,"STEP FAILURE %s\n",e.what());status=2;}return cleanup_failed?4:status;}
