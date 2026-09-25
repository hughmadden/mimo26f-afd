// Owned diagnostic plumbing only; included after an immutable driver snapshot.
#pragma once
#include "step_hist.h"
#include <memory>
#include <chrono>
namespace step_bench {
#ifndef M26_STEP_REDZONE
#error Select the frozen backend allocator layout explicitly
#endif
constexpr size_t redzone_bytes=M26_STEP_REDZONE;
static_assert(redzone_bytes==16||redzone_bytes==128,"unsupported frozen allocator");
struct Allocation {
    void* base=nullptr;size_t bytes;
    explicit Allocation(size_t n):bytes(n){
        size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"step memory");
        proof::need(n>0&&n+2*redzone_bytes<=free&&free-n-2*redzone_bytes>=uint64_t(8)*1024*1024*1024,"step8GiB reserve");
        ck(cudaMalloc(&base,n+2*redzone_bytes),"step allocate");auto status=cudaMemset(base,0xa5,n+2*redzone_bytes);
        if(status==cudaSuccess&&redzone_bytes==128)status=cudaMemset(static_cast<uint8_t*>(base)+redzone_bytes,0xff,n); // frozen B2 NaN poison
        if(status!=cudaSuccess){if(cudaFree(base)!=cudaSuccess)cleanup_failed=true;base=nullptr;ck(status,"step initialize");}
    }
    Allocation(const Allocation&)=delete;Allocation& operator=(const Allocation&)=delete;
    ~Allocation(){if(base&&cudaFree(base)!=cudaSuccess)cleanup_failed=true;}
    template<class T> T* ptr(){return reinterpret_cast<T*>(static_cast<uint8_t*>(base)+redzone_bytes);}
    void upload(const void* p){ck(cudaMemcpy(ptr<void>(),p,bytes,cudaMemcpyHostToDevice),"step upload");}
    void download(void* p){ck(cudaMemcpy(p,ptr<void>(),bytes,cudaMemcpyDeviceToHost),"step download");}
    void zero(){ck(cudaMemset(ptr<void>(),0,bytes),"step clear");}
    void guard(){std::array<uint32_t,redzone_bytes/4> lo{},hi{};ck(cudaMemcpy(lo.data(),base,redzone_bytes,cudaMemcpyDeviceToHost),"step guard low");ck(cudaMemcpy(hi.data(),static_cast<uint8_t*>(base)+bytes+redzone_bytes,redzone_bytes,cudaMemcpyDeviceToHost),"step guard high");for(size_t i=0;i<lo.size();++i)proof::need(lo[i]==0xa5a5a5a5u&&hi[i]==0xa5a5a5a5u,"step redzone damaged");}
};
struct Event {
    cudaEvent_t value=nullptr;Event(){ck(cudaEventCreate(&value),"step event");}
    ~Event(){if(value&&cudaEventDestroy(value)!=cudaSuccess)cleanup_failed=true;}
    Event(const Event&)=delete;Event& operator=(const Event&)=delete;
};
inline float elapsed(Event& start,Event& stop){
    ck(cudaEventSynchronize(stop.value),"step event sync");float ms=0;ck(cudaEventElapsedTime(&ms,start.value,stop.value),"step elapsed");proof::need(std::isfinite(ms)&&ms>0,"invalid step time");return ms;
}
inline std::vector<uint8_t> artifact(const std::string& dir,const proof::Json& source,const std::string& name){
    const auto data=proof::read(dir+"/"+name,128*1024*1024);const auto& pin=source.at("artifacts").at(name);
    proof::need(data.size()==pin.at("bytes").num()&&proof::sha256(data)==pin.at("sha256").str(),"step oracle artifact identity");return data;
}
inline std::vector<uint8_t> image(const std::string& root,const proof::Json& index,const proof::Json& source,int layer,int expert){
    std::vector<uint8_t> result(expert_bytes);const char* projection[]={"gate_proj","up_proj","down_proj"};
    for(int p=0;p<3;++p){const auto name="model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+"."+projection[p];
        const int rows=p==2?4096:2048,cols=p==2?2048:4096;
        const auto w=proof::tensor_bytes(root,index,name+".weight",rows,cols/2),s=proof::tensor_bytes(root,index,name+".weight_scale",rows,cols/32);
        bool found=false;for(const auto& b:source.at("blocks").list())if(b.at("name").str()==name){proof::need(!found,"duplicate source tensor pin");found=true;proof::need(proof::sha256(w)==b.at("weight_sha256").str()&&proof::sha256(s)==b.at("scale_sha256").str(),"step checkpoint/oracle source mismatch");}
        proof::need(found,"missing checkpoint source pin");proof::copy_rank(result,p,w,s,0);
    }
    return result;
}
inline void identity(){cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0),"step identity");proof::need(p.major==12&&p.minor==1&&p.multiProcessorCount==48,"step requires GB10 sm121/48SM");std::printf("STEP DEVICE name=%s arch=121 sms=48 l2_bytes=%d rank=0 weights_per_expert=%llu\n",p.name,p.l2CacheSize,(unsigned long long)expert_bytes);}
// Adapter owns all mutable data and is destroyed only after checked completion.
template<class Adapter> int measure(Adapter& adapter,const Workload& work,bool frozen=false){
    std::printf("STEP ALLOCATION frozen_redzone_bytes=%zu payload_alignment=%zu\n",redzone_bytes,redzone_bytes);
    Event begin,end;double total_ms=0;uint64_t total_bytes=0;bool noise_ok=true,noise_tested=false;
    for(size_t ordinal=0;ordinal<work.steps.size();++ordinal){const auto& s=work.steps[ordinal];
        std::printf("STEP SCOPE workload=%s ordinal=%zu weight_layer=%d route_layer=%d frozen_order=%d timing=expert_compute_only\n",work.name.c_str(),ordinal,s.layer,s.route_layer,int(frozen));
        adapter.configure(s);auto plan=schedule(s,sample_seed(work.seed,ordinal,41),frozen);validate(s,plan);adapter.prepare(plan);adapter.launch_all(0);proof::need(adapter.check()==0,"step pre-timing reference failure");
        if(ordinal==0){adapter.launch_all(1);proof::need(adapter.check()>0,"step numerical negative not detected");std::puts("STEP NEGATIVE PASS actual backend mutation");}
        for(int warm=0;warm<10;++warm){plan=schedule(s,sample_seed(work.seed,ordinal,warm),frozen);validate(s,plan);adapter.prepare(plan);adapter.launch_all(0);}ck(cudaDeviceSynchronize(),"step warm completion");
        std::vector<float> samples;
        for(int sample=0;sample<31;++sample){plan=schedule(s,sample_seed(work.seed,ordinal,10+sample),frozen);validate(s,plan);adapter.prepare(plan);
            ck(cudaEventRecord(begin.value),"step begin");adapter.launch_all(0);ck(cudaEventRecord(end.value),"step end");const auto ms=elapsed(begin,end);samples.push_back(ms);
            std::printf("STEP SAMPLE family=%s workload=%s ordinal=%zu layer=%d sample=%d experts=%d routes=%d bytes=%llu scheduled_bytes=%llu loads=%zu ms=%.9f schedule=%s\n",adapter.family(),work.name.c_str(),ordinal,s.layer,sample,s.experts,s.routes,(unsigned long long)distinct_bytes(s),(unsigned long long)scheduled_bytes(plan),plan.size(),double(ms),digest(plan).c_str());
        }
        proof::need(adapter.check()==0,"step post-timing reference failure");
        std::sort(samples.begin(),samples.end());const double ms=samples[15],gbps=double(distinct_bytes(s))/(ms*1e6);
        std::printf("STEP ROW family=%s workload=%s ordinal=%zu layer=%d experts=%d routes=%d bytes=%llu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f above_dram_peak=%d diagnostic_only=yes\n",adapter.family(),work.name.c_str(),ordinal,s.layer,s.experts,s.routes,(unsigned long long)distinct_bytes(s),ms,double(samples.front()),double(samples.back()),gbps,int(gbps>273));
        if(std::string(adapter.family())=="B2"&&s.hist.size()==1&&s.hist.count(1)&&s.experts==256){noise_tested=true;const double ratio=gbps/216.742768;const bool ok=ratio>=.95&&ratio<=1.05;noise_ok&=ok;std::printf("STEP M1 SANITY frozen_GBps=216.742768 ratio=%.9f band=0.95..1.05 pass=%d performance_gate_change=no\n",ratio,int(ok));}
        total_ms+=ms;total_bytes+=distinct_bytes(s);
        // Separate instrumented replay: never included in primary samples.
        plan=schedule(s,sample_seed(work.seed,ordinal,42),frozen);validate(s,plan);adapter.prepare(plan);
        for(size_t i=0;i<plan.size();++i){ck(cudaEventRecord(begin.value),"breakdown begin");adapter.launch_one(i,0);ck(cudaEventRecord(end.value),"breakdown end");const auto part=elapsed(begin,end);
            std::printf("STEP BREAKDOWN family=%s workload=%s ordinal=%zu original_M=%d native_M=%d pass=%d experts=%zu ms=%.9f scope=separate_one_shot_replay_not_additive\n",adapter.family(),work.name.c_str(),ordinal,plan[i].original_m,plan[i].rows,plan[i].pass,plan[i].experts.size(),double(part));}
        proof::need(adapter.check()==0,"step breakdown reference failure");
    }
    std::printf("STEP FINAL family=%s workload=%s steps=%zu sum_median_ms=%.9f distinct_bytes=%llu aggregate_effective_GBps=%.6f m1_sanity=%s no_promotion=yes\n",adapter.family(),work.name.c_str(),work.steps.size(),total_ms,(unsigned long long)total_bytes,double(total_bytes)/(total_ms*1e6),!noise_tested?"NOT_APPLICABLE":noise_ok?"PASS":"FAIL");
    return noise_ok?0:6;
}
}
