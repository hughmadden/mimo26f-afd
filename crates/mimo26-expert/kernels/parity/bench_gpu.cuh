// CUDA-event FFN timing, after independent real-reference checks. No tuning.
#include <cuda_profiler_api.h>
struct BenchEvent {
    cudaEvent_t value=nullptr;
    BenchEvent(){ck(cudaEventCreate(&value),"event create");}
    ~BenchEvent(){if(value && cudaEventDestroy(value)!=cudaSuccess)cleanup_failed=true;}
    BenchEvent(const BenchEvent&)=delete;BenchEvent& operator=(const BenchEvent&)=delete;
};
int bench_gpu(const std::string& root,const std::string& fixture_path,const std::string& golden,int residents,int profile_m=0,int diagnostic_m=0) {
    need(profile_m==0 || (residents==256 && (profile_m==4 || profile_m==5)),"B2 profile requires 256 residents and M4/M5");
    need(diagnostic_m==0 || (diagnostic_m==5 && residents==256 && !profile_m),"diagnostic requires only M5, all256, no profiler range");
    ready();need(M26X_BAKED_ARCH==121 && M26X_BAKED_SMS==48,"GB10 timing cell only");
    const auto fixture=proof::read_json(fixture_path);proof::validate_fixture(fixture);
    const auto index=proof::read_json(root+"/model.safetensors.index.json");
    const auto experts=proof::bench_experts(residents);
    const auto xref=proof::floats(golden+"/oracle-x.f32",64*4096);
    std::array<std::vector<float>,3> reference;
    for(int i=0;i<3;++i)reference[i]=proof::floats(golden+"/oracle-e"+std::to_string(experts[i])+".f32",64*4096);
    const size_t weight_bytes=size_t(residents)*M26X_QUARTER_SLICE_BYTES;
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"benchmark device properties");
    need(prop.l2CacheSize>0 && weight_bytes>size_t(prop.l2CacheSize)*2,"working set too small versus L2");
    Guarded weights(weight_bytes);
    for(int i=0;i<residents;++i) {
        const auto image=proof::bench_image(root,index,fixture,experts[i]);
        ck(cudaMemcpy(static_cast<uint8_t*>(weights.data())+size_t(i)*image.size(),image.data(),image.size(),cudaMemcpyHostToDevice),"real rank0 upload");
    }
    if(profile_m)std::printf("PROFILE CONTRACT family=B2 M=%d residents=256 bytes=%zu warmups=10 captured_launches=1; no event bandwidth samples\n",profile_m,weight_bytes);
    else if(diagnostic_m)std::printf("DIAGNOSTIC CONTRACT M=5 residents=256 bytes=%zu policy=forced4-paired duplicated_weight_decode=yes warmups=10 samples=31; not all-M qualification\n",weight_bytes);
    else std::printf("BENCH CONTRACT layer=1 rank=0 residents=%d unique_weight_scale_bytes=%zu l2_bytes=%d dtype=f32 source=real_checkpoint warmups=10 samples=31 timed_scope=three_GEMMs_plus_SiLU metric=effective_unique_weight_GBps clocks=unlocked\n",residents,weight_bytes,prop.l2CacheSize);
#if defined(M26X_EXACT_M) && M26X_EXACT_M
    std::puts("BENCH POLICY experimental_exact_M=1 templates=1,2,4,5,6,7,8 M8_work_unchanged=yes");
#endif
    int verdict=0;
    for(int m=1;m<=8;++m) {
        if((profile_m && m!=profile_m) || (diagnostic_m && m!=diagnostic_m))continue;
        const int tokens=residents*m;const size_t values=size_t(tokens)*4096;
        Guarded x(values*4),out(values*4),scratch(size_t(tokens)*1024*4);
        Device dids(size_t(residents)*4),doff(size_t(residents+1)*4),fault(4);
        std::vector<int32_t> ids,offsets={0};std::vector<float> input(values),result(values);
        for(int g=0;g<residents;++g) {
            ids.push_back(residents-1-g);offsets.push_back((g+1)*m);
            std::copy_n(xref.begin(),size_t(m)*4096,input.begin()+size_t(g)*m*4096);
        }
        x.upload(input.data());dids.upload(ids.data());doff.upload(offsets.data());ck(cudaMemset(fault.ptr,0,4),"bench fault clear");
        m26x_plan p{};p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;p.capacity_class=M26X_CAPACITY_CLASS;
        p.resident_experts=residents;p.n_groups=residents;p.padded_groups=3;p.total_tokens=tokens;p.max_m=m;
        p.grouped_bytes=weight_bytes;p.x_bytes=x.bytes;p.out_bytes=out.bytes;p.scratch_bytes=scratch.bytes;
        p.expert_ids=static_cast<int32_t*>(dids.ptr);p.group_offsets=static_cast<int32_t*>(doff.ptr);p.fault=static_cast<uint32_t*>(fault.ptr);
        need(m26x_validate_host_plan(&p,ids.data(),offsets.data())==0,"benchmark host plan");
        auto launch=[&](uint32_t naive) {ck(m26x_expert_ffn_v2(&p,static_cast<uint8_t*>(weights.data()),static_cast<float*>(x.data()),0,naive,
            static_cast<float*>(scratch.data()),out.data(),nullptr),"benchmark FFN");};
        auto validate=[&]() {
            ck(cudaDeviceSynchronize(),"benchmark verify sync");uint32_t bits=0;ck(cudaMemcpy(&bits,fault.ptr,4,cudaMemcpyDeviceToHost),"benchmark fault read");
            need(bits==0,"benchmark metadata fault");
            ck(cudaMemcpy(result.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost),"benchmark output read");
            for(float v:result)if(!std::isfinite(v))throw NumericalFailure("benchmark output nonfinite/unwritten");
            for(int i=0;i<3;++i) {
                const size_t start=size_t(residents-1-i)*m*4096;
                compare_values(std::vector<float>(result.begin()+start,result.begin()+start+size_t(m)*4096),
                    columns(reference[i],m,4096,0,4096),"benchmark real E"+std::to_string(experts[i])+" M"+std::to_string(m));
            }
            weights.check("bench weights");x.check("bench x");out.check("bench output");scratch.check("bench scratch");
        };
        launch(0);validate();std::printf("BENCH CORRECT M=%d oracle_experts=0,7,255 all_resident_outputs_finite=yes\n",m);
        bool mutation_check=m==1 || diagnostic_m || profile_m;
#if defined(M26X_EXACT_M) && M26X_EXACT_M
        mutation_check=mutation_check || (m>=5 && m<=7);
#endif
        if(mutation_check) {
            bool detected=false;launch(M26X_NAIVE_NIBBLE_SWAP);
            try{validate();}catch(const NumericalFailure& e){detected=true;std::printf("BENCH NEGATIVE PASS actual nibble mutation: %s\n",e.what());}
            need(detected,"benchmark oracle missed wrong nibble implementation");
        }
        for(int warm=0;warm<10;++warm)launch(0);ck(cudaDeviceSynchronize(),"warmup sync");
        if(profile_m){
            ck(cudaProfilerStart(),"B2 profiler start");launch(0);ck(cudaDeviceSynchronize(),"B2 profile sync");ck(cudaProfilerStop(),"B2 profiler stop");validate();
            std::printf("PROFILE PASS family=B2 M=%d kernels=4; diagnostic counters only, NOT a bandwidth gate\n",m);continue;
        }
        BenchEvent start,stop;std::vector<float> samples;
        for(int sample=0;sample<31;++sample) {
            ck(cudaEventRecord(start.value),"event start");launch(0);ck(cudaEventRecord(stop.value),"event stop");ck(cudaEventSynchronize(stop.value),"timing synchronize");
            float ms=0;ck(cudaEventElapsedTime(&ms,start.value,stop.value),"event elapsed");
            need(std::isfinite(ms) && ms>0,"invalid event sample");samples.push_back(ms);
            std::printf("BENCH SAMPLE residents=%d M=%d sample=%d ms=%.9f\n",residents,m,sample,double(ms));
        }
        validate();std::sort(samples.begin(),samples.end());
        const double gbps=proof::weight_gbps(weight_bytes,samples[15]);const int row=proof::bandwidth_verdict(gbps);
        std::printf("BENCH ROW residents=%d M=%d bytes=%zu median_ms=%.9f min_ms=%.9f max_ms=%.9f effective_GBps=%.6f verdict=%s\n",
            residents,m,weight_bytes,double(samples[15]),double(samples.front()),double(samples.back()),gbps,
            row==0?"TARGET":row==6?"STOP":row==7?"BELOW-TARGET-PIVOT-B1":"INVALID-ABOVE-PEAK");
        if(row==2)throw std::runtime_error("impossible effective bandwidth; investigate byte/time/cache accounting");
        if(row==6)verdict=6;else if(row==7 && !verdict)verdict=7;
    }
    if(profile_m)return 0;
    if(diagnostic_m) {
        std::printf("DIAGNOSTIC FINAL M=5 verdict=%d; not all-M qualification, retain failed row\n",verdict);
        return verdict;
    }
    std::printf("BENCH FINAL residents=%d verdict=%s; no tuning or promotion after failed threshold\n",residents,verdict==0?"TARGET":verdict==6?"STOP":"PIVOT-B1");
    return verdict;
}
