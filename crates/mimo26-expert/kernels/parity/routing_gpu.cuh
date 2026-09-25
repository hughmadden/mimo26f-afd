// Full256-resident, top8 capacity-boundary proof. Synthetic uniquely identified
// matrices, not a real-checkpoint/performance receipt. Included in driver namespace.
m26x_plan routing_plan(const proof::RoutingCase& c) {
    m26x_plan p{};p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;
    p.capacity_class=c.capacity;p.resident_experts=256;p.n_groups=256;p.padded_groups=3;
    p.total_tokens=c.tokens;p.max_m=c.m;p.grouped_bytes=uint64_t(256)*M26X_QUARTER_SLICE_BYTES;
    p.x_bytes=uint64_t(c.tokens)*4096*4;p.out_bytes=p.x_bytes;p.scratch_bytes=uint64_t(c.tokens)*512*2*4;
    return p;
}
void routing_host_plan_selftest() {
    for(int capacity:{256,2048,4096}) {
        const auto c=proof::routing_case(capacity);const auto original=routing_plan(c);
        need(m26x_validate_host_plan(&original,c.ids.data(),c.offsets.data())==0,"full256 host plan refused");
        auto bad=original;auto offsets=c.offsets;++bad.total_tokens;++bad.max_m;++offsets.back();
        need(m26x_validate_host_plan(&bad,c.ids.data(),offsets.data())!=0,"capacity+1 accepted");
        bad=original;bad.resident_experts=255;
        need(m26x_validate_host_plan(&bad,c.ids.data(),c.offsets.data())!=0,"resident255 accepted active ID255");
    }
    std::puts("HOST PLAN PASS full256:3 capacity boundaries and6 overflow/resident-ID refusals");
}
// Capture (never execute) launch APIs to prove malformed AOT manifests enqueue
// no nodes. The existing bypass mutations must enqueue work, powering the test.
void aot_launch_refusals(m26x_plan p,const uint8_t* weights,const float* x,float* scratch,void* out) {
    Device ids(4), offsets(8);
    const int32_t id=0, off[2]={0,1};ids.upload(&id);offsets.upload(off);
    p.n_groups=1;p.padded_groups=0;p.total_tokens=1;p.max_m=1;
    p.expert_ids=static_cast<int32_t*>(ids.ptr);p.group_offsets=static_cast<int32_t*>(offsets.ptr);
    struct Stream {
        cudaStream_t value=nullptr;
        Stream(){ck(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"AOT capture stream");}
        ~Stream(){if(value && cudaStreamDestroy(value)!=cudaSuccess)cleanup_failed=true;}
    } stream;
    for(int api=0;api<2;++api)for(int kind=0;kind<3;++kind)for(int bypass=0;bypass<2;++bypass) {
        auto bad=p;
        if(kind==0)++bad.manifest_sms;
        if(kind==1)++bad.manifest_arch;
        if(kind==2)bad.capacity_class=M26X_CAPACITY_CLASS==256?2048:256;
        const uint32_t mutation=bypass?(kind==2?M26X_NAIVE_AOT_CAPACITY_IGNORED:M26X_NAIVE_AOT_MIXED_GATE):0;
        ck(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal),"begin AOT capture");
        const auto result=api?m26x_expert_ffn_v2(&bad,weights,x,0,mutation,scratch,out,stream.value):
            m26x_grouped_gemm_v2(&bad,weights,x,0,0,mutation,out,stream.value);
        cudaGraph_t graph=nullptr;
        const auto end=cudaStreamEndCapture(stream.value,&graph);
        size_t nodes=0;
        const auto queried=end==cudaSuccess?cudaGraphGetNodes(graph,nullptr,&nodes):end;
        const auto destroyed=graph?cudaGraphDestroy(graph):cudaSuccess;
        ck(end,"end AOT capture");ck(queried,"AOT graph nodes");ck(destroyed,"destroy AOT graph");
        const auto expected=bypass?cudaSuccess:(kind==2?cudaErrorInvalidValue:cudaErrorInvalidDevice);
        need(result==expected,"AOT launch refusal status");
        need(nodes==size_t(bypass?(api?4:1):0),"AOT refusal launched work or bypass negative powerless");
        std::printf("AOT PRELAUNCH PASS class=%d api=%s mismatch=%s bypass=%d status=%d captured_nodes=%zu graph_executed=no\n",
            M26X_CAPACITY_CLASS,api?"FFN":"GEMM",kind==0?"SM-count":kind==1?"arch":"capacity",bypass,int(result),nodes);
    }
}
int routing_gpu(bool ordinal) {
    const auto c=proof::routing_case(M26X_CAPACITY_CLASS);ready();aot_refusals();
    const size_t weight_bytes=size_t(256)*M26X_QUARTER_SLICE_BYTES;
    const size_t tensor_bytes=size_t(c.tokens)*4096*4,scratch_bytes=size_t(c.tokens)*512*2*4;
    Guarded weights(weight_bytes),x(tensor_bytes),out(tensor_bytes),scratch(scratch_bytes);
    Device ids(256*4),offsets(257*4),fault(4);
    const size_t group_values=size_t(c.m)*4096;
    std::vector<float> input(group_values,0),result(group_values),sum(size_t(c.capacity)*8,0);
    for(int expert=0;expert<256;++expert) {
        const auto image=proof::routing_image(expert);
        ck(cudaMemcpy(static_cast<uint8_t*>(weights.data())+size_t(expert)*M26X_QUARTER_SLICE_BYTES,
            image.data(),image.size(),cudaMemcpyHostToDevice),"routing weights");
    }
    for(int group=0;group<256;++group) {
        for(int j=0;j<c.m;++j)input[size_t(j)*4096]=proof::routing_x(c.sources[c.offsets[group]+j]);
        ck(cudaMemcpy(static_cast<float*>(x.data())+size_t(group)*group_values,input.data(),group_values*4,
            cudaMemcpyHostToDevice),"routing input");
    }
    auto p=routing_plan(c);
    p.expert_ids=static_cast<int32_t*>(ids.ptr);p.group_offsets=static_cast<int32_t*>(offsets.ptr);p.fault=static_cast<uint32_t*>(fault.ptr);
    need(m26x_validate_host_plan(&p,c.ids.data(),c.offsets.data())==0,"capacity-boundary host plan refused");
    auto device_ids=c.ids;if(ordinal)for(int g=0;g<256;++g)device_ids[g]=g;
    ids.upload(device_ids.data());offsets.upload(c.offsets.data());ck(cudaMemset(fault.ptr,0,4),"routing fault clear");
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"routing reserve");
    need(free>=size_t(4096)*1024*1024,"routing buffers breached4GiB reserve");
    if(!ordinal)aot_launch_refusals(p,static_cast<const uint8_t*>(weights.data()),
        static_cast<const float*>(x.data()),static_cast<float*>(scratch.data()),out.data());
    ck(m26x_expert_ffn_v2(&p,static_cast<uint8_t*>(weights.data()),static_cast<float*>(x.data()),0,0,
        static_cast<float*>(scratch.data()),out.data(),nullptr),"full256 FFN");
    ck(cudaDeviceSynchronize(),"full256 synchronize");uint32_t fault_bits=0;
    ck(cudaMemcpy(&fault_bits,fault.ptr,4,cudaMemcpyDeviceToHost),"routing fault read");need(fault_bits==0,"routing metadata fault");
    weights.check("full256 weights");x.check("full256 input");out.check("full256 output");scratch.check("full256 scratch");
    double max_abs=0;size_t checked=0;
    for(int group=0;group<256;++group) {
        ck(cudaMemcpy(result.data(),static_cast<float*>(out.data())+size_t(group)*group_values,
            group_values*4,cudaMemcpyDeviceToHost),"routing output group");
        for(int j=0;j<c.m;++j) {
            const int source=c.sources[c.offsets[group]+j];
            for(int row=0;row<4096;++row) {
                const float got=result[size_t(j)*4096+row];
                const double want=proof::routing_oracle(c.ids[group],source,row),error=std::abs(double(got)-want);
                if(!std::isfinite(got) || (row>=8?got!=0:error>1e-5+1e-5*std::abs(want))) {
                    std::ostringstream message;message<<"full256 class="<<c.capacity<<" group="<<group<<" expert="<<c.ids[group]
                        <<" source="<<source<<" row="<<row<<" got="<<got<<" want="<<want<<" ordinal="<<ordinal;
                    throw NumericalFailure(message.str());
                }
                max_abs=std::max(max_abs,error);++checked;
                if(row<8)sum[size_t(source)*8+row]+=got*0.125f;
            }
        }
    }
    // Independent token-first top8 reference, rather than reusing the grouped
    // mapping. Remaining4088 coordinates were required to be exactly zero.
    for(int source=0;source<c.capacity;++source)for(int row=0;row<8;++row) {
        double want=0;for(int k=0;k<8;++k)want+=proof::routing_oracle((source%32)*8+k,source,row)*0.125;
        const float got=sum[size_t(source)*8+row];
        if(!std::isfinite(got) || std::abs(double(got)-want)>1e-5+1e-5*std::abs(want))
            throw NumericalFailure("top8 weighted scatter mismatch");
    }
    need(checked==size_t(c.tokens)*4096,"full256 output coverage");
    std::printf("ROUTING GPU PASS class=%d source_tokens=%d routed_tokens=%d groups=256 resident=256 M=%d outputs=%zu max_abs=%.9e ordinal=%d SYNTHETIC no bandwidth claim\n",
        c.capacity,c.capacity,c.tokens,c.m,checked,max_abs,int(ordinal));return 0;
}
