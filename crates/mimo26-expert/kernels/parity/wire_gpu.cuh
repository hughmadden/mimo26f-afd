// Export actual real-expert GPU FC2 rows for the independent Rust wire probe.
// Binary arrays remain in scratch; only textual receipts enter runs/.
int wire_gpu(const std::string& root,const std::string& inputs,const std::string& output) {
    ready();need(M26X_BAKED_ARCH==121 && M26X_BAKED_SMS==48,"Spark wire proof only");
    const std::array<int,8> experts={0,7,255,1,2,3,4,5};
    const auto fixture=proof::read_json(inputs+"/wire-source.json");
    need(fixture.at("layer").num()==1 && fixture.at("blocks").list().size()==24,"wire source manifest shape");
    std::set<std::string> names;
    for(const auto& b:fixture.at("blocks").list())need(names.insert(b.at("name").str()).second,"duplicate wire source");
    for(int e:experts)for(const char* p:{"gate_proj","up_proj","down_proj"})
        need(names.count("model.layers.1.mlp.experts."+std::to_string(e)+"."+p)==1,"missing wire source");
    const auto index=proof::read_json(root+"/model.safetensors.index.json");
    const auto input=proof::floats(inputs+"/wire-x.f32",8*4096);
    Guarded weights(size_t(32)*M26X_QUARTER_SLICE_BYTES);
    for(int rank=0;rank<4;++rank)for(int j=0;j<8;++j) {
        const auto image=proof::bench_image(root,index,fixture,experts[j],rank);
        ck(cudaMemcpy(static_cast<uint8_t*>(weights.data())+size_t(rank*8+j)*image.size(),image.data(),image.size(),cudaMemcpyHostToDevice),"wire real image upload");
    }
    for(int m:{1,2,4,8}) {
        const int tokens=32*m;const size_t values=size_t(tokens)*4096;
        Guarded x(values*4),out(values*4),scratch(size_t(tokens)*1024*4);
        Device dids(32*4),doff(33*4),fault(4);
        std::vector<int32_t> ids,offsets={0};std::vector<float> grouped_x(values),got(values),ordered(values);
        for(int g=0;g<32;++g) {
            ids.push_back(31-g);offsets.push_back((g+1)*m);
            std::copy_n(input.begin(),size_t(m)*4096,grouped_x.begin()+size_t(g)*m*4096);
        }
        x.upload(grouped_x.data());dids.upload(ids.data());doff.upload(offsets.data());
        m26x_plan p{};p.layout_version=2;p.manifest_arch=M26X_BAKED_ARCH;p.manifest_sms=M26X_BAKED_SMS;p.capacity_class=M26X_CAPACITY_CLASS;
        p.resident_experts=32;p.n_groups=32;p.padded_groups=3;p.total_tokens=tokens;p.max_m=m;
        p.grouped_bytes=weights.bytes;p.x_bytes=x.bytes;p.out_bytes=out.bytes;p.scratch_bytes=scratch.bytes;
        p.expert_ids=static_cast<int32_t*>(dids.ptr);p.group_offsets=static_cast<int32_t*>(doff.ptr);p.fault=static_cast<uint32_t*>(fault.ptr);
        need(m26x_validate_host_plan(&p,ids.data(),offsets.data())==0,"wire host plan");
        for(uint32_t naive:{0u,1u}) {
            if(naive && m!=1)continue;
            ck(cudaMemset(fault.ptr,0,4),"wire fault clear");
            ck(cudaMemset(out.data(),0xff,out.bytes),"wire output poison");
            ck(m26x_expert_ffn_v2(&p,static_cast<uint8_t*>(weights.data()),static_cast<float*>(x.data()),0,naive,
                static_cast<float*>(scratch.data()),out.data(),nullptr),"wire FFN");
            ck(cudaDeviceSynchronize(),"wire GPU sync");uint32_t bits=0;ck(cudaMemcpy(&bits,fault.ptr,4,cudaMemcpyDeviceToHost),"wire fault read");need(bits==0,"wire metadata fault");
            ck(cudaMemcpy(got.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost),"wire GPU readback");
            for(float value:got)if(!std::isfinite(value))throw NumericalFailure("wire GPU output nonfinite/unwritten");
            for(int rank=0;rank<4;++rank)for(int token=0;token<m;++token)for(int j=0;j<8;++j) {
                const size_t from=(size_t(31-(rank*8+j))*m+token)*4096;
                const size_t to=((size_t(rank)*m+token)*8+j)*4096;
                std::copy_n(got.begin()+from,4096,ordered.begin()+to);
            }
            weights.check("wire weights");x.check("wire x");out.check("wire out");scratch.check("wire scratch");
            const auto path=output+"/wire-"+(naive?"naive-":"")+"M"+std::to_string(m)+".f32";
            std::ofstream file(path,std::ios::binary);need(bool(file),"open GPU wire output");
            file.write(reinterpret_cast<const char*>(ordered.data()),ordered.size()*4);file.close();need(bool(file),"write GPU wire output");
            std::printf("WIRE GPU EXPORT M=%d ranks=4 routes=8 values=%zu naive=%u format=rank_token_route_hidden_f32 E-FP32; numerical oracle follows on the dev host\n",m,ordered.size(),naive);
        }
    }
    return 0;
}
