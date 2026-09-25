// Real connected v1 harness. Uses published lead units, never a replacement.
#include "bench_io.h"
#include "b1/prepare.cu"
#include "b1/group_plan.cu"
#include "b1/quant_v1.cu"
#include "b1/route_reduce.cu"
#include "b1/connected.cu"
namespace {
template<class T> T* ptr(Buffer& b){return static_cast<T*>(b.data());}
void zero_faults(Buffer& b,const char* message){std::vector<uint32_t> f(b.bytes/4);b.download(f.data());for(auto v:f)if(v)throw Numerical(message);b.guard();}
template<class T> void dump(const std::string& root,const char* name,const std::vector<T>& v){
    std::ofstream out(root+"/"+name,std::ios::binary);need(bool(out),"open dump");out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));need(bool(out),"write dump");
}
int ffn_test(const std::string& weights_root,const std::string& input,const std::string& output,int m,unsigned naive){
    need(m>=1 && m<=8 && naive<=1,"FFN arguments");gpu_ready();
    const char* init_mode=std::getenv("MIMO26_B1_INITCHECK");
    if(init_mode && std::string(init_mode)=="1")std::puts("INITCHECK PAYLOAD initial_poison=no redzones=yes");
    const auto source=proof::read_json(input+"/b1-source.json");
    const auto index=proof::read_json(weights_root+"/model.safetensors.index.json");
    const int experts[8]={0,7,255,1,2,3,4,5};
    auto xp=proof::read(input+"/b1-x-payload.u8"),xs=proof::read(input+"/b1-x-scales.u8"),wb=proof::read(input+"/b1-weights.f32");
    need(xp.size()==8*4096 && xs.size()==8*128 && wb.size()==8*8*4,"FFN reference input sizes");
    xp.resize(m*4096);xs.resize(m*128);wb.resize(m*8*4);
    Buffer dx(xp.size()),ds(xs.size()),dw(wb.size()),di(m*8*4),dr(256),dslots(256*4);
    dx.upload(xp.data());ds.upload(xs.data());dw.upload(wb.data());
    std::vector<int32_t> ids(m*8),slots(256,-1);std::vector<uint8_t> resident(256,0);
    for(int s=0;s<8;++s){resident[experts[s]]=1;slots[experts[s]]=s;for(int row=0;row<m;++row)ids[row*8+s]=experts[s];}
    di.upload(ids.data());dr.upload(resident.data());dslots.upload(slots.data());
    constexpr int cap=256*8;
    Buffer inv(cap*4),original(cap*4),gw(cap*4),groups(cap*sizeof(m26b1::Group)),count(4),fault(4);
    m26b1::PlanInput pi{256,uint32_t(m),ptr<int32_t>(di),ptr<float>(dw),uint64_t(m*8),ptr<uint8_t>(dr),256};
    m26b1::PlanStorage ps{ptr<int32_t>(inv),ptr<int32_t>(original),ptr<float>(gw),ptr<m26b1::Group>(groups),ptr<uint32_t>(count),ptr<uint32_t>(fault),cap,cap};
    ck(m26b1::plan_async(pi,ps,0),"plan launch");ck(cudaDeviceSynchronize(),"plan sync");zero_faults(fault,"plan fault");
    uint32_t ng=0;count.download(&ng);need(ng==8,"unexpected active group count");
    // Download only active metadata; do not rely on allocation poisoning to
    // make unused capacity readable in the initcheck cell.
    std::vector<m26b1::Group> hg(ng);std::vector<int32_t> ho(m*8);
    ck(cudaMemcpy(hg.data(),groups.data(),hg.size()*sizeof(hg[0]),cudaMemcpyDeviceToHost),"active groups download");
    ck(cudaMemcpy(ho.data(),original.data(),ho.size()*sizeof(ho[0]),cudaMemcpyDeviceToHost),"active routes download");
    std::vector<float> gate(4*m*8*512),up(gate.size()),h(gate.size()),partial(4*m*8*4096),rank(4*m*4096),decoded(rank.size()),wire(m*4096,0);
    std::vector<uint8_t> mp(gate.size()),ms(4*m*8*16);
    const size_t mid_n=size_t(ng)*16*512;
    Buffer canonical(m26b1::image_bytes),prepared(8*m26b1::image_bytes),dg(mid_n*4),du(mid_n*4),dh(mid_n*4),dq(mid_n),dqs(mid_n/32),qfault(mid_n/32*4),f1(ng*4*4),f2(ng*32*4),dy(size_t(m)*8*4096*4),dpre(size_t(m)*4096*4),dret(size_t(m)*4096*2),rfault(4);
    for(int r=0;r<4;++r){
        m26b1::PreparedInfo prepared_info{};
        for(int s=0;s<8;++s){const auto image=proof::bench_image(weights_root,index,source,experts[s],r);canonical.upload(image.data());
            ck(m26b1::prepare_async({2,4096,512,uint32_t(r),m26b1::image_bytes},ptr<uint8_t>(canonical),canonical.bytes,
                ptr<uint8_t>(prepared)+s*m26b1::image_bytes,m26b1::image_bytes,&prepared_info,0),"prepare launch");
            ck(cudaDeviceSynchronize(),"prepare before canonical reuse");}
        const m26b1::Pool pool{prepared_info,prepared.bytes,8};
        m26b1::connected_fc1<128><<<dim3(4,ng),128>>>(pool,ptr<uint8_t>(prepared),ptr<int32_t>(dslots),ps,ptr<uint32_t>(dx),ptr<uint8_t>(ds),ptr<float>(dh),ptr<float>(dg),ptr<float>(du),ptr<uint32_t>(f1));
        ck(cudaGetLastError(),"connected FC1");
        ck(m26b1::quantize_async({ptr<float>(dh),mid_n},{ptr<uint8_t>(dq),ptr<uint8_t>(dqs),ptr<uint32_t>(qfault),mid_n,mid_n/32,mid_n/32},0),"quantizer launch");
        m26b1::connected_fc2_fp8<128><<<dim3(32,ng),128>>>(pool,ptr<uint8_t>(prepared),ptr<int32_t>(dslots),ps,ptr<uint8_t>(dq),ptr<uint8_t>(dqs),ptr<uint32_t>(qfault),ptr<float>(dy),m*8,ptr<uint32_t>(f2),ptr<float>(dw),naive);
        ck(cudaGetLastError(),"connected FC2");
        ck(m26b1::reduce_async({uint32_t(m),ptr<float>(dy),ptr<float>(dw),uint64_t(m)*8*4096,uint64_t(m)*8},
            {ptr<float>(dpre),ptr<uint16_t>(dret),uint64_t(m)*4096,uint64_t(m)*4096,ptr<uint32_t>(rfault)},0),"route reduce");
        ck(cudaDeviceSynchronize(),"connected completion");
        zero_faults(f1,"FC1 staging fault");zero_faults(qfault,"quantizer fault");zero_faults(f2,"FC2 staging fault");zero_faults(rfault,"route reduction fault");
        std::vector<float> gg(mid_n),uu(mid_n),hh(mid_n);std::vector<uint8_t> qp(mid_n),qs(mid_n/32);std::vector<uint16_t> ret(m*4096);
        dg.download(gg.data());du.download(uu.data());dh.download(hh.data());dq.download(qp.data());dqs.download(qs.data());
        dy.download(partial.data()+size_t(r)*m*8*4096);dpre.download(rank.data()+size_t(r)*m*4096);dret.download(ret.data());
        for(uint32_t gi=0;gi<ng;++gi)for(int row=0;row<16;++row){
            if(row>=hg[gi].rows){for(int k=0;k<512;++k)need(gg[(gi*16+row)*512+k]==0 && uu[(gi*16+row)*512+k]==0 && hh[(gi*16+row)*512+k]==0,"padded intermediate");continue;}
            const int route=ho[hg[gi].route_base+row];need(route>=0 && route<m*8,"scatter metadata");
            const size_t from=(gi*16+row)*512,to=(size_t(r)*m*8+route)*512;
            std::copy_n(gg.data()+from,512,gate.data()+to);std::copy_n(uu.data()+from,512,up.data()+to);std::copy_n(hh.data()+from,512,h.data()+to);std::copy_n(qp.data()+from,512,mp.data()+to);std::copy_n(qs.data()+from/32,16,ms.data()+to/32);
        }
        for(size_t i=0;i<ret.size();++i){const uint32_t bits=uint32_t(ret[i])<<16;float v;std::memcpy(&v,&bits,4);decoded[size_t(r)*m*4096+i]=v;wire[i]+=v;}
        for(Buffer* b:{&canonical,&prepared,&dg,&du,&dh,&dq,&dqs,&dy,&dpre,&dret})b->guard();
        std::printf("PASS B1 connected execution rank=%d M=%d flag=%u; comparison pending, not numerical qualification\n",r,m,naive);
    }
    dx.download(xp.data());ds.download(xs.data());dw.download(wb.data());
    dump(output,"b1-x-payload.u8",xp);dump(output,"b1-x-scales.u8",xs);dump(output,"b1-weights.f32",wb);
    dump(output,"b1-gate.f32",gate);dump(output,"b1-up.f32",up);dump(output,"b1-h.f32",h);dump(output,"b1-mid_payload.u8",mp);dump(output,"b1-mid_scales.u8",ms);
    dump(output,"b1-partial.f32",partial);dump(output,"b1-rank.f32",rank);dump(output,"b1-return-decoded.f32",decoded);dump(output,"b1-wire.f32",wire);
    for(Buffer* b:{&dx,&ds,&dw,&di,&dr,&dslots,&inv,&original,&gw,&groups,&count,&fault})b->guard();return 0;
}
} // namespace
