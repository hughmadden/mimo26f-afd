// Real checkpoint rank0 packing and pre-committed effective-bandwidth contract.
#pragma once
#include "fixture_io.h"
#include "mimo26_slice_layout.h"
#include <cmath>
#include <cstring>
namespace proof {
inline int bandwidth_verdict(double gbps) {
    if(!std::isfinite(gbps) || gbps<=0 || gbps>273)return 2;
    if(gbps<163.8)return 6; // STOP, not an optimization invitation
    if(gbps<191.1)return 7; // BELOW-TARGET: pivot B1
    return 0;
}
inline double weight_gbps(size_t bytes,double milliseconds) {
    need(std::isfinite(milliseconds) && milliseconds>0,"invalid CUDA event interval");
    return double(bytes)/(milliseconds*1e6); // decimal GB/s, weights+scales once
}
inline std::vector<int> bench_experts(int count) {
    need(count==256 || count==57 || count==58,"resident count must be256/57/58");
    std::vector<int> result={0,7,255};
    for(int e=1;int(result.size())<count;++e)if(e!=7 && e!=255)result.push_back(e);
    return result;
}
inline void copy_rank(std::vector<uint8_t>& image,int p,const std::vector<uint8_t>& w,const std::vector<uint8_t>& s,int rank=0) {
    need(image.size()==3342336 && p>=0 && p<3 && rank>=0 && rank<4,"rank destination");
    need(w.size()==4194304 && s.size()==262144,"rank0 full source sizes");
    const size_t wo=p==0?0:p==1?1114112:2228224,so=p==0?1048576:p==1?2162688:3276800;
    if(p<2) {
        std::copy_n(w.begin()+rank*1048576,1048576,image.begin()+wo);
        std::copy_n(s.begin()+rank*65536,65536,image.begin()+so);
    } else {
        for(size_t row=0;row<4096;++row) {
            std::copy_n(w.begin()+row*1024+rank*256,256,image.begin()+wo+row*256);
            std::copy_n(s.begin()+row*64+rank*16,16,image.begin()+so+row*16);
        }
    }
}
inline std::vector<uint8_t> bench_image(const std::string& root,const Json& index,const Json& fixture,int expert,int rank=0) {
    std::vector<uint8_t> image(3342336);
    const char* projections[]={"gate_proj","up_proj","down_proj"};
    for(int p=0;p<3;++p) {
        const std::string name="model.layers.1.mlp.experts."+std::to_string(expert)+"."+projections[p];
        const int rows=p==2?4096:2048,cols=p==2?2048:4096;
        const auto w=tensor_bytes(root,index,name+".weight",rows,cols/2);
        const auto s=tensor_bytes(root,index,name+".weight_scale",rows,cols/32);
        const auto wh=sha256(w),sh=sha256(s);bool pinned=false;
        for(const auto& b:fixture.at("blocks").list())if(b.at("name").str()==name) {
            need(wh==b.at("weight_sha256").str() && sh==b.at("scale_sha256").str(),"benchmark pinned source hash mismatch");pinned=true;
        }
        if(expert==0 || expert==7 || expert==255)need(pinned,"missing pinned benchmark expert");
        std::printf("BENCH SOURCE %s weight=%s scale=%s expected_hash=%s\n",name.c_str(),wh.c_str(),sh.c_str(),pinned?"verified":"not-pinned-recorded-only");
        copy_rank(image,p,w,s,rank);
    }
    return image;
}
inline void bench_selftest() {
    need(bandwidth_verdict(163.799)==6 && bandwidth_verdict(163.8)==7 && bandwidth_verdict(191.099)==7 && bandwidth_verdict(191.1)==0,"STOP/pivot boundaries");
    need(bandwidth_verdict(273.01)==2 && bandwidth_verdict(NAN)==2,"impossible timing accepted");
    need(std::abs(weight_gbps(3342336,1)-3.342336)<1e-12,"GB/s units");
    for(double ms:{0.,-1.,double(NAN)}) {bool refused=false;try{weight_gbps(1,ms);}catch(const std::exception&){refused=true;}need(refused,"bad timer accepted");}
    for(int n:{57,58,256}) {const auto ids=bench_experts(n);need(std::set<int>(ids.begin(),ids.end()).size()==size_t(n),"duplicate real expert");}
    std::vector<uint8_t> w(4194304),s(262144),image(3342336);
    for(size_t i=0;i<w.size();++i)w[i]=uint8_t((i/1024+i%1024)%251);
    for(size_t i=0;i<s.size();++i)s[i]=uint8_t((i/64+3*(i%64))%253);
    for(int rank=0;rank<4;++rank)for(int p=0;p<3;++p) {
        copy_rank(image,p,w,s,rank);
        const size_t wo=p==0?0:p==1?1114112:2228224,so=p==0?1048576:p==1?2162688:3276800;
        const size_t rows=p==2?4096:512,pitch=p==2?256:2048,scale_pitch=p==2?16:128;
        for(size_t row=0;row<rows;++row) {
            for(size_t col=0;col<pitch;++col)need(image[wo+row*pitch+col]==w[(p==2?row*1024+rank*256:(rank*512+row)*2048)+col],"payload rectangle");
            for(size_t col=0;col<scale_pitch;++col)need(image[so+row*scale_pitch+col]==s[(p==2?row*64+rank*16:(rank*512+row)*128)+col],"scale rectangle");
        }
    }
    for(int rank:{-1,4}) {bool refused=false;try{copy_rank(image,0,w,s,rank);}catch(const std::exception&){refused=true;}need(refused,"invalid rank accepted");}
    std::puts("HOST PASS benchmark/packing: decimal bytes/time, STOP/pivot/peak boundaries, bad timers, unique real IDs, all 4 rank rectangle bytes, invalid ranks refused");
}
} // namespace proof
