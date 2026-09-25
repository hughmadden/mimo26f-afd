// Primitive proof only. No prepared-layout, FFN, quantizer or timing claim.
#include "mxfp4_ptx.cuh"
#include "mimo26_expert_device.cuh"
#include "fixture_io.h"
#include <array>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#ifndef M26B1_TARGET_ARCH
#define M26B1_TARGET_ARCH 121
#endif
namespace {
bool cleanup_failed=false;
struct Numerical:std::runtime_error {using std::runtime_error::runtime_error;};
struct Cuda:std::runtime_error {using std::runtime_error::runtime_error;};
void need(bool b,const char* text){if(!b)throw std::runtime_error(text);}
void ck(cudaError_t e,const char* text){if(e!=cudaSuccess)throw Cuda(std::string(text)+": "+cudaGetErrorString(e));}
cudaError_t initialize_buffer(void* base,size_t n) {
    const char* mode=std::getenv("MIMO26_B1_INITCHECK");
    if(!mode || std::string(mode)!="1")return cudaMemset(base,0xa5,n+32);
    // Initcheck must see unwritten payloads. Poison redzones, not allocations.
    const auto first=cudaMemset(base,0xa5,16);
    return first==cudaSuccess?cudaMemset(static_cast<uint8_t*>(base)+16+n,0xa5,16):first;
}
__global__ void initcheck_read_probe(const uint32_t* x,uint32_t* out) { *out=*x; }
struct Buffer {
    void* base=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory");need(n+32<=free && free-n-32>=size_t(8)*1024*1024*1024,"8 GiB device reserve");ck(cudaMalloc(&base,n+32),"allocate");const auto error=initialize_buffer(base,n);if(error!=cudaSuccess){if(cudaFree(base)!=cudaSuccess)cleanup_failed=true;base=nullptr;ck(error,"poison");}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(base && cudaFree(base)!=cudaSuccess)cleanup_failed=true;}
    void* data(){return static_cast<uint8_t*>(base)+16;}
    void upload(const void* p){ck(cudaMemcpy(data(),p,bytes,cudaMemcpyHostToDevice),"upload");}
    void download(void* p){ck(cudaMemcpy(p,data(),bytes,cudaMemcpyDeviceToHost),"download");}
    void guard(){uint32_t lo[4],hi[4];ck(cudaMemcpy(lo,base,16,cudaMemcpyDeviceToHost),"guard lo");ck(cudaMemcpy(hi,static_cast<uint8_t*>(data())+bytes,16,cudaMemcpyDeviceToHost),"guard hi");for(int i=0;i<4;++i)need(lo[i]==0xa5a5a5a5u && hi[i]==0xa5a5a5a5u,"redzone");}
};
constexpr uint8_t e4[]={0,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc};
constexpr double mag[]={0,.5,1,1.5,2,3,4,6};
double value(int n){return n&8?-mag[n&7]:mag[n&7];}
int acode(int m,int k){return (m*3+k*5+(k/32)*7)%16;}
int bcode(int n,int k){return (n*7+k*3+(k/32)*5)%16;}
int ascale(int m,int kb){return 125+(m+kb)%5;}
int bscale(int n,int kb){return 124+(n+2*kb)%7;}
struct Fixture {
    std::array<uint32_t,16*32> a{};std::array<uint32_t,8*16> b{};
    std::array<uint32_t,16> sa{};std::array<uint32_t,8> sb{};
    Fixture(){
        for(int m=0;m<16;++m){for(int k=0;k<128;++k)a[m*32+k/4]|=uint32_t(e4[acode(m,k)])<<(8*(k%4));for(int kb=0;kb<4;++kb)sa[m]|=uint32_t(ascale(m,kb))<<(8*kb);}
        for(int n=0;n<8;++n){for(int k=0;k<128;++k)b[n*16+k/8]|=uint32_t(bcode(n,k))<<(4*(k%8));for(int kb=0;kb<4;++kb)sb[n]|=uint32_t(bscale(n,kb))<<(8*kb);}
    }
};
std::array<float,128> reference(unsigned naive=0){
    std::array<float,128> result{};
    for(int m=0;m<16;++m)for(int n=0;n<8;++n){double sum=0;
        for(int k=0;k<128;++k){const int kb=naive==2?0:k/32;const int bn=naive==4?(n+1)%8:n;
            const auto av=std::ldexp(value(acode(m,k)),ascale(m,kb)-127);
            const auto bv=std::ldexp(value(bcode(n,naive==1?(k^1):k)),bscale(bn,kb)-127);
            sum+=av*bv;
        }result[m*8+n]=float(sum);
    }return result;
}
void host_test(){
    const Fixture f;const auto expected=reference();bool nonzero=false;
    for(int m=0;m<16;++m)for(int k=0;k<128;++k)need(((f.a[m*32+k/4]>>(8*(k%4)))&255)==e4[acode(m,k)],"A packing");
    for(int n=0;n<8;++n)for(int k=0;k<128;++k)need(((f.b[n*16+k/8]>>(4*(k%8)))&15)==unsigned(bcode(n,k)),"B packing");
    for(auto x:expected)nonzero|=x!=0;need(nonzero,"degenerate fixture");
    for(unsigned flag:{1u,2u,4u})need(reference(flag)!=expected,"ineffective mutation");
    std::puts("HOST PASS B1 fixture: asymmetric 16x8x128, four K32 blocks, A/B packing, nibble/scale-byte/scale-lane detectors");
}
__global__ void containers(uint32_t* words,uint32_t* decoded,unsigned naive){
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;if(pair>=4096)return;
    const int code=pair/256,scale=pair%256;uint32_t packed=0;
    for(int j=0;j<8;++j)packed|=uint32_t((code+j*5)%16)<<(4*j);
    if(naive==1)packed=((packed&0x0f0f0f0fu)<<4)|((packed&0xf0f0f0f0u)>>4);
    const auto spread=m26b1::e2m1_containers(packed);words[pair*2]=spread.x;words[pair*2+1]=spread.y;
    for(int j=0;j<8;++j){const auto byte=((j<4?spread.x:spread.y)>>(8*(j%4)))&255;
        decoded[pair*8+j]=__float_as_uint(m26x::decode(byte>>2,scale,0));}
}
__global__ void fragment(const uint32_t* a,const uint32_t* b,const uint32_t* sa,const uint32_t* sb,float* out,unsigned naive){
    const int lane=threadIdx.x,g=lane/4,c=lane%4;float d[4]={0,0,0,0};
    const uint32_t sfa=c<2?sa[g+c*8]:0;
    const uint32_t sfb=sb[naive==4?(g+1)%8:g];
    #pragma unroll
    for(int kb=0;kb<4;++kb){
        const uint32_t a0=a[g*32+kb*8+c*2],a2=a[g*32+kb*8+c*2+1];
        const uint32_t a1=a[(g+8)*32+kb*8+c*2],a3=a[(g+8)*32+kb*8+c*2+1];
        uint32_t packed=b[g*16+kb*4+c];
        if(naive==1)packed=((packed&0x0f0f0f0fu)<<4)|((packed&0xf0f0f0f0u)>>4);
        const auto w=m26b1::e2m1_containers(packed);
        if(naive==2 || kb==0)m26b1::mma_e4m3_e2m1<0,0>(d,a0,a1,a2,a3,w.x,w.y,sfa,sfb);
        else if(kb==1)m26b1::mma_e4m3_e2m1<1,1>(d,a0,a1,a2,a3,w.x,w.y,sfa,sfb);
        else if(kb==2)m26b1::mma_e4m3_e2m1<2,2>(d,a0,a1,a2,a3,w.x,w.y,sfa,sfb);
        else m26b1::mma_e4m3_e2m1<3,3>(d,a0,a1,a2,a3,w.x,w.y,sfa,sfb);
    }
    out[g*8+2*c]=d[0];out[g*8+2*c+1]=d[1];out[(g+8)*8+2*c]=d[2];out[(g+8)*8+2*c+1]=d[3];
}
void gpu_ready(){
    const char* opt=std::getenv("MIMO26_BUILDER_GPU");need(opt && std::string(opt)=="1","GPU opt-in required");
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0),"identity");
    need(p.major*10+p.minor==M26B1_TARGET_ARCH && p.multiProcessorCount==48,"wrong B1 target/SM count");
    std::printf("GPU %s arch=%d SMs=%d B1 E4M3xE2M1 primitive only; no FFN/performance qualification\n",p.name,p.major*10+p.minor,p.multiProcessorCount);
}
int gpu_test(unsigned naive){
    gpu_ready();
    Buffer dw(4096*2*4),dd(4096*8*4);
    containers<<<16,256>>>(static_cast<uint32_t*>(dw.data()),static_cast<uint32_t*>(dd.data()),naive);
    ck(cudaGetLastError(),"container launch");ck(cudaDeviceSynchronize(),"container sync");
    std::vector<uint32_t> words(4096*2),decoded(4096*8);dw.download(words.data());dd.download(decoded.data());dw.guard();dd.guard();
    size_t bad=0;const double limit=std::numeric_limits<float>::max();
    for(int pair=0;pair<4096;++pair)for(int j=0;j<8;++j){const int n=(pair/256+j*5)%16;
        if(((words[pair*2+j/4]>>(8*(j%4)))&255)!=unsigned(n<<2))++bad;
        const double v=std::ldexp(value(n),std::min(pair%256,254)-127);
        const float want=float(std::max(-limit,std::min(limit,v)));uint32_t bits;std::memcpy(&bits,&want,4);
        if(decoded[pair*8+j]!=bits)++bad;
    }
    std::printf("%s B1 containers flag=%u mismatches=%zu; 32768 lane-byte checks plus 32768 conforming decoder checks (all 4096 code/scale pairs); not exceptional-scale MMA proof\n",bad?"ORACLE_MISMATCH":"PASS",naive,bad);
    const Fixture f;Buffer da(sizeof(f.a)),db(sizeof(f.b)),dsa(sizeof(f.sa)),dsb(sizeof(f.sb)),out(128*4);
    da.upload(f.a.data());db.upload(f.b.data());dsa.upload(f.sa.data());dsb.upload(f.sb.data());
    fragment<<<1,32>>>(static_cast<uint32_t*>(da.data()),static_cast<uint32_t*>(db.data()),static_cast<uint32_t*>(dsa.data()),static_cast<uint32_t*>(dsb.data()),static_cast<float*>(out.data()),naive);
    ck(cudaGetLastError(),"MMA launch");ck(cudaDeviceSynchronize(),"MMA sync");
    std::array<float,128> got{};out.download(got.data());out.guard();da.guard();db.guard();dsa.guard();dsb.guard();
    const auto want=reference();size_t mma_bad=0;double max_abs=0;
    for(size_t i=0;i<got.size();++i){const double error=std::abs(double(got[i])-want[i]);max_abs=std::max(max_abs,error);
        if(!std::isfinite(got[i]) || error>1e-5+1e-5*std::abs(want[i]))++mma_bad;}
    std::printf("%s B1 MMA flag=%u mismatches=%zu/128 maxabs=%.12g ordinary_scales=124..130 four_K32_blocks; selectors A/B=0..3\n",mma_bad?"ORACLE_MISMATCH":"PASS",naive,mma_bad,max_abs);
    return bad || mma_bad?3:0;
}
}
namespace {
__global__ void real_containers(const uint32_t* words,const uint8_t* scales,const uint8_t* positions,uint32_t* output,int count,unsigned naive){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint32_t packed=words[i];if(naive==1)packed=((packed&0x0f0f0f0fu)<<4)|((packed&0xf0f0f0f0u)>>4);
    const auto spread=m26b1::e2m1_containers(packed);const int j=positions[i];
    const uint32_t nibble=(((j<4?spread.x:spread.y)>>(8*(j%4)))&255)>>2;
    output[i]=__float_as_uint(m26x::decode(nibble,scales[i],0));
}
int real_test(const std::string& root,const std::string& fixture_path,unsigned naive){
    gpu_ready();const auto f=proof::read_json(fixture_path);proof::validate_fixture(f);
    const auto index=proof::read_json(root+"/model.safetensors.index.json");
    std::vector<uint32_t> words,expected;std::vector<uint8_t> scales,positions;
    for(const auto& block:f.at("blocks").list()){
        const auto name=block.at("name").str();const auto& shape=block.at("shape").list();
        const int rows=int(shape[0].num()),cols=int(shape[1].num());
        const auto w=proof::tensor(root,index,name+".weight",rows,cols/2,block.at("weight_sha256").str());
        const auto s=proof::tensor(root,index,name+".weight_scale",rows,cols/32,block.at("scale_sha256").str());
        const auto& where=block.at("positions").list();const auto& bits=block.at("expected_f32_bits").list();
        for(size_t i=0;i<where.size();++i){const int row=int(where[i].list()[0].num()),k=int(where[i].list()[1].num());
            uint32_t word=0;std::memcpy(&word,w.data()+size_t(row)*(cols/2)+(k/8)*4,4);words.push_back(word);
            scales.push_back(s[size_t(row)*(cols/32)+k/32]);positions.push_back(uint8_t(k%8));
            expected.push_back(uint32_t(std::stoul(bits[i].str(),nullptr,16)));
        }
        std::printf("B1 SOURCE PASS %s weight=%s scale=%s\n",name.c_str(),block.at("weight_sha256").str().c_str(),block.at("scale_sha256").str().c_str());
    }
    need(words.size()==55296,"real container sample count");
    Buffer dw(words.size()*4),ds(scales.size()),dp(positions.size()),out(expected.size()*4);
    dw.upload(words.data());ds.upload(scales.data());dp.upload(positions.data());
    real_containers<<<int((words.size()+255)/256),256>>>(static_cast<uint32_t*>(dw.data()),static_cast<uint8_t*>(ds.data()),static_cast<uint8_t*>(dp.data()),static_cast<uint32_t*>(out.data()),int(words.size()),naive);
    ck(cudaGetLastError(),"real container launch");ck(cudaDeviceSynchronize(),"real container sync");
    std::vector<uint32_t> got(expected.size());out.download(got.data());dw.guard();ds.guard();dp.guard();out.guard();
    size_t bad=0;for(size_t i=0;i<got.size();++i)bad+=got[i]!=expected[i];
    std::printf("%s B1 real container K1 flag=%u mismatches=%zu/55296 hashes=54; conforming decoder after lossless container recovery, not exceptional-scale MMA\n",bad?"ORACLE_MISMATCH":"PASS",naive,bad);
    return bad?3:0;
}
}
#include "compute_test.cuh"
#include "rank_compute_test.cuh"
#include "ffn_test.cuh"
#include "bench_connected.cuh"
int main(int argc,char** argv){int result=0;try{
    if(argc==2 && std::string(argv[1])=="--selftest"){host_test();compute_host_test();rank_host_test();bench_connected_host();}
    else if(argc==2 && std::string(argv[1])=="--initcheck-negative"){
        const char* mode=std::getenv("MIMO26_B1_INITCHECK");need(mode && std::string(mode)=="1","uninitialized-payload mode required");
        gpu_ready();Buffer x(4),out(4);
        initcheck_read_probe<<<1,1>>>(static_cast<const uint32_t*>(x.data()),static_cast<uint32_t*>(out.data()));
        ck(cudaGetLastError(),"initcheck negative launch");ck(cudaDeviceSynchronize(),"initcheck negative completion");
        std::puts("INITCHECK NEGATIVE executed deliberate uninitialized payload read; sanitizer must reject");
    }else if(argc==2 && std::string(argv[1])=="--all-math"){
        result=compute_test(0);if(result==0 && !cleanup_failed)result=rank_test(0);
    }else if(argc==3 && std::string(argv[1])=="--gpu"){
        const std::string f=argv[2];need(f=="0" || f=="1" || f=="2" || f=="4","bad mutation");result=gpu_test(unsigned(std::stoul(f)));
    }else if(argc==3 && std::string(argv[1])=="--compute"){
        const std::string f=argv[2];need(f=="0" || f=="1" || f=="2" || f=="4" || f=="8","bad compute mutation");result=compute_test(unsigned(std::stoul(f)));
    }else if(argc==3 && std::string(argv[1])=="--rank"){
        const std::string f=argv[2];need(f=="0" || f=="1" || f=="2" || f=="4","bad rank mutation");result=rank_test(unsigned(std::stoul(f)));
    }else if(argc==5 && std::string(argv[1])=="--profile-connected"){
        const std::string m=argv[4];need(m=="1" || m=="8","bad B1 profile M");result=bench_connected(argv[2],argv[3],std::stoi(m));
    }else if(argc==4 && std::string(argv[1])=="--bench-connected"){
        result=bench_connected(argv[2],argv[3]);
    }else if(argc==7 && std::string(argv[1])=="--ffn"){
        result=ffn_test(argv[2],argv[3],argv[4],std::stoi(argv[5]),unsigned(std::stoul(argv[6])));
    }else if(argc==5 && std::string(argv[1])=="--real"){
        const std::string f=argv[4];need(f=="0" || f=="1","bad real mutation");result=real_test(argv[2],argv[3],unsigned(std::stoul(f)));
    }else throw std::runtime_error("use --selftest, --all-math, --gpu 0|1|2|4, --compute 0|1|2|4|8, --rank 0|1|2|4, or --real weights fixture 0|1");
}catch(const Numerical& e){std::fprintf(stderr,"ORACLE_MISMATCH %s\n",e.what());result=3;}
catch(const Cuda& e){std::fprintf(stderr,"CUDA_FAILURE %s\n",e.what());result=4;}
catch(const std::exception& e){std::fprintf(stderr,"INPUT_FAILURE %s\n",e.what());result=2;}
return cleanup_failed?4:result;}
