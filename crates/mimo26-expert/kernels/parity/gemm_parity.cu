/* Standalone nvcc-only proof driver. Full checkpoint tensors, never synthesized
 * from expected bits; both source hashes checked. No Rust/Python on remote host.
 * Host-only modes never call CUDA. GPU modes require explicit opt-in using
 * the legacy MIMO26_BUILDER_GPU flag; latest user policy controls the operator.
 */
#include "mimo26_expert_kernels.h"
#include "fixture_io.h"
#include "tp4_io.h"
#include "routing_proof.h"
#include "bench_io.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using proof::need;
namespace {
bool cleanup_failed=false;
struct CudaFailure:std::runtime_error { using std::runtime_error::runtime_error; };
void ck(cudaError_t e,const char* operation) {
    if(e!=cudaSuccess) throw CudaFailure(std::string(operation)+": "+cudaGetErrorString(e));
}
struct Device {
    void* ptr=nullptr; size_t bytes;
    explicit Device(size_t n):bytes(n) {
        size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"allocation preflight");
        need(n<=free && free-n>=size_t(4098)*1024*1024,"allocation would breach4GiB reserve plus allocator slack");
        ck(cudaMalloc(&ptr,n),"cudaMalloc");
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device() { if(ptr) {const auto e=cudaFree(ptr);if(e!=cudaSuccess){cleanup_failed=true;std::fprintf(stderr,"cudaFree: %s\n",cudaGetErrorString(e));}} }
    void upload(const void* source) {ck(cudaMemcpy(ptr,source,bytes,cudaMemcpyHostToDevice),"upload");}
};
void ready() {
    const char* permit=std::getenv("MIMO26_BUILDER_GPU");need(permit && std::string(permit)=="1","GPU modes require explicit authorized opt-in");
    int32_t arch=0,sms=0;ck(m26x_device_identity(&arch,&sms),"identity");
    ck(m26x_check_aot(M26X_BAKED_ARCH,M26X_BAKED_SMS,M26X_CAPACITY_CLASS,0),"baked/live AOT identity");
    // Full checkpoint decode needs <40MiB; reserve256MiB for buffers/module
    // overhead in addition to the4GiB that must remain free for other work.
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory guard");
    need(free>=size_t(4352)*1024*1024,"need4GiB free reserve plus256MiB working headroom");
    std::printf("GPU arch=%d physical_sms=%d baked_arch=%d baked_sms=%d class=%d free_bytes=%zu dtype=f32\n",
        arch,sms,M26X_BAKED_ARCH,M26X_BAKED_SMS,M26X_CAPACITY_CLASS,free);
}
uint32_t flag(const char* s) {
    const std::string text=s;
    need(text=="0" || text=="1" || text=="2" || text=="4" || text=="128","only individual unpack mutations supported");
    return uint32_t(std::stoul(text));
}
std::vector<uint32_t> gpu_unpack(const std::vector<uint8_t>& w,const std::vector<uint8_t>& s,int rows,int cols,uint32_t naive) {
    const size_t count=size_t(rows)*cols;Device dw(w.size()),ds(s.size()),out((count+8)*4);
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"post-allocation reserve");
    need(free>=size_t(4096)*1024*1024,"4GiB reserve breached; release buffers without launching");
    dw.upload(w.data());ds.upload(s.data());ck(cudaMemset(out.ptr,0xa5,out.bytes),"output poison");
    auto* result=static_cast<uint32_t*>(out.ptr)+4;
    ck(m26x_unpack_matrix(static_cast<uint8_t*>(dw.ptr),dw.bytes,static_cast<uint8_t*>(ds.ptr),ds.bytes,
        rows,cols,naive,reinterpret_cast<float*>(result),count*4,nullptr),"unpack launch");
    ck(cudaDeviceSynchronize(),"unpack synchronize");
    std::vector<uint32_t> words(count+8);ck(cudaMemcpy(words.data(),out.ptr,out.bytes,cudaMemcpyDeviceToHost),"download");
    for(size_t i=0;i<4;++i)need(words[i]==0xa5a5a5a5u && words[count+4+i]==0xa5a5a5a5u,"output guard overwritten");
    return std::vector<uint32_t>(words.begin()+4,words.end()-4);
}
#include "tp4_gpu.cuh"
#include "routing_gpu.cuh"
#include "bench_gpu.cuh"
#include "wire_gpu.cuh"
int decoder_check(uint32_t naive) {
    ready();const int rows=16,cols=8192;
    std::vector<uint8_t> w(rows*cols/2),s(rows*cols/32);
    for(int n=0;n<16;++n) {
        std::fill(w.begin()+n*(cols/2),w.begin()+(n+1)*(cols/2),uint8_t(n|(n<<4)));
        for(int b=0;b<256;++b)s[n*256+b]=uint8_t(b);
    }
    const auto got=gpu_unpack(w,s,rows,cols,naive);
    const double lut[]={0,.5,1,1.5,2,3,4,6};const double max=std::numeric_limits<float>::max();size_t bad=0;
    for(int n=0;n<16;++n) for(int k=0;k<cols;++k) {
        double v=lut[n&7];if(n&8)v=-v;
        v=std::ldexp(v,std::min(k/32,254)-127);
        const float want=float(std::max(-max,std::min(max,v)));uint32_t bits;std::memcpy(&bits,&want,4);
        if(bits!=got[n*cols+k])++bad;
    }
    std::printf("%s decoder flag=%u mismatches=%zu/%zu (4096 distinct code/scale pairs)\n",
        bad?"ORACLE_MISMATCH":"PASS",naive,bad,got.size());return bad?3:0;
}
int fixture_run(const std::string& mode,const std::string& root,const std::string& fixture_path,
    const std::string& output,uint32_t naive) {
    const auto fixture=proof::read_json(fixture_path);proof::validate_fixture(fixture);
    const auto index=proof::read_json(root+"/model.safetensors.index.json");
    const bool gpu=mode=="dump-unpack";if(gpu)ready();
    std::ofstream dump;
    if(gpu) {dump.open(output,std::ios::out|std::ios::trunc);need(bool(dump),"cannot create dump");dump<<"{\"blocks\":[";}
    size_t blocks=0,samples=0,bad=0;
    for(const auto& b:fixture.at("blocks").list()) {
        const std::string name=b.at("name").str();const auto& shape=b.at("shape").list();
        const int rows=int(shape[0].num()),cols=int(shape[1].num());
        const auto w=proof::tensor(root,index,name+".weight",rows,cols/2,b.at("weight_sha256").str());
        const auto s=proof::tensor(root,index,name+".weight_scale",rows,cols/32,b.at("scale_sha256").str());
        std::printf("SOURCE PASS %s payload=%s scales=%s\n",name.c_str(),proof::sha256(w).c_str(),proof::sha256(s).c_str());
        if(gpu) {
            const auto words=gpu_unpack(w,s,rows,cols,naive);
            if(blocks)dump<<',';
            dump<<"{\"name\":\""<<name<<"\",\"weight_sha256\":\""<<proof::sha256(w)
                <<"\",\"scale_sha256\":\""<<proof::sha256(s)<<"\",\"bits\":[";
            const auto& positions=b.at("positions").list();const auto& expected=b.at("expected_f32_bits").list();
            for(size_t i=0;i<positions.size();++i) {
                const auto& p=positions[i].list();const auto bit=proof::hex32(words[p[0].num()*cols+p[1].num()]);
                if(i)dump<<',';dump<<'"'<<bit<<'"';++samples;if(bit!=expected[i].str())++bad;
            }
            dump<<"]}";
        }
        ++blocks;
    }
    if(gpu) {dump<<"]}\n";dump.close();need(bool(dump),"dump write failed");need(samples==55296,"sample count mismatch");}
    std::printf("%s %s flag=%u blocks=%zu hashes=%zu samples=%zu mismatches=%zu\n",
        bad?"ORACLE_MISMATCH":"PASS",mode.c_str(),naive,blocks,blocks*2,samples,bad);
    return bad?3:0;
}
int run(int argc,char** argv) {
    need(argc>=2,"modes: selftest | audit weights fixture | decoder-check flag | dump-unpack weights fixture output flag");
    const std::string mode=argv[1];
    if(mode=="selftest" && argc==2) {proof::selftest();tp4_comparator_selftest();std::puts("HOST PASS JSON negatives, path guards, SHA256 known vectors");return 0;}
    if(mode=="wire-export" && argc==5)return wire_gpu(argv[2],argv[3],argv[4]);
    if(mode=="bench-selftest" && argc==2) {proof::bench_selftest();return 0;}
    if(mode=="diagnostic" && argc==6) {
        need(std::string(argv[5])=="5","forced-four diagnostic M must be 5");
#if defined(M26X_DIAGNOSTIC_FORCE4) && M26X_DIAGNOSTIC_FORCE4
        return bench_gpu(argv[2],argv[3],argv[4],256,0,5);
#else
        throw std::runtime_error("forced-four diagnostic requires its explicit compile-time policy");
#endif
    }
    if(mode=="profile" && argc==6) {
        const std::string m=argv[5];need(m=="4" || m=="5","B2 profile M must be 4 or 5");
        return bench_gpu(argv[2],argv[3],argv[4],256,std::stoi(m));
    }
    if(mode=="bench" && argc==6) {
        const std::string n=argv[5];need(n=="256" || n=="57" || n=="58","invalid bench resident count");
        return bench_gpu(argv[2],argv[3],argv[4],std::stoi(n));
    }
    if(mode=="audit" && argc==4)return fixture_run(mode,argv[2],argv[3],"",0);
    if(mode=="routing-selftest" && argc==2) {proof::routing_selftest();routing_host_plan_selftest();return 0;}
    if(mode=="routing" && argc==3) {
        const std::string mutation=argv[2];need(mutation=="0" || mutation=="1","routing mutation must be0 or1");
        return routing_gpu(mutation=="1");
    }
    if(mode=="tp4-audit" && argc==4)return proof::tp4_audit(argv[2],argv[3]);
    if(mode=="tp4" && argc==5) {
        const std::string f=argv[4];
        need(f=="0" || f=="1" || f=="4" || f=="8" || f=="64" || f=="128","unsupported TP4 mutation");
        return tp4_gpu(argv[2],argv[3],uint32_t(std::stoul(f)));
    }
    if(mode=="decoder-check" && argc==3)return decoder_check(flag(argv[2]));
    if(mode=="dump-unpack" && argc==6)return fixture_run(mode,argv[2],argv[3],argv[4],flag(argv[5]));
    throw std::runtime_error("unknown mode/arguments");
}
}
int main(int argc,char** argv) {
    int result=2;
    try {result=run(argc,argv);}
    catch(const CudaFailure& e){std::fprintf(stderr,"CUDA_FAILURE %s\n",e.what());result=4;}
    catch(const NumericalFailure& e){std::fprintf(stderr,"ORACLE_MISMATCH %s\n",e.what());result=3;}
    catch(const PaddingRead& e){std::fprintf(stderr,"PADDING_READ_DETECTED %s\n",e.what());result=5;}
    catch(const std::exception& e){std::fprintf(stderr,"INPUT_OR_HARNESS_FAILURE %s\n",e.what());result=2;}
    return cleanup_failed?4:result;
}
