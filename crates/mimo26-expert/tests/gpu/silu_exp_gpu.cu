// Exhaustive diagnostic only. Includes the actual current activation unchanged.
#include "silu_exp_reference.h"
#include "b1/grouped.cu"
#include <cuda_runtime.h>
using namespace exp_pin;
namespace {
bool cleanup_failed=false;
void ck(cudaError_t e,const char* why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
struct Device {
    float2* ptr=nullptr;
    explicit Device(size_t n){ck(cudaMalloc(&ptr,n*sizeof(float2)),"allocate output");}
    ~Device(){if(ptr && cudaFree(ptr)!=cudaSuccess)cleanup_failed=true;}
    Device(const Device&)=delete;
};
__global__ void probe(uint64_t begin,float2* out,uint32_t n){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const uint32_t b=uint32_t(begin+i);const float x=__uint_as_float(b);
    out[i]=make_float2(expf(x),(b&0x7fffffff)<0x7f800000?m26b1::silu_up(x,1.f):__uint_as_float(0x7fc00000));
}
__global__ void witnesses(float2* out){
    const float g[]={-.19146597385406494f,.327481746673584f,-.1987401247024536f,.23654770851135254f};
    const float u[]={-.17092028260231018f,.14912837743759155f,.5958439111709595f,.2174588441848755f};
    const int i=threadIdx.x;if(i>=4)return;
    const float e=expf(g[i]>=0?-g[i]:g[i]);
    const float wrong=__uint_as_float(__float_as_uint(e)+1);
    const float s=g[i]>=0?g[i]/(1.f+wrong):(g[i]*wrong)/(1.f+wrong);
    out[i]=make_float2(m26b1::silu_up(g[i],u[i]),s*u[i]);
}
struct Stats {uint64_t mismatch=0,max_ulp=0;uint32_t input=0,got=0,want=0;};
void update(Stats& s,uint32_t b,uint32_t got,uint32_t want){
    if(got==want)return;++s.mismatch;const auto distance=ulps(got,want);
    if(distance>s.max_ulp){s.max_ulp=distance;s.input=b;s.got=got;s.want=want;}
}
void json_stats(std::ostream& out,const Stats& s){out<<"{\"mismatch\":"<<s.mismatch<<",\"max_ulp\":"<<s.max_ulp<<",\"input_bits\":"<<s.input<<",\"got_bits\":"<<s.got<<",\"reference_bits\":"<<s.want<<"}";}
int run(const std::string& root,const std::string& fixes){
    selftest();const auto table=corrections(fixes);
    const char* permit=std::getenv("MIMO26_BUILDER_GPU");need(permit && std::string(permit)=="1","GPU permission required");
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"device identity");
    need(prop.major==12 && prop.minor==1 && prop.multiProcessorCount==48,"require GB10 sm121/48SM");
    size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory guard");need(free>=size_t(8)*1024*1024*1024+64*1024*1024,"8GiB reserve");
    std::printf("GPU %s arch=121 SMs=48 free_bytes=%zu diagnostic-only\n",prop.name,free);
    constexpr uint32_t n=1<<20;Device d(n);std::vector<float2> host(n);
    witnesses<<<1,4>>>(d.ptr);ck(cudaGetLastError(),"witness launch");ck(cudaMemcpy(host.data(),d.ptr,4*sizeof(float2),cudaMemcpyDeviceToHost),"witness download");
    const float want[]={.014801025390625f,.02838134765625f,-.0533447265625f,.02874755673110485f};
    int wrong=0;for(int i=0;i<4;++i){need(bits(host[i].x)==bits(want[i]),"retained witness not reproduced");wrong+=bits(host[i].y)!=bits(want[i]);}
    need(wrong>0,"one-ULP exp perturbation negative powerless");
    std::printf("WITNESS PASS four retained GPU activations reproduced bitwise; exp+1ULP mutant changes=%d/4\n",wrong);
    std::ofstream near(root+"/gpu-midpoints.csv"), examples(root+"/gpu-examples.csv");
    need(bool(near)&&bool(examples),"output files");
    near<<"input_bits,reference_bits,double_bits,midpoint_bits,gap_double_ulps,gpu_exp_bits,corrected_reference_bits\n";
    examples<<"kind,input_bits,gpu_bits,reference_bits\n";
    Stats ep,en,hp,hn;uint64_t boundary_bad=0,class_bad=0,nan=0,inf=0,special_bad=0,near_count=0,correction_count=0,hash=1469598103934665603ull;
    unsigned emitted[4]={};
    for(uint64_t begin=0;begin<(1ull<<32);begin+=n){
        probe<<<(n+255)/256,256>>>(begin,d.ptr,n);ck(cudaGetLastError(),"exp launch");
        ck(cudaMemcpy(host.data(),d.ptr,n*sizeof(float2),cudaMemcpyDeviceToHost),"exp download");
        for(uint32_t i=0;i<n;++i){const uint32_t b=uint32_t(begin+i);const auto r=reference(b);const uint32_t expected=corrected(b,r,table),got=bits(host[i].x);
            hash=(hash^expected)*1099511628211ull;
            if(!finite(b)){if((b&0x7fffffff)>0x7f800000){++nan;special_bad+=(got&0x7fffffff)<=0x7f800000;}else{++inf;special_bad+=got!=expected;}continue;}
            boundary_bad+=(expected==0 || expected==0x3f800000 || expected==0x7f800000) && got!=expected;
            correction_count+=expected!=r.value;const unsigned sign=b>>31;auto& es=sign?en:ep;update(es,b,got,expected);
            if(got!=expected && emitted[sign]++<24){char row[100];std::snprintf(row,sizeof row,"exp,%08x,%08x,%08x\n",b,got,expected);examples<<row;}
            if(r.near){++near_count;char row[240];std::snprintf(row,sizeof row,"%08x,%08x,%016llx,%016llx,%.17g,%08x,%08x\n",b,r.value,(unsigned long long)bits64(r.y),(unsigned long long)bits64(r.mid),r.gap,got,expected);near<<row;}
            const float g=f32(b);const uint32_t t=bits(g>=0?-g:g);const auto er=sign?r:reference(t);
            const float want_h=activation(g,1.f,f32(corrected(t,er,table)));const uint32_t wh=bits(want_h),gh=bits(host[i].y);
            auto& hs=sign?hn:hp;update(hs,b,gh,wh);
            if(gh!=wh){bool member=false;const int64_t eb=corrected(t,er,table);
                for(int delta=-2;delta<=2;++delta){const int64_t candidate=eb+delta;if(candidate<0 || candidate>0x3f800000)continue;
                    member|=bits(activation(g,1.f,f32(uint32_t(candidate))))==gh;}
                class_bad+=!member;
            }
            if(gh!=wh && emitted[2+sign]++<24){char row[100];std::snprintf(row,sizeof row,"activation-u1,%08x,%08x,%08x\n",b,gh,wh);examples<<row;}
        }
        if(((begin+n)&0xfffffff)==0){std::printf("GPU SCAN covered=%llu/4294967296 exp_bad=%llu activation_bad=%llu\n",(unsigned long long)(begin+n),(unsigned long long)(ep.mismatch+en.mismatch),(unsigned long long)(hp.mismatch+hn.mismatch));std::fflush(stdout);}
    }
    for(const auto* s:{&ep,&en}){char row[100];std::snprintf(row,sizeof row,"exp-worst,%08x,%08x,%08x\n",s->input,s->got,s->want);examples<<row;}
    for(const auto* s:{&hp,&hn}){char row[100];std::snprintf(row,sizeof row,"activation-u1-worst,%08x,%08x,%08x\n",s->input,s->got,s->want);examples<<row;}
    near.close();examples.close();need(bool(near)&&bool(examples),"output write failure");need(special_bad==0,"special exp class mismatch");
    std::ofstream out(root+"/gpu-summary.json");out<<"{\"patterns\":4294967296,\"finite\":4278190080,\"nan\":"<<nan<<",\"infinite\":"<<inf<<",\"special_bad\":"<<special_bad<<",\"near_midpoints\":"<<near_count<<",\"reference_corrections\":"<<correction_count<<",\"exp_positive\":";json_stats(out,ep);out<<",\"exp_negative\":";json_stats(out,en);out<<",\"activation_positive_u1\":";json_stats(out,hp);out<<",\"activation_negative_u1\":";json_stats(out,hn);
    out<<",\"zero_one_infinity_exp_mismatches\":"<<boundary_bad<<",\"activation_exp2_set_nonmembers_u1\":"<<class_bad<<",\"corrected_reference_fnv64\":\""<<std::hex<<hash<<"\",\"scope\":\"exp all bit patterns; actual B1 SiLU all finite g at u=1 plus four retained (g,u) witnesses; not exhaustive all (g,u) pairs; Decimal audit pending\"}\n";
    need(bool(out),"summary write failure");return 0;
}
}
int main(int argc,char** argv){int result=0;try{
    if(argc==2 && std::string(argv[1])=="--selftest")selftest();
    else {need(argc==4 && std::string(argv[1])=="--gpu","use --selftest or --gpu output corrections");result=run(argv[2],argv[3]);}
}catch(const std::exception& e){std::fprintf(stderr,"PROBE_FAILURE %s\n",e.what());result=2;}
return cleanup_failed?4:result;}
