#include "silu_exp_reference.h"
using namespace exp_pin;
int main(int argc,char** argv){try{
    selftest();if(argc==2 && std::string(argv[1])=="--selftest")return 0;
    need(argc==3 && std::string(argv[1])=="--scan","use --selftest or --scan directory");
    const std::string root=argv[2];std::ofstream near(root+"/midpoints.csv");need(bool(near),"open midpoint CSV");
    near<<"input_bits,reference_bits,double_bits,midpoint_bits,gap_double_ulps\n";
    uint64_t nf=0,nn=0,ni=0,nnear=0,hash=1469598103934665603ull;
    for(uint64_t b=0;b<(1ull<<32);++b){const auto r=reference(uint32_t(b));
        if(finite(uint32_t(b)))++nf;else if((b&0x7fffffff)==0x7f800000)++ni;else ++nn;
        hash=(hash^r.value)*1099511628211ull;
        if(r.near){++nnear;char row[180];std::snprintf(row,sizeof row,"%08x,%08x,%016llx,%016llx,%.17g\n",unsigned(b),r.value,(unsigned long long)bits64(r.y),(unsigned long long)bits64(r.mid),r.gap);near<<row;}
        if((b&0xfffffff)==0xfffffff){std::printf("CPU SCAN covered=%llu/4294967296 near=%llu\n",(unsigned long long)(b+1),(unsigned long long)nnear);std::fflush(stdout);}
    }
    near.close();need(bool(near),"midpoint write failure");
    std::ofstream summary(root+"/cpu-summary.json");
    summary<<"{\"patterns\":4294967296,\"finite\":"<<nf<<",\"nan\":"<<nn<<",\"infinite\":"<<ni<<",\"near_midpoints\":"<<nnear<<",\"reference_fnv64\":\""<<std::hex<<hash<<"\",\"scope\":\"all-bit-pattern enumeration; binary64 exp candidate plus analytic constant regions; Decimal validation pending\"}\n";
    need(bool(summary),"summary write failure");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"INPUT_FAILURE %s\n",e.what());return 2;}}
