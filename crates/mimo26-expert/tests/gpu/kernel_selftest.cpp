// Host-only tests for shared CUDA arithmetic/indexing; no GPU is initialized.
#define M26X_NO_CUDA_RUNTIME
#include "mimo26_expert_kernels.h"
#include "mimo26_expert_tile.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <cstdlib>
static void require(bool ok,const char* what) {
    if(!ok) { std::fprintf(stderr,"FAIL: %s\n",what); std::exit(1); }
}
static uint32_t bits(float v) { uint32_t b; std::memcpy(&b,&v,4); return b; }
static uint32_t oracle(int n,int s,uint32_t flag) {
    // Independent FP64 LUT/ldexp oracle, NOT the arithmetic bit composition.
    const double magnitudes[]={0,.5,1,1.5,2,3,4,6};
    double v=magnitudes[n&7]; if(n&8) v=-v;
    int e=(flag&M26X_NAIVE_SCALE_ONE) ? 0 : ((flag&M26X_NAIVE_E8M0_NO_CLAMP)?s:std::min(s,254))-127;
    const double max=std::numeric_limits<float>::max();
    return bits(float(std::max(-max,std::min(max,std::ldexp(v,e)))));
}
static bool paired_coverage(int groups,int uniform,bool drop_second=false,bool wrong_group=false) {
    std::vector<int> offsets{0};
    for(int g=0;g<groups;++g) offsets.push_back(offsets.back()+(uniform<0?g%6:uniform));
    std::vector<int> seen(offsets.back());
    for(int by=0;by<2*(groups+3);++by) {
        if(drop_second && by%2)continue;
        const int group=wrong_group?by:m26x_tile_group<true>(by);
        if(group>=groups)continue; // padded metadata remains unread
        const int begin=offsets[group],end=offsets[group+1];
        const int start=begin+m26x_tile_offset<true,4>(by,0);
        for(int t=0;t<4 && start+t<end;++t)++seen[start+t];
    }
    return std::all_of(seen.begin(),seen.end(),[](int n){return n==1;});
}
static bool stage_coverage(int m,bool truncate=false) {
    std::vector<int> seen(m*256);
    for(int thread=0;thread<M26X_THREADS;++thread)
        for(int v=thread;v<m*64;v+=M26X_THREADS) {
            const int dst=(v/64)*256+m26x_x_swizzle((v%64)*4);
            require(dst>=0 && dst+3<int(seen.size()),"activation float4 stage OOB");
            for(int j=0;j<4;++j)++seen[dst+j];
            if(truncate)break;
        }
    return std::all_of(seen.begin(),seen.end(),[](int n){return n==1;});
}
int main() {
    for(int m:{1,2,4,5,6,7,8})require(stage_coverage(m),"exact-M stage loses/duplicates words");
    require(!stage_coverage(5,true),"truncated odd-M stage mutation powerless");
    std::puts("HOST PASS exact-M float4 staging: seven tile widths, odd-row tail and powered truncation negative");
    for(int groups:{0,1,3,256})for(int m=-1;m<=5;++m)
        require(paired_coverage(groups,m),"paired four-row grid loses/duplicates rows");
    require(!paired_coverage(1,5,true),"missing fifth-row mutation powerless");
    require(!paired_coverage(3,5,false,true),"wrong paired group mutation powerless");
    for(int y=0;y<8;++y)for(int z=0;z<3;++z) {
        require(m26x_tile_group<false>(y)==y,"baseline group mapping changed");
        require(m26x_tile_offset<false,4>(y,z)==4*z,"baseline row mapping changed");
    }
    std::puts("HOST PASS R13 paired CTA coverage: empty/ragged/M0..5/full256/padded groups; two powered mapping negatives");
    unsigned checked=0;
    for(uint32_t flag:{0u,M26X_NAIVE_E8M0_NO_CLAMP,M26X_NAIVE_SCALE_ONE})
        for(int n=0;n<16;++n) for(int s=0;s<256;++s) {
            if(m26x_decode_bits(n,s,flag)!=oracle(n,s,flag)) {
                std::fprintf(stderr,"decode n=%d scale=%d flag=%u got=%08x want=%08x\n",
                    n,s,flag,m26x_decode_bits(n,s,flag),oracle(n,s,flag));
                return 1;
            }
            ++checked;
        }
    require(m26x_decode_bits(2,255,0)!=m26x_decode_bits(2,255,M26X_NAIVE_E8M0_NO_CLAMP),"unclamp mutation powerless");
    require(m26x_decode_bits(2,125,0)!=m26x_decode_bits(2,125,M26X_NAIVE_SCALE_ONE),"scale-one mutation powerless");
    std::array<int,256> seen{};
    for(int k=0;k<256;++k) {
        const int i=m26x_x_swizzle(k);
        require(i>=0 && i<256,"swizzle OOB"); ++seen[i];
        require(i==m26x_x_swizzle(k-k%4)+k%4,"swizzle breaks aligned float4");
    }
    for(int n:seen) require(n==1,"swizzle not a bijection");
    for(int q=0;q<32;++q) {
        std::array<bool,32> banks{};
        for(int lane=0;lane<8;++lane) {
            const int bank=m26x_x_swizzle(lane*32+q)%32;
            require(!banks[bank],"8-lane activation bank conflict"); banks[bank]=true;
        }
    }
    for(int rows:{512,4096}) {
        std::vector<int> stores(rows);
        for(int block=0;block<rows/32;++block) for(int thread=0;thread<256;++thread) {
            const int row=m26x_owned_row(block,thread);
            require(row>=0 && row<rows,"row OOB");
            if(thread%8==0) ++stores[row];
        }
        for(int n:stores) require(n==1,"duplicate or missing output store");
    }
    for(int cols:{512,4096}) {
        std::vector<int> coverage(cols);
        for(int k0=0;k0<cols;k0+=256) for(int lane=0;lane<8;++lane) for(int q=0;q<32;++q)
            ++coverage[k0+lane*32+q];
        for(int n:coverage) require(n==1,"K vector coverage mismatch");
    }
    require(m26x_bf16_bits(0x80000000u)==0x8000u,"negative BF16 zero");
    require(m26x_bf16_bits(0x3f808000u)==0x3f80u,"BF16 even tie");
    require(m26x_bf16_bits(0x3f818000u)==0x3f82u,"BF16 odd tie");
    require(m26x_bf16_bits(0x7f7fffffu)==0x7f80u,"BF16 overflow");
#ifdef M26X_WITH_OBJECT
    m26x_plan plan{};
    plan.layout_version=2; plan.capacity_class=2048; plan.resident_experts=4;
    plan.n_groups=3; plan.padded_groups=1; plan.total_tokens=3; plan.max_m=2;
    plan.grouped_bytes=uint64_t(4)*M26X_QUARTER_SLICE_BYTES;
    int32_t ids[]={3,-99,0}, offsets[]={0,1,1,3};
    require(m26x_validate_host_plan(&plan,ids,offsets)==0,"valid sparse/empty host plan");
    plan.layout_version=1; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"v1 plan accepted"); plan.layout_version=2;
    plan.max_m=1; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"undersized max-M accepted"); plan.max_m=2;
    plan.grouped_bytes--; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"short image accepted"); plan.grouped_bytes++;
    ids[0]=4; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"invalid live ID accepted"); ids[0]=3;
    offsets[0]=1; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"nonzero start accepted"); offsets[0]=0;
    offsets[2]=-1; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"negative offset accepted"); offsets[2]=1;
    offsets[3]=4; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"wrong terminal offset accepted"); offsets[3]=3;
    plan.capacity_class=512; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"unknown capacity accepted"); plan.capacity_class=2048;
    plan.n_groups=-1; require(m26x_validate_host_plan(&plan,ids,offsets)!=0,"negative groups accepted");
    std::puts("HOST PLAN PASS: valid sparse/empty plan + nine malformed-plan refusals (actual linked CUDA TU host function)");
#endif
    std::printf("HOST PASS: %u arithmetic decoder cases; signed-zero/subnormal/saturation; mutations; row ownership; vector coverage; swizzle; BF16 RNE\n",checked);
}
