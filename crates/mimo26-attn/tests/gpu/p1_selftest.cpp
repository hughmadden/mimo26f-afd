// CPU layout/accounting checks only, NOT numerical or hardware qualification.
#include "../../kernels/include/prefill_tc_storage.h"
#include "../../kernels/include/decode_tc_layout.h"
#include "../../kernels/bench/metrics.h"
#include <array>
#include <vector>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
int checks=0;
void check(bool v,const char* why){++checks;if(!v){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
int main(){
  using F=m26tc::PrefillStorage<uint16_t,true>;
  using B=m26tc::PrefillStorage<uint16_t,false>;
  using S=m26tc::PrefillStorage<uint16_t,true,true>;
  check(sizeof(F)==88128&&sizeof(B)==38976,"typed footprints");
  check(sizeof(S)==48192,"split typed footprint (P1M=32 f32q)");
  check(sizeof(F)+1024<=102400&&2*(sizeof(B)+1024)<=102400,"shared-only capacity ceilings");
  check(offsetof(F,phase)+sizeof(F::Phase)<=offsetof(F,v),"V never aliases phase");
  check(sizeof(F::Phase)==8448,"phase size dominated by score/P/alpha, not K");
  for(int rows:{16,64,128}){
    int depth=rows==128?16:192;std::vector<int> hit(rows*depth);
    for(int r=0;r<rows;++r)for(int k=0;k<depth;++k)++hit[m26tc::tile_index(rows,r,k)];
    check(std::all_of(hit.begin(),hit.end(),[](int n){return n==1;}),"swizzle bijection");
  }
  for(int m=0;m<64;m+=16)for(int d=0;d<192;d+=16)for(int reg=0;reg<4;++reg){
    std::set<int> banks;
    for(int lane=0;lane<32;++lane){
      int row=m26tc::a_row(lane,reg),k=d+m26tc::a_col(lane,reg);
      int at=m*16+m26tc::tile_index(64,row,k);
      check(at==m26tc::tile_index(64,m+row,k),"subtile pointer offset exact");
      banks.insert((at/2)%32);
    }
    check(banks.size()==32,"Q/P packed A banks");
  }
  std::vector<int> qk(64*16),pv(64*128),p(64*16);
  for(int tid=0;tid<256;++tid){
    int w=tid/32,l=tid%32,base=(w/2)*16;
    for(int e=0;e<4;++e){
      int r=base+m26tc::c_row(l,e);
      ++qk[r*16+(w%2)*8+m26tc::c_col(l,e)];
      for(int f=0;f<8;++f)++pv[r*128+(w%2)*64+f*8+m26tc::c_col(l,e)];
      ++p[(tid/4)*16+tid%4+e*4];
    }
  }
  for(auto* v:{&qk,&pv,&p})check(std::all_of(v->begin(),v->end(),[](int n){return n==1;}),"complete disjoint fragment coverage");
  // Split variant (P1M=32, 4 warps): two warp-pairs of 16 rows, same disjoint coverage.
  {
    std::vector<int> qk(32*16),pv(32*128),p(32*16);
    for(int tid=0;tid<128;++tid){
      int w=tid/32,l=tid%32,base=(w/2)*16;
      for(int e=0;e<4;++e){
        int r=base+m26tc::c_row(l,e);
        ++qk[r*16+(w%2)*8+m26tc::c_col(l,e)];
        for(int f=0;f<8;++f)++pv[r*128+(w%2)*64+f*8+m26tc::c_col(l,e)];
        ++p[(tid/4)*16+tid%4+e*4];
      }
    }
    for(auto* v:{&qk,&pv,&p})check(std::all_of(v->begin(),v->end(),[](int n){return n==1;}),"split complete disjoint fragment coverage");
  }
  for(int nkv:{4,8})for(int t:{1,3,4,5,8,9,17}){
    int rep=64/nkv,qt=64/rep;std::vector<int> rows(t*64);
    for(int b=0;b<(t+qt-1)/qt;++b)for(int kh=0;kh<nkv;++kh)for(int r=0;r<64;++r){
      int tr=b*qt+r/rep,h=kh*rep+r%rep;if(tr<t)++rows[tr*64+h];
    }
    check(std::all_of(rows.begin(),rows.end(),[](int n){return n==1;}),"ragged query/GQA coverage");
  }
  for(int nkv:{4,8})for(int t:{1,3,4,5,8,9,17}){
    int rep=64/nkv,qt=32/rep;std::vector<int> rows(t*64);
    for(int b=0;b<(t+qt-1)/qt;++b)for(int kh=0;kh<nkv;++kh)for(int r=0;r<32;++r){
      int tr=b*qt+r/rep,h=kh*rep+r%rep;if(tr<t)++rows[tr*64+h];
    }
    check(std::all_of(rows.begin(),rows.end(),[](int n){return n==1;}),"split ragged query/GQA coverage");
  }
  // Independent enumeration of scheduled tiles, testing diagonal/window edges.
  for(int nkv:{4,8})for(int t:{1,3,4,5,8,9,17})for(int s:{17,31,65})for(int win:{0,1,7,16,128}){
    uint64_t matrix_pairs=0;int qt=nkv;
    for(int b=0;b<t;b+=qt)for(int j=0;j<s;j+=16){
      bool any=false;
      for(int q=b;q<std::min(t,b+qt);++q)for(int k=j;k<std::min(s,j+16);++k)
        any|=k<=s-t+q&&(!win||k>=s-t+q-win+1);
      if(any)matrix_pairs+=64*16*nkv;
    }
    for(bool residual:{false,true})check(bench::p1_mma_flops(t,s,nkv,win,residual)==2.0*((residual?3:1)*192+256)*matrix_pairs,"causal/window padded MMA work");
  }
  for(bool residual:{false,true})for(int s:{131072,1048576}){
    double useful=bench::flops(2048,s),executed=bench::p1_mma_flops(2048,s,4,0,residual);
    check(executed>useful*(residual?2.6:1.4),"full-context padding cannot use nominal factor");
  }
  for(auto shape:{std::array<int,4>{0,128,4,0},{129,128,4,0},{65536,65536,4,0},{1,128,3,0},{1,128,4,-1}}){
    bool rejected=false;try{bench::p1_mma_flops(shape[0],shape[1],shape[2],shape[3]);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"invalid P1 accounting shape");
  }
  std::printf("RESULT: PASS P1 CPU layout/accounting checks=%d Q3_shared=%zu Q1_shared=%zu split_shared=%zu (NOT GPU qualification)\n",checks,sizeof(F),sizeof(B),sizeof(S));
}
