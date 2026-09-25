#include "../../kernels/bench/p1_check.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main() {
  int count=0;auto check=[&](bool ok){if(!ok){std::fprintf(stderr,"FAIL P1 benchmark check %d\n",count);std::exit(1);}++count;};
  std::vector<float> ref(12,.5f),got=ref;
  check(bench::p1_full_check(got,ref,12).ok);
  for(size_t i=0;i<12;++i){got=ref;got[i]=0;auto r=bench::p1_full_check(got,ref,12);check(!r.ok&&r.index==i);}
  got.assign(12,0);check(!bench::p1_full_check(got,ref,12).ok);
  for(float bad:{float(NAN),float(INFINITY),-float(INFINITY)}) {
    got=ref;got.back()=bad;check(!bench::p1_full_check(got,ref,12).ok);
    check(!bench::p1_full_check(ref,got,12).ok);
  }
  got=ref;got[0]+=1e-5f;check(bench::p1_full_check(got,ref,12).ok);
  got[0]+=2e-5f;check(!bench::p1_full_check(got,ref,12).ok);
  check(!bench::p1_full_check(got,ref,11).ok);check(!bench::p1_full_check({},ref,12).ok);
  check(!bench::p1_full_check(got,{},12).ok);check(!bench::p1_full_check({}, {},0).ok);
  for(int i=-16;i<=16;++i){float q=i*.03125f;uint32_t bits;std::memcpy(&bits,&q,4);check(bench::p1_query_bits_ok(bits));}
  for(uint32_t bits:{0x3f800001u,0x7f800000u,0xff800000u,0x7fc00000u,0x00010000u})check(!bench::p1_query_bits_ok(bits));
  check(bench::p1_query_bits_ok(0x80000000u));check(bench::p1_query_bits_ok(0x00800000u));
  for(int s:{2048,32768,131072})check(bench::p1_reference_batch(s)==8);
  check(bench::p1_reference_batch(1048576)==2);
  auto mapping_ok=[](int t,int batch,int wrong) {
    std::vector<int> visits(t,0);
    for(int first=0;first<t;first+=batch) {
      int n=std::min(batch,t-first);
      for(int local=0;local<n;++local) {
        ++visits[first+local];
        for(int h=0;h<64;++h)for(int sp=0;sp<256;++sp) {
          int idx=(local*64+h)*256+sp;
          int lt=idx/(256*64),lh=(idx/256)%64,ls=idx%256;
          if(wrong==1)std::swap(lt,lh);
          int base=wrong==2?0:first;
          if(base+lt!=first+local||ls!=sp||
             base*64*192+(lt*64+lh)*192!=((first+local)*64+h)*192||
             base*64*128+(lt*64+lh)*128!=((first+local)*64+h)*128||
             ((lt*64+lh)*256+ls)*130!=idx*130)return false;
        }
      }
    }
    return std::all_of(visits.begin(),visits.end(),[](int n){return n==1;});
  };
  for(int t:{17,2048})for(int batch:{1,2,8})check(mapping_ok(t,batch,0));
  for(int batch:{1,2,8})for(int wrong:{1,2})check(!mapping_ok(17,batch,wrong));
  std::printf("RESULT: PASS P1 benchmark checker/lattice/slab %d checks (CPU only)\n",count);
}
