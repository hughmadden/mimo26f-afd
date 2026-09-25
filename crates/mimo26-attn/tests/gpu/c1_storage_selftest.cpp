// CPU storage/address model only; no GPU bank-counter or occupancy claim.
#include "decode_c1_model_storage.h"
#include "decode_tc_layout.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <set>
using S=m26tc::WarpPipeModel<uint16_t>;
int checks=0;
void check(bool v,const char* what) { ++checks; if(!v){std::fprintf(stderr,"FAIL %s\n",what);std::exit(1);} }
size_t offset(const S& s,const void* p) { return static_cast<const char*>(p)-reinterpret_cast<const char*>(&s); }
int main() {
  S s{};
  check(sizeof(S)==44416,"typed size");
  check(2*(sizeof(S)+1024)<=102400,"two-CTA byte model, not occupancy");
  check(sizeof(S::Slot::Prob)<=sizeof(s.slot[0].phase.k),"P phase fits K");
  check(offset(s,s.raw)%16==0,"raw aligned for cp.async");
  check(offset(s,s.k_ready)%8==0 && offset(s,s.p_ready)%8==0 && offset(s,s.free_slot)%8==0,"barriers aligned");
  check(offset(s,s.free_slot)+sizeof(s.free_slot)<=offset(s,&s.bad),"barriers disjoint from bad flag");
  for(int b=0;b<2;++b) {
    auto& t=s.slot[b];
    check(offset(s,t.v)+sizeof(t.v)<=offset(s,t.phase.k),"V outside phase alias");
    check(offset(s,t.phase.p.ph)+sizeof(t.phase.p.ph)<=offset(s,t.phase.p.pl) &&
          offset(s,t.phase.p.pl)+sizeof(t.phase.p.pl)<=offset(s,t.phase.p.score) &&
          offset(s,t.phase.p.score)+sizeof(t.phase.p.score)<=offset(s,t.phase.p.alpha),"P subregions disjoint");
    check(offset(s,t.phase.k)+sizeof(t.phase.k)<=offset(s,t.visible),"visibility outside phase alias");
    bool k_cover=true,k_banks=true,v_cover=true,v_banks=true;
    std::set<int> ki,vi;
    for(int i=0;i<16*192/2;i+=32) {
      std::set<int> banks;
      for(int lane=0;lane<32;++lane) {
        int r=m26tc::producer_row(i+lane,192),d=m26tc::producer_col(i+lane,192);
        int x=m26tc::tile_index(16,r,d);
        k_cover &= r<16 && d<192 && !(x&1) && x>=0 && x+1<16*192;
        k_cover &= ki.insert(x/2).second;
        banks.insert(int((offset(s,t.phase.k)+2*x)/4)%32);
      }
      k_banks &= banks.size()==32;
    }
    for(int i=0;i<16*128/2;i+=32) {
      std::array<int,32> counts{};
      for(int lane=0;lane<32;++lane) {
        int z=i+lane,r=(z/128)*2,d=z%128,x=m26tc::tile_index(128,d,r);
        v_cover &= !(x&1) && x>=0 && x+1<16*128 && vi.insert(x/2).second;
        ++counts[((offset(s,t.v)+2*x)/4)%32];
      }
      for(int n:counts) v_banks &= n==0 || n==4;
    }
    check(k_cover && ki.size()==16*192/2,"N16 K packed coverage");
    check(k_banks,"N16 K producer conflict-free word map");
    check(v_cover && vi.size()==16*128/2,"N16 V packed coverage");
    check(v_banks,"N16 V retains known four-way stores");
    bool loads=true;
    // All four A registers: Q planes and P planes; each packed load is one word.
    for(const void* base : {static_cast<void*>(s.qh),static_cast<void*>(s.ql),static_cast<void*>(s.qt),
                           static_cast<void*>(t.phase.p.ph),static_cast<void*>(t.phase.p.pl)}) {
      for(int reg=0;reg<4;++reg) {
        std::set<int> banks;
        for(int lane=0;lane<32;++lane) {
          int x=m26tc::tile_index(16,m26tc::a_row(lane,reg),m26tc::a_col(lane,reg));
          banks.insert(int((offset(s,base)+2*x)/4)%32);
        }
        loads &= banks.size()==32;
      }
    }
    // K and V B operands, with every legal eight-column output fragment.
    for(int rows : {16,128}) for(int frag=0;frag<rows/8;++frag) for(int reg=0;reg<2;++reg) {
      std::set<int> banks;
      size_t base=offset(s,rows==16?t.phase.k:t.v);
      for(int lane=0;lane<32;++lane) {
        int x=m26tc::tile_index(rows,frag*8+m26tc::b_col(lane),m26tc::b_row(lane,reg));
        banks.insert(int((base+2*x)/4)%32);
      }
      loads &= banks.size()==32;
    }
    check(loads,"both slots retain conflict-free packed operand loads");
  }
  bool raw=true;std::set<int> chunks;
  for(int r=0;r<16;++r)for(int d=0;d<192;d+=16) {
    int x=m26tc::raw_k(r,d);raw &= x>=0 && x+15<16*192 && !(x&15) && chunks.insert(x).second;
  }
  check(raw && chunks.size()==16*12,"N16 raw K aligned bijection");
  std::printf("RESULT: PASS C1 storage model %d/%d bytes=%zu (CPU model only, NOT GPU qualification)\n",checks,checks,sizeof(S));
}
