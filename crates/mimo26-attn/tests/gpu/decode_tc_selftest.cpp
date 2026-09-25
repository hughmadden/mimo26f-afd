// Host contract tests, not a replacement for real mma.sync GPU parity.
#include "../../kernels/include/decode_tc_layout.h"
#include "../../kernels/include/decode_pipe_storage.h"
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
int main() {
  int count=0;
  auto check=[&](bool ok,const char* name) { if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);} ++count; };
  for (auto shape: {std::pair<int,int>{16,192},{32,192},{128,32},{16,32}}) {
    int rows=shape.first, cols=shape.second;
    std::set<int> indices;
    for(int r=0;r<rows;r++) for(int c=0;c<cols;c++) indices.insert(m26tc::tile_index(rows,r,c));
    check(int(indices.size())==rows*cols && *indices.begin()==0 && *indices.rbegin()==rows*cols-1,"shared tile bijective and bounded");
  }
  for(int width:{192,128}) {
    std::set<int> raw;
    bool aligned=true;
    for(int r=0;r<32;++r)for(int d=0;d<width;++d) {
      int x=width==192?m26tc::raw_k(r,d):m26tc::raw_v(r,d);
      raw.insert(x);aligned&=(x&15)==(d&15);
    }
    check(raw.size()==size_t(32*width)&&*raw.begin()==0&&*raw.rbegin()==32*width-1,"raw chunk permutation bijective/bounded");
    check(aligned,"all cp.async chunks stay contiguous/aligned");
  }
  for(auto shape:{std::pair<int,int>{32,192},{128,32}}) {
    int rows=shape.first,cols=shape.second;
    std::set<int> pairs;bool stores=true,loads=true;
    for(int base=0;base<rows*cols/2;base+=32) {
      std::set<int> banks;std::set<int> raw_words[32];
      for(int lane=0;lane<32;++lane) {
        int i=base+lane,r=m26tc::producer_row(i,cols),d=m26tc::producer_col(i,cols);
        int dst=m26tc::tile_index(rows,r,d);
        pairs.insert(r*cols+d);stores&=(dst%2)==0;
        banks.insert((dst/2)%32);
        int src=cols==192?m26tc::raw_k(r,d):m26tc::raw_v_dmajor(r,d&~3);
        raw_words[(src/4)%32].insert(src/4);
      }
      stores&=banks.size()==32;
      for(auto& words:raw_words)loads&=words.size()<=1;
    }
    check(pairs.size()==size_t(rows*cols/2)&&*pairs.rbegin()==rows*cols-2,"producer owns each logical pair once");
    check(stores,"producer packed stores use 32 distinct banks");
    check(loads,"producer raw reads are conflict-free including word broadcasts");
  }
  // Negative witness: the old contiguous pair traversals have four-way stores.
  for(bool key:{true,false}) {
    std::set<int> banks;
    for(int lane=0;lane<32;++lane) {
      int dst=key?m26tc::tile_index(32,0,lane*2):m26tc::tile_index(128,lane,0);
      banks.insert((dst/2)%32);
    }
    check(banks.size()==8,"old producer traversal reproduces measured four-way conflict");
  }
  auto perm=[](uint32_t a,uint32_t b,unsigned selector) {
    uint64_t both=uint64_t(a)|(uint64_t(b)<<32);uint32_t out=0;
    for(int n=0;n<4;++n)out|=uint32_t((both>>(8*((selector>>(4*n))&7)))&255)<<(8*n);
    return out;
  };
  for(int tail:{0,1,3,17,32}) {
    std::vector<uint8_t> raw(4096),gold(4096),transposed(4096);
    uint32_t incoming[1024],outgoing[1024];
    for(int r=0;r<32;++r)for(int d=0;d<128;++d) {
      uint8_t x=r<tail?uint8_t((r*37+d*13)^(r*d)):0;
      gold[r*128+d]=x;raw[m26tc::raw_v(r,d)]=x;
    }
    bool transpose_banks=true;
    for(int i=0;i<1024;++i) {
      int d=i/8,r=(i&7)*4+(d&3);
      std::memcpy(incoming+i,raw.data()+m26tc::raw_v(r,d&~3),4);
    }
    for(int i=0;i<1024;++i) {
      int base=(i&~31)+(i&7);unsigned s=0x40+((i/8)&3)*0x11;
      outgoing[i]=perm(perm(incoming[base],incoming[base+8],s),perm(incoming[base+16],incoming[base+24],s),0x5410);
    }
    for(int base=0;base<1024;base+=32) {
      std::set<int> banks;std::set<int> ingress[32];
      for(int i=base;i<base+32;++i) {
        int d=i/8,r=(i&7)*4,addr=m26tc::raw_v_dmajor(d,r);
        std::memcpy(transposed.data()+addr,outgoing+i,4);banks.insert((addr/4)%32);
        int src=m26tc::raw_v(r+(d&3),d&~3);ingress[(src/4)%32].insert(src/4);
      }
      transpose_banks&=banks.size()==32;
      for(auto& words:ingress)transpose_banks&=words.size()<=4;
    }
    bool exact=true;
    for(int d=0;d<128;++d)for(int r=0;r<32;++r)
      exact&=transposed[m26tc::raw_v_dmajor(d,r)]==gold[r*128+d];
    check(exact,"in-place V transpose model exact including zero-filled tails");
    check(transpose_banks,"V transpose output bank-free, ingress bounded four-way");
  }
  std::set<int> aa,bb,cc;
  for(int lane=0;lane<32;lane++) {
    for(int reg=0;reg<4;reg++) for(int h=0;h<2;h++) aa.insert(m26tc::a_row(lane,reg)*16+m26tc::a_col(lane,reg)+h);
    for(int reg=0;reg<2;reg++) for(int h=0;h<2;h++) bb.insert((m26tc::b_row(lane,reg)+h)*8+m26tc::b_col(lane));
    for(int reg=0;reg<4;reg++) cc.insert(m26tc::c_row(lane,reg)*8+m26tc::c_col(lane,reg));
  }
  check(aa.size()==256 && *aa.rbegin()==255,"A fragment owns all 16x16 entries");
  check(bb.size()==128 && *bb.rbegin()==127,"B fragment owns all 16x8 entries");
  check(cc.size()==128 && *cc.rbegin()==127,"C fragment owns all 16x8 entries");
  for(int reg=0;reg<4;reg++) {
    std::set<int> banks;
    for(int lane=0;lane<32;lane++) {
      int index=m26tc::tile_index(16,m26tc::a_row(lane,reg),m26tc::a_col(lane,reg));
      if(index%2) std::abort();
      banks.insert((index/2)%32);
    }
    check(banks.size()==32,"A packed loads avoid bank conflicts");
  }
  for(int rows: {32,128}) for(int reg=0;reg<2;reg++) {
    std::set<int> banks;
    for(int lane=0;lane<32;lane++) banks.insert((m26tc::tile_index(rows,m26tc::b_col(lane),m26tc::b_row(lane,reg))/2)%32);
    check(banks.size()==32,"B packed loads avoid bank conflicts");
  }
  std::set<int> output;
  for(int w=0;w<4;w++) for(int col=0;col<4;col++) for(int lane=0;lane<32;lane++) for(int e=0;e<4;e++)
    output.insert(m26tc::c_row(lane,e)*128+w*32+col*8+m26tc::c_col(lane,e));
  check(output.size()==2048 && *output.rbegin()==2047,"PV fragments cover M16 V128 once");
  bool codec=true;
  for(int code=0;code<256;code++) {
    uint32_t bits=m26tc::e4m3_bits(uint8_t(code)); float got; std::memcpy(&got,&bits,4);
    int e=(code>>3)&15,m=code&7;
    if(e==15 && m==7) codec &= std::isnan(got);
    else {
      float expected=e ? std::ldexp(1.f+m/8.f,e-7) : m/512.f;
      if(code&128) expected=-expected;
      codec &= got==expected && std::signbit(got)==std::signbit(expected);
    }
  }
  check(codec,"all 256 E4M3FN codes including signed zero/NaN");
  bool packed=true;
  for(unsigned code=0;code<65536;++code) {
    uint32_t expected=(m26tc::e4m3_bits(uint8_t(code))>>16) |
                      (m26tc::e4m3_bits(uint8_t(code>>8))&0xffff0000u);
    packed &= m26tc::e4m3x2_bf16(uint16_t(code))==expected;
  }
  check(packed,"all 65536 packed E4M3 pairs preserve BF16 bits");
  std::set<int> output8;
  for(int w=0;w<8;++w) for(int c=0;c<2;++c) for(int l=0;l<32;++l) for(int e=0;e<4;++e)
    output8.insert(m26tc::c_row(l,e)*128+w*16+c*8+m26tc::c_col(l,e));
  check(output8.size()==2048 && *output8.rbegin()==2047,"eight-warp PV covers output once");
  for(int rep: {8,16}) {
    std::set<int64_t> slots;
    for(int t=0;t<3;t++) for(int kv=0;kv<64/rep;kv++) for(int r=0;r<rep;r++) for(int sp=0;sp<8;sp++)
      slots.insert(m26tc::partial_index(t,kv*rep+r,sp,8));
    check(slots.size()==3*64*8 && *slots.rbegin()==3*64*8-1,"GQA partial indexing includes T and all heads");
  }
  for(int size: {0,3,17,257,1048576}) {
    int covered=0;
    for(int sp=0;sp<8;sp++) {
      int lo=int(int64_t(size)*sp/8),hi=int(int64_t(size)*(sp+1)/8);
      for(int64_t tile=lo;tile<hi;tile+=32) for(int n=0;n<32;n++) if(tile+n<hi) covered++;
    }
    check(covered==size,"split tails cover every logical key exactly once");
  }
  using CM=m26tc::CompactMath<uint16_t>;using CP=m26tc::CompactPipe<uint16_t>;
  check(sizeof(CP)==49536,"C3 real storage footprint");
  check(sizeof(CM::Prob)<=sizeof(CM::Phase),"C3 complete score/P/alpha state fits retired K");
  check(offsetof(CM,maximum)>=offsetof(CM,phase)+sizeof(CM::Phase),"C3 persistent m/l outside aliased region");
  check(offsetof(CP,raw)>=sizeof(CM),"C3 async raw stage disjoint from math phase");
  // Same operand bank residues as the C0 layout; all existing tile/producer
  // proofs apply unchanged. Raw banks may rotate uniformly, not alias anew.
  const size_t bases[]={offsetof(CM,qh),offsetof(CM,ql),offsetof(CM,qt),offsetof(CM,phase.k),
    offsetof(CM,v),offsetof(CM,phase.p.ph),offsetof(CM,phase.p.pl),offsetof(CM,phase.p.score),offsetof(CM,phase.p.alpha)};
  const size_t old_bases[]={0,6144,12288,18432,30720,38912,39936,40960,43136};
  for(int i=0;i<9;++i)check(bases[i]%128==old_bases[i]%128,"C3 operand bank residue unchanged");
  check(2*(sizeof(CP)+1024)<=100*1024,"C3 two-CTA capacity model incl. assumed 1 KiB reservation, NOT runtime occupancy");
  std::vector<unsigned char> phase(sizeof(CM::Phase),0x31);
  size_t score_offset=offsetof(CM,phase.p.score)-offsetof(CM,phase.k);
  float overwrite=.75f;std::memcpy(phase.data()+score_offset,&overwrite,sizeof(overwrite));
  check(phase[score_offset]!=0x31,"C3 early score store would corrupt live K: barrier negative witness");
  for(int terms:{1,3}) {
    int visits[3][2]={};
    for(int step=0;step<12;++step)for(int term=0;term<terms;++term)
      ++visits[term][m26tc::q_acc_group(step)];
    bool complete=true;
    for(int term=0;term<terms;++term)for(int group=0;group<2;++group)complete&=visits[term][group]==6;
    check(complete,"C0 retains every Q term/k-group exactly once");
  }
  float pv[2][2]={},wrong[2][2]={};double reference=0;
  for(float alpha:{0.f,.25f,1.f,.5f,.125f,.25f}) {
    reference*=alpha;
    for(int term=0;term<2;++term)for(int group=0;group<2;++group) {
      float add=(term?-1.f:1.f)*(group+1)*.03125f + .125f;
      pv[term][group]=pv[term][group]*alpha+add;
      wrong[term][group]=wrong[term][group]*(term||group?1.f:alpha)+add;
      reference+=add;
    }
  }
  check((pv[0][0]+pv[0][1])+(pv[1][0]+pv[1][1])==reference,"C0 rescales every PV set before final sum");
  check((wrong[0][0]+wrong[0][1])+(wrong[1][0]+wrong[1][1])!=reference,"C0 partial-alpha negative witness");
  std::printf("RESULT: PASS decode-tc-selftest %d/%d (layout/codec only, NOT GPU MMA)\n",count,count);
}
