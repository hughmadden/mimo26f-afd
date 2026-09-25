#include "../../kernels/bench/metrics.h"
#include "../../kernels/include/decode_tc_layout.h"
#include <cstdio>
#include <limits>

int main() {
  int count = 0;
  auto check = [&](bool ok, const char* name) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", name); std::exit(1); }
    ++count;
  };
  check(bench::tc_default_splits(131072)==85,"128K exact-wave default");
  check(bench::tc_default_splits(1048576)==510,"1M exact-wave default");
  for(int p: {85,255,510})check(bench::tc_wave_fill(p)==1.0,"full-wave gate point");
  for(int p: {64,128,256,512})check(bench::tc_wave_fill(p)<1.0,"partial-wave diagnostic");
  check(bench::kv_bytes(1048576, 4) == 1342177280ULL, "one-layer bytes");
  check(bench::kv_bytes(1048576, 4) * 9 == 12079595520ULL, "nine GA layer bytes");
  check(bench::pairs(2048, 2048) == 2098176, "causal triangular count");
  check(bench::pairs(2048, 32768) == 65012736, "prefix plus causal triangle");
  check(bench::pairs(2048, 32768, 128) == 262144, "SWA useful work");
  check(bench::pairs(2048, 2048, 128) == 254016, "SWA short prefix triangle");
  check(bench::flops(1, 1048576) == 42949672960.0, "QK plus PV FLOPs");
  check(bench::gbs(1253000000, 1) == 1253, "GB/s decimal units");
  check(bench::tflops(100e9, 1) == 100, "TFLOPS decimal units");
  check(bench::median({3,1,2}) == 2, "odd median");
  check(bench::median({4,1,3,2}) == 2.5, "even median");
  for (auto samples : {std::vector<float>{}, std::vector<float>{0},
       std::vector<float>{std::numeric_limits<float>::quiet_NaN()}}) {
    bool rejected = false;
    try { bench::median(samples); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid measurement refuses");
  }
  check(bench::memory_fits(bench::reserve_bytes + 8, 8), "memory boundary");
  check(!bench::memory_fits(bench::reserve_bytes + 7, 8), "reserve protected");
  check(!bench::memory_fits(1, 8), "no unsigned underflow");
  check(bench::aot_matches(120, 170, 120, 170), "5090 positive identity");
  check(!bench::aot_matches(89, 170, 120, 170), "wrong arch refuses");
  check(!bench::aot_matches(120, 188, 120, 170), "wrong SM count refuses");
  check(bench::tc_decode_mma_flops(131072,128) == 13958643712.0, "Q3/P2 executed work");
  check(bench::tc_decode_mma_flops(131072,85) == 14193786880.0, "nonaligned split padding");
  check(bench::tc_decode_mma_flops(131072,128,false)==bench::flops(1,131072)*1.4,"native Q1/P2 factor");
  check(bench::tc_decode_mma_flops(131072,85,false)==57344.0*85*49*32,"native split padding");
  check(bench::tc_decode_mma_flops(1048576,255,false)==57344.0*255*129*32,"native 1M P255 padding");
  check(bench::tc_decode_mma_flops(1048576,510,false)==57344.0*510*65*32,"native 1M P510 padding");
  check(bench::tc_decode_mma_flops(1,8) == 3407872.0, "empty splits do no MMA");
  // C1 N16 padding differs from C3; 1M/P255 has 239 short + 16 long splits.
  for(bool residual:{false,true}) {
    double factor=residual?106496.0:57344.0;
    check(bench::tc_decode_mma_flops(131072,85,residual,16)==factor*85*97*16,"C1 128K/P85 padding");
    check(bench::tc_decode_mma_flops(1048576,255,residual,16)==factor*(239*257+16*258)*16,"C1 1M/P255 mixed tile counts");
    check(bench::tc_decode_mma_flops(1048576,510,residual,16)==factor*510*129*16,"C1 1M/P510 padding");
  }
  check(bench::tc_decode_mma_flops(1,8,true,16)==106496.0*16,"C1 empty splits do no MMA");
  for(int tile:{0,8,64}) {
    bool rejected=false;
    try{bench::tc_decode_mma_flops(128,8,true,tile);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"reject unsupported decode MMA tile");
  }
  for (int bad : {0,-1}) {
    bool rejected=false;
    try { bench::tc_decode_mma_flops(128,bad); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid MMA split count");
  }
  for(auto shape : {std::pair<int,int>{64,64},{32,32},{32,64}}) {
    int m=shape.first,n=shape.second;
    std::vector<int> scores(m*n),outputs(m*128);
    for(int warp=0;warp<m/8;++warp) for(int lane=0;lane<32;++lane) {
      int mr=warp/2*16,nc=warp%2*(n/2);
      for(int f=0;f<n/16;++f) for(int e=0;e<4;++e)
        ++scores.at((mr+m26tc::c_row(lane,e))*n+nc+f*8+m26tc::c_col(lane,e));
      for(int c=0;c<8;++c) for(int e=0;e<4;++e)
        ++outputs.at((mr+m26tc::c_row(lane,e))*128+(warp%2)*64+c*8+m26tc::c_col(lane,e));
    }
    check(std::all_of(scores.begin(),scores.end(),[](int x){return x==1;}),"eta score ownership");
    check(std::all_of(outputs.begin(),outputs.end(),[](int x){return x==1;}),"eta output ownership");
  }
  check(bench::eta_shared_bytes(64,64)==95232,"GA micro shared budget");
  check(bench::eta_shared_bytes(32,32)==45568,"SWA micro shared budget");
  check(bench::eta_shared_bytes(32,64)==78336,"alternate micro shared budget");
  check(bench::eta_mma_flops(64,64,true)==6815744,"eta Q3/P2 work");
  check(bench::eta_mma_flops(64,64,false)==3670016,"eta Q1/P2 work");
  std::printf("RESULT: PASS bench-selftest %d/%d\n", count, count);
}
