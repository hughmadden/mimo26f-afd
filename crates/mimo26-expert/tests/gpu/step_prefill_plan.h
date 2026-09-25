// F5-only native wide plans. The qualified split8 routing scheduler is unchanged.
#pragma once
#include "step_hist.h"
namespace step_bench {
inline void prefill_shape(const Step& s){
    proof::need(s.layer==1&&s.route_layer==1&&s.hist.size()==1,"F5 requires one frozen weight-layer1 bin");
    const int m=s.hist.begin()->first;
    proof::need((m==1||m==16||m==64)&&s.hist.begin()->second==256&&s.experts==256&&s.routes==256*m,"F5 shape must be all256 M1/M16/M64");
    proof::need(s.routes<=8*2048,"F5 exceeds frozen class2048 route capacity");
}
inline void validate_prefill(const Step& s,const std::vector<Load>& p){
    prefill_shape(s);proof::need(p.size()==1,"F5 must be one native plan, not split host passes");const auto& l=p[0];
    proof::need(l.original_m==s.hist.begin()->first&&l.rows==l.original_m&&l.pass==0&&l.experts.size()==256,"wrong F5 native rows/count");
    std::set<int> ids;for(int e:l.experts)proof::need(e>=0&&e<256&&ids.insert(e).second,"F5 duplicate/out-of-range expert");
}
inline std::vector<Load> schedule_prefill(const Step& s,uint64_t seed){
    prefill_shape(s);auto split=schedule(s,seed);Load wide=split.front();wide.rows=wide.original_m;wide.pass=0;
    std::vector<Load> out{wide};validate_prefill(s,out);return out;
}
}
