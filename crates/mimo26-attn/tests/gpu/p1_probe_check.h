#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
// Full scans: exact sentinels outside the valid output extent, finite 1e-5
// inside, explicit NaN only where the independent fail-closed expectation says.
inline bool p1_probe_matches(const std::vector<float>& got,const std::vector<float>& expected,
                             size_t begin,size_t count) {
  if(got.size()!=expected.size()||begin>expected.size()||count>expected.size()-begin)return false;
  bool ok=true;
  for(size_t i=0;i<got.size();++i) {
    if(i<begin||i>=begin+count)ok&=std::isfinite(got[i])&&got[i]==expected[i];
    else if(std::isnan(expected[i]))ok&=std::isnan(got[i]);
    else ok&=std::isfinite(got[i])&&std::abs(got[i]-expected[i])<=1e-5f;
  }
  return ok;
}
