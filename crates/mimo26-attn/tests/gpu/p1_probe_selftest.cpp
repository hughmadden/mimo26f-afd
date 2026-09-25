#include "p1_probe_check.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
int main(){
  int checks=0;auto check=[&](bool ok){++checks;if(!ok)std::exit(1);};
  std::vector<float> expected={12345,.5f,NAN,0,12345,12345};
  auto verify=[&](const std::vector<float>& v){return p1_probe_matches(v,expected,1,3);};
  check(verify(expected));
  for(int at=0;at<6;++at){auto v=expected;v[at]=at==2?0.f:42.f;check(!verify(v));}
  auto v=expected;v[1]=NAN;check(!verify(v)); // masked 0*NaN / whole-CTA poisoning
  v=expected;v[2]=INFINITY;check(!verify(v));
  v=expected;v[1]=0;check(!verify(v)); // zero-output greenwash
  v=expected;v[1]+=.0001f;check(!verify(v));
  v=expected;v[1]+=.000001f;check(verify(v));
  v=expected;v.pop_back();check(!verify(v));
  check(!p1_probe_matches(expected,expected,7,0));
  check(!p1_probe_matches(expected,expected,1,std::numeric_limits<size_t>::max()));
  std::printf("RESULT: PASS P1 probe checker %d checks (CPU only)\n",checks);
}
