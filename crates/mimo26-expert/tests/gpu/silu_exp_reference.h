// Independent diagnostic reference. Not a serving implementation or lattice change.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cfenv>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <limits>
#include <string>
namespace exp_pin {
inline void need(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
inline float f32(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
inline uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
inline double f64(uint64_t b){double x;std::memcpy(&x,&b,8);return x;}
inline uint64_t bits64(double x){uint64_t b;std::memcpy(&b,&x,8);return b;}
inline bool finite(uint32_t b){return (b&0x7fffffff)<0x7f800000;}
inline float add(float a,float b){volatile float r=a+b;return r;}
inline float mul(float a,float b){volatile float r=a*b;return r;}
inline float div(float a,float b){volatile float r=a/b;return r;}
inline float activation(float g,float u,float e){const float d=add(1.f,e);return mul(g>=0?div(g,d):div(mul(g,e),d),u);}
struct Ref {uint32_t value;bool near=false;double y=0,mid=0,gap=0;};
// Certified constant regions have wide margins; Decimal checker independently
// verifies their boundary values. Remaining inputs use binary64 exp then RN32.
inline Ref reference(uint32_t b){
    const uint32_t a=b&0x7fffffff;
    if(a>0x7f800000)return {0x7fc00000}; // NaN classification, payload not a contract.
    if(a==0x7f800000)return {b>>31?0u:0x7f800000u};
    const float x=f32(b);
    if(x>90.f)return {0x7f800000};
    if(x<-105.f)return {0};
    if(a<0x32800000)return {0x3f800000}; // |x| < 2^-26.
    const double y=std::exp(double(x));
    const uint32_t c=bits(float(y));
    // Virtual successor2^128 represents the overflow rounding boundary.
    const double center=c==0x7f800000?std::ldexp(1.,128):double(f32(c));
    const double lower=c?double(f32(c-1)):0.;
    const double upper=c>=0x7f7fffff?std::ldexp(1.,128):double(f32(c+1));
    const double ml=(lower+center)*.5,mu=(center+upper)*.5;
    const double midpoint=c==0x7f800000?ml:c==0?mu:std::abs(y-ml)<std::abs(y-mu)?ml:mu;
    const double ulp=f64(bits64(y)+1)-y;
    const double distance=std::abs(y-midpoint);
    return {c,distance<=256*ulp,y,midpoint,distance/ulp};
}
inline std::map<uint32_t,uint32_t> corrections(const std::string& path){
    std::map<uint32_t,uint32_t> out;if(path=="-")return out;
    std::ifstream f(path);need(bool(f),"correction file missing");std::string line;
    while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;std::istringstream s(line);uint32_t x,y;s>>std::hex>>x>>y;need(bool(s)&&out.emplace(x,y).second,"malformed/duplicate correction");}
    return out;
}
inline uint32_t corrected(uint32_t b,const Ref& r,const std::map<uint32_t,uint32_t>& fix){
    if(!r.near)return r.value;const auto it=fix.find(b);return it==fix.end()?r.value:it->second;
}
inline uint64_t ulps(uint32_t a,uint32_t b){
    auto ordered=[](uint32_t v)->uint32_t{return v>>31?~v:v^0x80000000u;};
    const uint64_t x=ordered(a),y=ordered(b);return x>y?x-y:y-x;
}
inline void selftest(){
    need(std::fegetround()==FE_TONEAREST,"reference needs RNE");
    volatile float minimum=f32(1),unity=1.f;
    need(bits(mul(minimum,unity))==1,"host FTZ/DAZ detected");
    need(reference(0).value==0x3f800000 && reference(0x80000000).value==0x3f800000,"signed zero exp");
    need(reference(0x7f800000).value==0x7f800000 && reference(0xff800000).value==0,"infinities");
    need(reference(0x7fc00001).value==0x7fc00000,"NaN classification");
    need(reference(bits(-104.f)).value==0 && reference(bits(-103.f)).value==1,"subnormal/zero exp");
    need(reference(bits(90.f)).value==0x7f800000,"overflow");
    need(reference(bits(-0.1987401247024536f)).value==1062329339u,"E184 witness exp");
    const auto r=reference(bits(-0.1987401247024536f));
    need(bits(activation(-0.1987401247024536f,.5958439111709595f,f32(r.value)))==bits(-.0533447265625f),"E184 activation");
    need(bits(activation(-0.f,1.f,1.f))==0x80000000,"negative zero activation");
    need(ulps(0x3f800000,0x3f800001)==1 && ulps(0x80000001,0x80000002)==1,"ULP order");
    need(reference(0x33800000).near,"midpoint-adjacent case missing");
    need(bits(activation(-0.1987401247024536f,.5958439111709595f,f32(r.value+1)))!=bits(-.0533447265625f),"wrong exp negative powerless");
    std::puts("HOST PASS exp reference: specials, gradual underflow, RNE, signed zero, E184 and powered one-ULP exp negative");
}
}
