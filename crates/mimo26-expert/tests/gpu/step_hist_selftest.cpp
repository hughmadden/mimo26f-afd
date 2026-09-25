#include "step_hist.h"
#include <iostream>
using namespace step_bench;
void selftest(){
    const std::string head="{\"test\":{\"steps\":[{\"layer\":1,\"hist\":";
    const std::string tail="}]}}";
    auto get=[&](const std::string& hist){return parse(proof::parse(head+hist+tail),"test").steps.at(0);};
    const auto all=get("{\"1\":256}");const auto baseline=schedule(all,0,true);validate(all,baseline);
    proof::need(baseline.size()==1&&baseline[0].rows==1&&baseline[0].experts.size()==256,"all-M1 is not one frozen-shaped load");
    proof::need(baseline[0].experts.front()==254 && baseline[0].experts[253]==255 && baseline[0].experts[254]==7 && baseline[0].experts.back()==0,"frozen reversed pool order");
    proof::need(distinct_bytes(all)==855638016 && scheduled_bytes(baseline)==855638016,"frozen all256 byte basis");
    const auto mix=get("{\"1\":8,\"9\":8,\"17\":8}");const auto good=schedule(mix,123);validate(mix,good);
    proof::need(good.size()==6&&mix.routes==216&&mix.experts==24,"tail scheduling");
    proof::need(distinct_bytes(mix)==24*expert_bytes&&scheduled_bytes(good)==48*expert_bytes,"distinct/reloaded byte conflation");
    proof::need(digest(good)==digest(schedule(mix,123)) && digest(good)!=digest(schedule(mix,124)),"seed determinism/randomization");
    std::set<std::string> fingerprints;
    for(int i=0;i<41;++i){auto run=schedule(mix,sample_seed(20260924,0,i));validate(mix,run);proof::need(fingerprints.insert(digest(run)).second,"warm/sample schedule reused");}
    for(int m=1;m<=128;++m){auto s=get("{\""+std::to_string(m)+"\":256}");auto plan=schedule(s,m);validate(s,plan);proof::need(plan.size()==size_t((m+7)/8)&&scheduled_bytes(plan)==distinct_bytes(s)*((m+7)/8),"M pass count/bytes");}
    unsigned negatives=0;
    auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}proof::need(failed,"unpowered step contract negative");++negatives;};
    for(const char* h:{"{}","[]","{\"0\":8}","{\"01\":8}","{\"-1\":8}","{\"1.0\":8}","{\"129\":8}","{\"1\":0}","{\"1\":257}","{\"1\":true}","{\"1\":1.0}","{\"1\":-1}","{\"1\":256,\"2\":1}","{\"1\":2,\"1\":3}"})reject([&]{get(h);});
    reject([&]{parse(proof::parse("{\"test\":{\"hist\":{\"1\":256}}}"),"test");});
    reject([&]{parse(proof::parse("{\"test\":{\"steps\":[]}}"),"test");});
    reject([&]{parse(proof::parse("{\"test\":{\"steps\":[{\"layer\":0,\"hist\":{\"1\":8}}]}}"),"test");});
    reject([&]{parse(proof::parse("{\"C1-w8\":{\"steps\":[{\"layer\":1,\"hist\":{\"1\":63}}]}}"),"C1-w8");});
    reject([&]{parse(proof::parse("{\"C1-w8\":{\"steps\":[{\"layer\":1,\"hist\":{\"16\":4}}]}}"),"C1-w8");});
    reject([&]{auto bad=good;bad.pop_back();validate(mix,bad);});
    reject([&]{auto bad=good;bad.push_back(bad[0]);validate(mix,bad);});
    reject([&]{auto bad=good;bad[0].experts[1]=bad[0].experts[0];validate(mix,bad);});
    reject([&]{auto bad=good;bad[0].experts[0]=256;validate(mix,bad);});
    reject([&]{auto bad=good;bad[1].experts[0]=bad[0].experts[0];validate(mix,bad);});
    reject([&]{auto bad=good;std::swap(bad[2].experts[0],bad[2].experts[1]);validate(mix,bad);});
    reject([&]{auto bad=good;bad[2].rows=8;validate(mix,bad);});
    reject([&]{auto bad=good;bad[2].pass=0;validate(mix,bad);});
    reject([&]{auto bad=good;bad[2].original_m=8;validate(mix,bad);});
    std::cout<<"HOST PASS step scheduling: all-M1 frozen shape/bytes,128 exact-M+tail cases,41 distinct reproducible schedules,"<<negatives<<" powered negatives; no GPU/timing claim\n";
}
int main(int argc,char** argv){try{
    selftest();
    if(argc==2&&std::string(argv[1])=="--selftest")return 0;
    proof::need(argc==3,"use --selftest or histogram.json workload");const auto w=parse(proof::read_json(argv[1]),argv[2]);
    std::cout<<"STEP INPUT workload="<<w.name<<" steps="<<w.steps.size()<<" seed="<<w.seed<<"\n";
    for(size_t i=0;i<w.steps.size();++i){const auto& s=w.steps[i];const auto plan=schedule(s,sample_seed(w.seed,i,10));validate(s,plan);
        std::cout<<"STEP PLAN ordinal="<<i<<" layer="<<s.layer<<" experts="<<s.experts<<" routes="<<s.routes<<" native_loads="<<plan.size()<<" distinct_bytes="<<distinct_bytes(s)<<" scheduled_bytes="<<scheduled_bytes(plan)<<" schedule_sha256="<<digest(plan)<<"\n";}
    return 0;
}catch(const std::exception& e){std::cerr<<"STEP INPUT FAILURE "<<e.what()<<"\n";return 2;}}
