#include "step_prefill_plan.h"
#include <functional>
int main(int argc,char** argv){try{
    using namespace step_bench;int negatives=0;
    auto reject=[&](const std::function<void()>& f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}proof::need(bad,"unpowered F5 negative");++negatives;};
    for(int m:{1,16,64}){
        Step s{1,1,{{m,256}},256,m*256};std::set<std::string> seen;
        for(int j=0;j<41;++j){const auto seed=sample_seed(20260924,0,j);auto p=schedule_prefill(s,seed);validate_prefill(s,p);
            proof::need(p[0].experts==schedule(s,seed)[0].experts,"F5 changed deterministic expert ordering");
            proof::need(distinct_bytes(s)==855638016&&scheduled_bytes(p)==855638016,"F5 native-plan byte accounting");seen.insert(digest(p));}
        proof::need(seen.size()==41,"F5 random schedule reuse");auto p=schedule_prefill(s,7);
        auto bad=p;bad[0].rows=0;reject([&]{validate_prefill(s,bad);});bad=p;bad[0].experts[0]=bad[0].experts[1];reject([&]{validate_prefill(s,bad);});
        bad=p;bad[0].experts[0]=256;reject([&]{validate_prefill(s,bad);});bad=p;bad[0].pass=1;reject([&]{validate_prefill(s,bad);});
        if(m>8){bad=schedule(s,7);reject([&]{validate_prefill(s,bad);});}
        auto wrong=s;wrong.layer=2;reject([&]{prefill_shape(wrong);});wrong=s;wrong.routes-=1;reject([&]{prefill_shape(wrong);});
    }
    Step too_wide{1,1,{{128,256}},256,32768};reject([&]{schedule_prefill(too_wide,1);});
    std::printf("F5 HOST PASS all256 M1/M16/M64 native plans,41 schedules each, %d powered negatives; no GPU claim\n",negatives);
    proof::need(argc==1||argc==3,"F5 host expects optional histogram workload");
    if(argc==3){const auto w=parse(proof::read_json(argv[1]),argv[2]);proof::need(w.steps.size()==1,"F5 one step");auto p=schedule_prefill(w.steps[0],w.seed);std::printf("F5 HOST PLAN workload=%s native_M=%d experts=256 routes=%d host_passes=1 digest=%s\n",w.name.c_str(),p[0].rows,w.steps[0].routes,digest(p).c_str());}
}catch(const std::exception& e){std::fprintf(stderr,"F5 HOST FAILURE %s\n",e.what());return 2;}}
