// Harness-only R18 scheduling contract; no kernel or numerical-policy changes.
#pragma once
#include "fixture_io.h"
#include <numeric>
namespace step_bench {
constexpr uint64_t expert_bytes=3342336;
struct Step {int layer,route_layer;std::map<int,int> hist;int experts=0,routes=0;};
struct Workload {std::string name;uint64_t seed=20260924;std::vector<Step> steps;};
struct Load {int original_m,rows,pass;std::vector<int> experts;};
inline int integer(const proof::Json& j,int low,int high,const char* label){
    const auto n=j.num();proof::need(n>=uint64_t(low)&&n<=uint64_t(high),label);return int(n);
}
inline Workload parse(const proof::Json& root,const std::string& name){
    const auto& w=root.at(name);proof::need(w.kind=='o',"workload must be an object");
    proof::need(w.object.count("steps"),"per-layer-step histograms required; aggregate hist cannot reconstruct simultaneous expert loads");
    Workload out;out.name=name;if(w.object.count("seed"))out.seed=w.at("seed").num();
    const auto& steps=w.at("steps").list();proof::need(!steps.empty()&&steps.size()<=100000,"step count must be1..100000");
    const int tokens=name=="C1-w8"?8:name=="C4-w8"?32:name=="C16-w8"?128:0;
    for(const auto& raw:steps){
        Step s{};s.layer=integer(raw.at("layer"),1,47,"MoE layer must be1..47");
        s.route_layer=raw.object.count("route_layer")?integer(raw.at("route_layer"),1,47,"route layer must be1..47"):s.layer;
        const auto& h=raw.at("hist");proof::need(h.kind=='o'&&!h.object.empty(),"empty/non-object step histogram");
        for(const auto& item:h.object){
            const auto& key=item.first;proof::need(!key.empty()&&key[0]!='0'&&key.find_first_not_of("0123456789")==std::string::npos,"noncanonical M key");
            const auto wide=std::stoull(key);proof::need(wide>=1&&wide<=128,"M must be1..128");const int m=int(wide);
            const int count=integer(item.second,1,256,"active count must be1..256");
            proof::need(s.hist.emplace(m,count).second,"duplicate M");s.experts+=count;s.routes+=m*count;
            proof::need(s.experts<=256,"more than256 distinct experts in one layer-step");
            if(tokens)proof::need(m<=tokens,"M exceeds workload token count");
        }
        if(tokens)proof::need(s.routes==tokens*8,"route count differs from top8 workload contract");
        out.steps.push_back(s);
    }
    return out;
}
inline uint64_t random64(uint64_t& state){
    uint64_t z=(state+=0x9e3779b97f4a7c15ull);z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
    z=(z^(z>>27))*0x94d049bb133111ebull;return z^(z>>31);
}
inline uint64_t sample_seed(uint64_t seed,uint64_t step,uint64_t sample){
    seed^=(step+1)*0xd1b54a32d192ed03ull;seed^=(sample+1)*0x94d049bb133111ebull;return random64(seed);
}
inline std::vector<Load> schedule(const Step& s,uint64_t seed,bool frozen_order=false){
    std::vector<int> ids(256);std::iota(ids.begin(),ids.end(),0);
    if(frozen_order){
        // Frozen B2 launch reverses pool slots; pool order is0,7,255,1,2,...
        ids={0,7,255};for(int e=1;e<255;++e)if(e!=7)ids.push_back(e);std::reverse(ids.begin(),ids.end());
    }else{
        // Explicit algorithm: identical assignments in B1/B2 and across STL versions.
        for(int i=255;i>0;--i){const int j=int(random64(seed)%uint64_t(i+1));std::swap(ids[i],ids[j]);}
    }
    std::vector<Load> out;size_t used=0;
    for(const auto& bin:s.hist){const int m=bin.first,n=bin.second;
        std::vector<int> chosen(ids.begin()+used,ids.begin()+used+n);used+=n;
        for(int first=0,pass=0;first<m;first+=8,++pass)out.push_back({m,std::min(8,m-first),pass,chosen});
    }
    return out;
}
inline void validate(const Step& s,const std::vector<Load>& loads){
    std::set<int> distinct;size_t at=0;int routes=0;
    for(const auto& bin:s.hist){std::vector<int> chosen;
        for(int first=0,pass=0;first<bin.first;first+=8,++pass){
            proof::need(at<loads.size(),"missing native pass");const auto& l=loads[at++];
            proof::need(l.original_m==bin.first&&l.rows==std::min(8,bin.first-first)&&l.pass==pass,"wrong exact-M pass/tail");
            proof::need(l.experts.size()==size_t(bin.second),"wrong active expert count");
            std::set<int> unique;
            for(int e:l.experts)proof::need(e>=0&&e<256&&unique.insert(e).second,"duplicate/out-of-range expert per load");
            if(!pass){chosen=l.experts;for(int e:chosen)proof::need(distinct.insert(e).second,"expert assigned to different original-M bins");}
            else proof::need(chosen==l.experts,"expert mapping changed between passes of one original M");
            routes+=l.rows*int(l.experts.size());
        }
    }
    proof::need(at==loads.size()&&int(distinct.size())==s.experts&&routes==s.routes,"step coverage/accounting mismatch");
}
inline uint64_t distinct_bytes(const Step& s){return uint64_t(s.experts)*expert_bytes;}
inline uint64_t scheduled_bytes(const std::vector<Load>& loads){uint64_t count=0;for(const auto& l:loads)count+=l.experts.size();return count*expert_bytes;}
inline std::string digest(const std::vector<Load>& loads){
    std::vector<uint8_t> bytes;
    auto append=[&](uint32_t v){for(int k=0;k<4;++k)bytes.push_back(uint8_t(v>>(8*k)));};
    for(const auto& l:loads){append(l.original_m);append(l.rows);append(l.pass);append(uint32_t(l.experts.size()));for(int e:l.experts)append(e);}
    return proof::sha256(bytes);
}
}
