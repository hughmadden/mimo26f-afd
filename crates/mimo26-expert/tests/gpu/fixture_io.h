// First-party harness-only JSON/SHA256/safetensors reader. No external runtime.
// SHA256 follows FIPS180-4, like this repository's mimo26-repack implementation.
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace proof {
inline void need(bool ok,const std::string& msg) { if(!ok) throw std::runtime_error(msg); }
struct Json {
    char kind='n'; std::string text; std::vector<Json> array; std::map<std::string,Json> object;
    const Json& at(const std::string& key) const { need(kind=='o',"expected object"); return object.at(key); }
    const std::string& str() const { need(kind=='s',"expected string"); return text; }
    uint64_t num() const {
        need(kind=='d' && !text.empty() && text.find_first_not_of("0123456789")==std::string::npos,"expected unsigned integer");
        return std::stoull(text);
    }
    const std::vector<Json>& list() const { need(kind=='a',"expected array"); return array; }
};
class Parser {
    const std::string& s; size_t p=0,nodes=0;
    void ws() { while(p<s.size() && (s[p]==' ' || s[p]=='\n' || s[p]=='\r' || s[p]=='\t')) ++p; }
    char take() { need(p<s.size(),"truncated JSON"); return s[p++]; }
    std::string string() {
        need(take()=='"',"expected quote"); std::string out;
        for(;;) {
            const unsigned char c=take(); if(c=='"') return out;
            need(c>=32,"JSON control character");
            if(c!='\\') { out+=char(c); continue; }
            const char e=take();
            if(e=='"' || e=='\\' || e=='/') out+=e;
            else if(e=='b') out+='\b'; else if(e=='f') out+='\f';
            else if(e=='n') out+='\n'; else if(e=='r') out+='\r'; else if(e=='t') out+='\t';
            else throw std::runtime_error("unsupported JSON escape (ASCII checkpoint metadata required)");
        }
    }
    bool digit() const { return p<s.size() && s[p]>='0' && s[p]<='9'; }
    void digits() { need(digit(),"expected digit"); while(digit()) ++p; }
    Json value(unsigned depth) {
        need(depth<=64 && ++nodes<=2000000,"JSON resource bound"); ws(); need(p<s.size(),"missing value"); Json j;
        const char c=s[p];
        if(c=='"') { j.kind='s'; j.text=string(); return j; }
        if(c=='{' || c=='[') {
            ++p; j.kind=c=='{'?'o':'a'; const char end=c=='{'?'}':']'; ws();
            if(p<s.size() && s[p]==end) {++p;return j;}
            for(;;) {
                if(c=='{') {
                    ws(); const std::string key=string(); ws(); need(take()==':',"missing colon");
                    need(j.object.emplace(key,value(depth+1)).second,"duplicate JSON key");
                } else j.array.push_back(value(depth+1));
                ws(); const char next=take(); if(next==end) return j; need(next==',',"missing comma");
            }
        }
        for(const char* literal:{"true","false","null"}) {
            const std::string t=literal;
            if(s.compare(p,t.size(),t)==0) {p+=t.size();j.kind='n';j.text=t;return j;}
        }
        const size_t begin=p; if(c=='-') ++p; need(p<s.size(),"bad number");
        if(s[p]=='0') ++p; else digits();
        if(p<s.size() && s[p]=='.') {++p;digits();}
        if(p<s.size() && (s[p]=='e'||s[p]=='E')) {++p;if(p<s.size()&&(s[p]=='+'||s[p]=='-'))++p;digits();}
        j.kind='d'; j.text=s.substr(begin,p-begin); return j;
    }
public:
    explicit Parser(const std::string& text):s(text) {need(s.size()<=64*1024*1024,"JSON too large");}
    Json parse() { Json j=value(0); ws();need(p==s.size(),"trailing JSON");return j; }
};
inline Json parse(const std::string& text) { return Parser(text).parse(); }
inline std::vector<uint8_t> read(const std::string& path,uint64_t limit=64*1024*1024) {
    std::ifstream f(path,std::ios::binary|std::ios::ate); need(bool(f),"cannot read "+path);
    const auto size=f.tellg(); need(size>=0 && uint64_t(size)<=limit,"file size bound "+path);
    std::vector<uint8_t> data(static_cast<size_t>(size)); f.seekg(0);
    if(!data.empty()) {f.read(reinterpret_cast<char*>(data.data()),data.size());need(bool(f),"short read "+path);}
    return data;
}
inline Json read_json(const std::string& path) { const auto data=read(path);return parse(std::string(data.begin(),data.end())); }
inline std::string hex32(uint32_t v) { const char* h="0123456789abcdef";std::string s(8,'0');for(int i=7;i>=0;--i){s[i]=h[v&15];v>>=4;}return s; }
inline uint32_t rotr(uint32_t x,int r) {return (x>>r)|(x<<(32-r));}
inline std::string sha256(const std::vector<uint8_t>& input) {
    static const uint32_t k[]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> data=input; const uint64_t bits=uint64_t(data.size())*8;
    data.push_back(0x80); while(data.size()%64!=56)data.push_back(0);
    for(int i=7;i>=0;--i)data.push_back(uint8_t(bits>>(i*8)));
    for(size_t base=0;base<data.size();base+=64) {
        uint32_t w[64];
        for(int i=0;i<16;++i) {w[i]=0;for(int j=0;j<4;++j)w[i]=(w[i]<<8)|data[base+i*4+j];}
        for(int i=16;i<64;++i) {
            const uint32_t a=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            const uint32_t b=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+a+w[i-7]+b;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;++i) {
            const uint32_t t1=v+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            v=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
    std::string result; for(uint32_t v:h)result+=hex32(v);return result;
}
inline bool leaf_name(const std::string& name) {
    return !name.empty() && name!="." && name!=".." &&
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")==std::string::npos;
}
inline std::vector<uint8_t> tensor_bytes(const std::string& root,const Json& index,const std::string& name,
    uint64_t rows,uint64_t cols) {
    const std::string shard=index.at("weight_map").at(name).str();need(leaf_name(shard),"unsafe shard path");
    const std::string path=root+"/"+shard;
    std::ifstream f(path,std::ios::binary|std::ios::ate);need(bool(f),"open shard "+path);
    const auto end=f.tellg();need(end>=8,"short shard");const uint64_t size=uint64_t(end);f.seekg(0);
    uint8_t prefix[8];f.read(reinterpret_cast<char*>(prefix),8);need(bool(f),"short prefix");
    uint64_t n=0;for(int i=7;i>=0;--i)n=(n<<8)|prefix[i];
    need(n<=64*1024*1024 && n<=size-8,"bad header length");
    std::string text(size_t(n),' ');f.read(&text[0],n);need(bool(f),"short header");
    const Json header=parse(text);const Json& t=header.at(name);
    need(t.at("dtype").str()=="U8","wrong dtype "+name);
    const auto& shape=t.at("shape").list();
    need(shape.size()==2 && shape[0].num()==rows && shape[1].num()==cols,"wrong shape "+name);
    const auto& offsets=t.at("data_offsets").list();need(offsets.size()==2,"wrong offset count");
    const uint64_t lo=offsets[0].num(),hi=offsets[1].num();
    need(lo<=hi && hi<=size-8-n && hi-lo==rows*cols && hi-lo<=64*1024*1024,"bad tensor range "+name);
    std::vector<uint8_t> data(size_t(hi-lo));f.seekg(std::streamoff(8+n+lo));
    f.read(reinterpret_cast<char*>(data.data()),data.size());need(bool(f),"short tensor "+name);
    return data;
}
inline std::vector<uint8_t> tensor(const std::string& root,const Json& index,const std::string& name,
    uint64_t rows,uint64_t cols,const std::string& hash) {
    auto data=tensor_bytes(root,index,name,rows,cols);
    need(sha256(data)==hash,"SHA256 mismatch "+name);return data;
}
inline void validate_fixture(const Json& fixture) {
    const auto& blocks=fixture.at("blocks").list();need(blocks.size()==27,"expected27 blocks");std::set<std::string> names;
    for(const auto& b:blocks) {
        const std::string name=b.at("name").str();need(leaf_name(name) && names.insert(name).second,"bad/duplicate block name");
        const auto& shape=b.at("shape").list();need(shape.size()==2,"bad logical shape");
        const uint64_t r=shape[0].num(),c=shape[1].num();
        const bool down=name.size()>=10 && name.substr(name.size()-10)==".down_proj";
        const bool gate=name.size()>=10 && name.substr(name.size()-10)==".gate_proj";
        const bool up=name.size()>=8 && name.substr(name.size()-8)==".up_proj";
        need((down && r==4096 && c==2048) || ((gate||up) && r==2048 && c==4096),"bad fixture geometry");
        const auto& pos=b.at("positions").list();const auto& bits=b.at("expected_f32_bits").list();
        need(pos.size()==2048 && bits.size()==pos.size(),"expected2048 samples per block");
        for(size_t i=0;i<pos.size();++i) {
            const auto& p=pos[i].list();need(p.size()==2 && p[0].num()<r && p[1].num()<c,"sample OOB");
            const auto& word=bits[i].str();need(word.size()==8 && word.find_first_not_of("0123456789abcdef")==std::string::npos,"bad oracle bits");
        }
        for(const char* key:{"weight_sha256","scale_sha256"}) {
            const auto& h=b.at(key).str();need(h.size()==64 && h.find_first_not_of("0123456789abcdef")==std::string::npos,"bad hash");
        }
    }
}
inline void selftest() {
    const std::vector<std::pair<std::string,std::string>> vectors={
        {"","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc","ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq","248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000,'a'),"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
    for(const auto& v:vectors)need(sha256(std::vector<uint8_t>(v.first.begin(),v.first.end()))==v.second,"SHA256 vector");
    need(parse("{\"x\":[12,\"ok\"],\"y\":-1.25e+3}").at("x").list()[0].num()==12,"JSON positive");
    for(const char* bad:{"{\"x\":1,\"x\":2}","[1,]","[01]","[1e]","[","{}tail","{\"x\":\"\\q\"}"}) {
        bool rejected=false;try{parse(bad);}catch(const std::exception&){rejected=true;}need(rejected,"bad JSON accepted");
    }
    need(!leaf_name("../x") && !leaf_name("/x") && leaf_name("model_pp0_ep0_shard0.safetensors"),"path checks");
}
} // namespace proof
