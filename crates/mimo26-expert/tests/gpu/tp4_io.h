// Host-only validation of ACTUAL Rust repack outputs and independent full-matrix
// oracle artifacts. No CUDA calls and no production decoder used by this audit.
#pragma once
#include "fixture_io.h"
#include "mimo26_slice_layout.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <utility>
namespace proof {
inline const char* projection(int p) {return p==0?"gate_proj":p==1?"up_proj":"down_proj";}
inline size_t payload_offset(int p) {return p==0?M26X_GATE_PAYLOAD_OFF:p==1?M26X_UP_PAYLOAD_OFF:M26X_DOWN_PAYLOAD_OFF;}
inline size_t scale_offset(int p) {return p==0?M26X_GATE_SCALE_OFF:p==1?M26X_UP_SCALE_OFF:M26X_DOWN_SCALE_OFF;}
inline std::string tag(int layer,int expert) {
    char text[32];std::snprintf(text,sizeof(text),"L%02d_E%03d",layer,expert);return text;
}
inline std::string tensor_prefix(int layer,int expert) {
    return "model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+".";
}
inline std::vector<float> floats(const std::string& file,size_t count) {
    const auto bytes=read(file);need(bytes.size()==count*4,"wrong float-file length "+file);
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    std::vector<float> values(count);
    for(size_t i=0;i<count;++i) {
        const uint32_t b=uint32_t(bytes[4*i]) | (uint32_t(bytes[4*i+1])<<8) |
            (uint32_t(bytes[4*i+2])<<16) | (uint32_t(bytes[4*i+3])<<24);
        std::memcpy(&values[i],&b,4);need(std::isfinite(values[i]),"nonfinite oracle/input "+file);
    }
    return values;
}
inline void layout_marker(const std::string& root) {
    const auto bytes=read(root+"/layout.version",16);
    need(std::string(bytes.begin(),bytes.end())=="2\n","staged layout marker must be v2");
}
inline void manifest_contract(const Json& manifest,int layer,int expert) {
    need(manifest.at("format").str()=="mimo26-repack-manifest" && manifest.at("version").num()==2,"manifest must be v2");
    for(const auto& kv:std::map<std::string,uint64_t>{{"ep_ranks",4},{"expert_bytes",13369344},
        {"experts_per_layer",256},{"hidden",4096},{"intermediate",2048},{"moe_layers",47},{"quarter_slice_bytes",3342336}})
        need(manifest.at("geometry").at(kv.first).num()==kv.second,"manifest geometry mismatch");
    const auto& layout=manifest.at("layout").list();need(layout.size()==6,"manifest layout size");
    for(int p=0;p<3;++p) for(int region=0;region<2;++region) {
        const auto& d=layout[2*p+region];
        need(d.at("proj").str()==projection(p) && d.at("region").str()==(region?"scales":"payload") &&
            d.at("off").num()==(region?scale_offset(p):payload_offset(p)) &&
            d.at("len").num()==uint64_t(region?65536:1048576),"manifest layout offset/length mismatch");
    }
    const auto& slices=manifest.at("slices").list();need(slices.size()==4,"need four rank entries");
    std::set<uint64_t> seen;
    for(const auto& s:slices) {
        const uint64_t rank=s.at("rank").num();need(rank<4 && seen.insert(rank).second,"duplicate/invalid rank");
        need(s.at("layer").num()==uint64_t(layer) && s.at("expert").num()==uint64_t(expert),"wrong expert identity");
        need(s.at("file").str()==tag(layer,expert)+"_R"+std::to_string(rank)+".slice","noncanonical slice filename");
        need(s.at("bytes").num()==3342336,"wrong declared slice length");
        need(s.at("source").at("shard").str()=="model_pp0_ep"+std::to_string(expert/4)+"_shard0.safetensors","wrong source shard");
        const auto& tensors=s.at("source").at("tensors").list();need(tensors.size()==6,"wrong source descriptor count");
        for(int p=0;p<3;++p) for(int scale=0;scale<2;++scale) {
            const auto& t=tensors[2*p+scale];const auto& shape=t.at("shape").list();
            need(t.at("name").str()==tensor_prefix(layer,expert)+projection(p)+(scale?".weight_scale":".weight") &&
                t.at("dtype").str()=="U8" && shape.size()==2 && shape[0].num()==uint64_t(p==2?4096:2048) &&
                shape[1].num()==uint64_t(p==2?(scale?64:1024):(scale?128:2048)),"wrong source tensor descriptor");
        }
    }
}
struct Tp4Case {
    std::array<std::vector<uint8_t>,4> image;
    std::array<std::vector<float>,4> partial,down;
    std::vector<float> gate,up,h,y;
};
inline Tp4Case load_case(const std::string& root,const Json& fixture,int layer,int expert) {
    const std::string name=tag(layer,expert);const Json manifest=read_json(root+"/"+name+".manifest.json");
    manifest_contract(manifest,layer,expert);Tp4Case c;
    for(const auto& entry:manifest.at("slices").list()) {
        const auto rank=entry.at("rank").num();auto bytes=read(root+"/"+entry.at("file").str());
        need(bytes.size()==3342336 && sha256(bytes)==entry.at("sha256").str(),"slice size/SHA mismatch "+name);
        c.image[rank]=std::move(bytes);
    }
    for(int p=0;p<3;++p) {
        const std::string prefix=tensor_prefix(layer,expert)+projection(p);const Json* b=nullptr;
        for(const auto& block:fixture.at("blocks").list()) if(block.at("name").str()==prefix)b=&block;
        need(b!=nullptr,"missing oracle block");
        for(int scale=0;scale<2;++scale) {
            const auto full=read(root+"/"+name+"."+projection(p)+(scale?".s":".w"));
            const int divisor=scale?32:2,full_rows=p==2?4096:2048,full_cols=p==2?2048:4096;
            need(full.size()==size_t(full_rows)*full_cols/divisor &&
                sha256(full)==b->at(scale?"scale_sha256":"weight_sha256").str(),"raw full source SHA mismatch");
            // Literal source-coordinate rectangle formula, independent of repack
            // functions and not inferred from any GPU-computed answer.
            for(int rank=0;rank<4;++rank) {
                const int rows=p==2?4096:512,cols=p==2?512:4096;
                for(int row=0;row<rows;++row) {
                    const int source_row=p==2?row:rank*512+row,source_col=p==2?rank*512:0;
                    const auto* want=full.data()+size_t(source_row)*(full_cols/divisor)+source_col/divisor;
                    const auto* got=c.image[rank].data()+(scale?scale_offset(p):payload_offset(p))+size_t(row)*(cols/divisor);
                    need(std::memcmp(want,got,cols/divisor)==0,"slice source-coordinate rectangle mismatch");
                }
            }
        }
    }
    c.gate=floats(root+"/"+name+".gate.f32",64*2048);c.up=floats(root+"/"+name+".up.f32",64*2048);
    c.h=floats(root+"/"+name+".h.f32",64*2048);c.y=floats(root+"/"+name+".y.f32",64*4096);
    for(int rank=0;rank<4;++rank) {
        c.partial[rank]=floats(root+"/"+name+".partial_R"+std::to_string(rank)+".f32",64*4096);
        c.down[rank]=floats(root+"/"+name+".down_R"+std::to_string(rank)+".f32",64*4096);
    }
    for(size_t i=0;i<c.y.size();++i) {
        double sum=0;for(int rank=0;rank<4;++rank)sum+=c.partial[rank][i];
        need(std::abs(sum-c.y[i])<=1e-5+1e-5*std::abs(double(c.y[i])),"independent partial goldens do not sum to full oracle");
    }
    return c;
}
inline int tp4_audit(const std::string& root,const std::string& path) {
    layout_marker(root);const auto fixture=read_json(path);validate_fixture(fixture);
    const auto x=floats(root+"/x.f32",64*4096);(void)x;
    for(int layer:{1,24,46})for(int expert:{0,7,255}) {
        const auto c=load_case(root,fixture,layer,expert);(void)c;
        std::printf("TP4 HOST AUDIT %s: four v2 slice SHAs; six full source SHAs; all rectangle bytes; finite goldens\n",tag(layer,expert).c_str());
    }
    std::puts("TP4 HOST PASS:36 Rust-produced images +54 source hashes, all ranks/projections byte-exact rectangles, independent partial sums");return 0;
}
} // namespace proof
