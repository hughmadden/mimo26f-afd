// Synthetic full-shape routing fixture. Independent analytic FFN oracle;
// supplements, never replaces, the real-checkpoint TP4 proof.
#pragma once
#include "fixture_io.h"
#include <array>
#include <cmath>
#include <cstring>
namespace proof {
struct RoutingCase {
    int capacity,m,tokens;
    std::vector<int32_t> ids,offsets,sources;
};
inline RoutingCase routing_case(int capacity) {
    need(capacity==256 || capacity==2048 || capacity==4096,"invalid routing capacity");
    RoutingCase c{capacity,capacity/32,8*capacity,{},{0},{}};
    for(int group=0;group<256;++group) {
        const int expert=(73*group+19)%256;c.ids.push_back(expert);
        for(int j=0;j<c.m;++j)c.sources.push_back(expert/8+32*j);
        c.offsets.push_back(int(c.sources.size()));
    }
    return c;
}
inline float routing_x(int source) {return float(1+source%13)*0.125f;}
inline double routing_oracle(int expert,int source,int row) {
    if(row>=8)return 0;
    const double x=routing_x(source),gate=x*((expert&(1<<row))?2:1);
    return (gate/(1+std::exp(-gate)))*x;
}
inline std::vector<uint8_t> routing_image(int expert) {
    need(expert>=0 && expert<256,"invalid synthetic expert");
    // Literal v2 offsets, not imported decoder/index helpers. All scale bytes
    // equal127 (scale1). Only24 payload bytes are nonzero in each full image.
    std::vector<uint8_t> image(3342336,0);
    for(size_t off:{1048576u,2162688u,3276800u})std::fill_n(image.begin()+off,65536,uint8_t(127));
    for(int row=0;row<8;++row) {
        image[size_t(row)*2048]=uint8_t((expert&(1<<row))?4:2); // gate column0 =2 or1
        image[1114112+size_t(row)*2048]=2; // up column0 =1
        image[2228224+size_t(row)*256+row/2]=uint8_t(2<<((row%2)*4)); // down identity on first8 columns
    }
    return image;
}
inline double reference_value(const std::vector<uint8_t>& image,int projection,int row,int col) {
    constexpr size_t offsets[]={0,1114112,2228224};
    const int cols=projection==2?512:4096;
    const auto byte=image[offsets[projection]+size_t(row)*cols/2+col/2];
    const unsigned code=(byte>>((col%2)*4))&15;
    constexpr double lut[]={0,.5,1,1.5,2,3,4,6};
    return (code&8?-1:1)*lut[code&7]; // fixture scales are verified separately as127
}
inline void routing_selftest() {
    for(int capacity:{256,2048,4096}) {
        const auto c=routing_case(capacity);std::vector<unsigned> masks(capacity,0);
        std::set<int> experts;
        need(c.ids.size()==256 && c.offsets.size()==257 && c.offsets.back()==8*capacity,"routing shape");
        for(int g=0;g<256;++g) {
            const int expert=c.ids[g];need(experts.insert(expert).second,"duplicate resident ID");
            need(c.offsets[g+1]-c.offsets[g]==capacity/32,"group width");
            for(int i=c.offsets[g];i<c.offsets[g+1];++i) {
                const int source=c.sources[i];need(source>=0 && source<capacity,"source token OOB");
                need(expert/8==source%32,"wrong source/expert routing relation");
                const unsigned bit=1u<<(expert%8);need(!(masks[source]&bit),"duplicate top8 expert");masks[source]|=bit;
            }
        }
        for(auto mask:masks)need(mask==255,"missing top8 route");
    }
    std::set<unsigned> signatures;
    for(int expert=0;expert<256;++expert) {
        const auto image=routing_image(expert);unsigned signature=0;
        for(int p=0;p<3;++p) {
            const size_t wo=p==0?0:p==1?1114112:2228224,so=p==0?1048576:p==1?2162688:3276800;
            need(std::count_if(image.begin()+wo,image.begin()+wo+1048576,[](uint8_t x){return x!=0;})==8,"payload nonzero count");
            need(std::all_of(image.begin()+so,image.begin()+so+65536,[](uint8_t x){return x==127;}),"synthetic scale bytes");
        }
        for(int row=0;row<8;++row) {
            const double w=reference_value(image,0,row,0);if(w==2)signature|=1u<<row;
            need(w==((expert&(1<<row))?2:1) && reference_value(image,1,row,0)==1,"gate/up synthetic encoding");
            for(int k=0;k<8;++k)need(reference_value(image,2,row,k)==double(k==row),"down identity encoding");
            for(int source:{0,6,12}) {
                const double x=routing_x(source),gate=w*x;
                const double from_image=gate/(1+std::exp(-gate))*x;
                need(std::abs(from_image-routing_oracle(expert,source,row))<1e-12,"analytic oracle disagrees with decoded fixture");
            }
        }
        need(signature==unsigned(expert) && signatures.insert(signature).second,"resident signatures not unique");
    }
    // Numerical discriminator must reject group ordinal in place of explicit ID.
    const auto c=routing_case(256);int mismatches=0;
    for(int g=0;g<256;++g)for(int row=0;row<8;++row)
        if(std::abs(routing_oracle(g,c.sources[c.offsets[g]],row)-routing_oracle(c.ids[g],c.sources[c.offsets[g]],row))>1e-5)++mismatches;
    need(mismatches>0,"routing oracle cannot detect ordinal substitution");
    std::printf("HOST PASS routing:3 capacity boundaries,256 unique full-size expert images,top8 coverage,ordinal negative=%d mismatches\n",mismatches);
}
} // namespace proof
