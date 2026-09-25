// R13 isolated asynchronous scale-copy candidate. Prepared ABI/layout unchanged.
// Local derivative of the approved staging.cuh unit (b12x provenance/hash there).
// Apache-2.0; see LICENSE.b12x. No additional upstream source imported.
#pragma once
#include "staging.cuh"
namespace m26b1 {
M26B1_HD inline Status check_scale16_tile(const Pool& pool,const Tile& t){
    const auto status=check_tile(pool,t);
    if(status!=Status::ok || !t.active)return status;
    // K128-aligned Width128 only: four complete scale bytes per N row.
    // Width64/down and K32-offset slices require padding/repacking: reject them.
    return t.width==128 && t.k_start%128==0?Status::ok:Status::geometry;
}
// Requires check_scale16_tile==ok, active=true, packet<32. Each packet contains
// four consecutive N rows' four K32 scale bytes, all within one prepared N256 tile.
M26B1_HD inline uint64_t scale16_source(const Tile& t,uint32_t packet){
    const uint64_t n=uint64_t(t.n_start)+(t.gate?local_i:0)+uint64_t(packet)*4;
    const uint64_t k_tiles=t.down?local_i/128:hidden/128;
    return uint64_t(t.slot)*image_bytes+region_offset(t.down?Region::s2:Region::s13)
        +((n/256)*k_tiles+t.k_start/128)*1024+(n%256)*4;
}
#ifdef __CUDACC__
// Same collective preconditions/extents/alignment as stage_async, restricted to
// the tile geometry above. Caller commits, waits and publishes via CTA barrier.
// Both payload and scales are now in that SAME async group, with no CPU/FP decode.
template<bool AsyncScales>
__device__ __forceinline__ bool stage_mlp(const Pool& pool,const Tile& t,const uint8_t* source,
        uint8_t* shared_payload,uint8_t* shared_scales,uint32_t tid,uint32_t threads){
    if constexpr(!AsyncScales){
        return stage_async(pool,t,source,shared_payload,shared_scales,tid,threads);
    }else{
        if(check_scale16_tile(pool,t)!=Status::ok || !threads)return false;
        if(!t.active)return true;
        for(uint32_t task=tid;task<payload_transfers(t);task+=threads){
            const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(shared_payload+task*16));
            const auto* src=source+payload_source(t,task);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst),"l"(src) : "memory");
        }
        for(uint32_t packet=tid;packet<32;packet+=threads){
            const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(shared_scales+packet*16));
            const auto* src=source+scale16_source(t,packet);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst),"l"(src) : "memory");
        }
        return true;
    }
}
#endif
} // namespace m26b1
