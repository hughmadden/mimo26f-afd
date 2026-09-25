#pragma once
// First-party R13 diagnostic CTA mapping. Adjacent four-row tiles share an
// expert's L2 working set; they do NOT share a staged CTA weight read.
#ifdef __CUDACC__
#define M26X_TILE_HD __host__ __device__
#else
#define M26X_TILE_HD
#endif
template<bool Paired>
M26X_TILE_HD constexpr int m26x_tile_group(int block_y) {
    return Paired ? block_y / 2 : block_y;
}
template<bool Paired, int M>
M26X_TILE_HD constexpr int m26x_tile_offset(int block_y, int block_z) {
    return (Paired ? block_y % 2 : block_z) * M;
}
#undef M26X_TILE_HD
