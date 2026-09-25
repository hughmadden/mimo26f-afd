//! ABI layout pins for the attention FFI (`mimo26_attn::ffi`) — the Rust mirror
//! of `kernels/include/mimo26_attn_kernels.h`. CPU-only (no CUDA header): the
//! declarations type-check on the merge gate with no nvcc.

use std::mem::{offset_of, size_of};

use mimo26_attn::ffi::M26Geom;

#[test]
fn m26_geom_matches_the_c_abi() {
    assert_eq!(size_of::<M26Geom>(), 32, "m26_geom must be 32 bytes");
    assert_eq!(offset_of!(M26Geom, n_q), 0);
    assert_eq!(offset_of!(M26Geom, n_kv), 4);
    assert_eq!(offset_of!(M26Geom, d_qk), 8);
    assert_eq!(offset_of!(M26Geom, d_v), 12);
    assert_eq!(offset_of!(M26Geom, window), 16);
    assert_eq!(offset_of!(M26Geom, value_scale), 24);
}
