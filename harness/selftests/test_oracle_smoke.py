"""Oracle import seam — the twin (oracle/mimo26, P-201) must import clean and
expose the load-path entry points the engine mirrors (map §3/§4).  R6
foundation: before the harness ever touches a model it proves the reference
stack is present and internally consistent.  Two-run: env-invariant.
"""
from __future__ import annotations

import importlib

import numpy as np

MODULES = (
    "mimo26", "mimo26.config", "mimo26.kv", "mimo26.loader", "mimo26.model",
    "mimo26.util", "mimo26.checkpoint", "mimo26.quant", "mimo26.quant.fp8_block",
    "mimo26.quant.mxfp4", "mimo26.nn", "mimo26.nn.layers",
)


def test_twin_package_imports():
    """P-201's 12 imported modules import clean and the map §3/§4 entry points
    exist: loader.py:28 canonical_name / :59 qkv_segments / :67
    reconstruct_layer_qkv, quant/fp8_block.py:120 split / :170 bug oracle,
    quant/mxfp4.py:67 pack / :88 unpack."""
    for name in MODULES:
        importlib.import_module(name)
    from mimo26.config import MiMoConfig
    from mimo26.loader import canonical_name, is_backbone_weight, qkv_segments, \
        reconstruct_layer_qkv
    from mimo26.quant import fp8_block as fb
    from mimo26.quant import mxfp4 as mx
    for fn in (MiMoConfig.tiny, canonical_name, is_backbone_weight, qkv_segments,
               reconstruct_layer_qkv, fb.split_shard_major_fused,
               fb.dequantize_naive_fused, mx.pack, mx.unpack):
        assert callable(fn)


def test_tiny_config_geometry_sanity():
    """config.py:233 tiny() stays internally consistent and keeps the
    QK-192/V-128 ASYMMETRY class (config.py:127-139): q_rows != v_rows on both
    layer kinds."""
    from mimo26.config import GA, SWA, MiMoConfig
    cfg = MiMoConfig.tiny()
    assert cfg.num_hidden_layers == len(cfg.hybrid_layer_pattern) == 4
    assert cfg.attn_dims(GA) == (256, 64, 32, 128)   # q=8*32, k=2*32, v=2*16, o_in=8*16
    assert cfg.attn_dims(SWA) == (256, 128, 64, 128)  # k=4*32, v=4*16
    for dims in (cfg.attn_dims(GA), cfg.attn_dims(SWA)):
        assert dims[0] != dims[2]


def test_tiny_load_and_codec_micro_smoke():
    """Micro smoke on the real entry points: tiny fp8 block roundtrip
    (quant/fp8_block.py:80/:64) + the loader.py:67-94 happy path at 2 ranks
    with a padded per-shard grid."""
    from mimo26.config import MiMoConfig
    from mimo26.loader import reconstruct_layer_qkv
    from mimo26.quant import fp8_block as fb
    w = np.linspace(-1, 1, 64).reshape(8, 8).astype(np.float32)
    codes, scale = fb.quantize_block(w, block=(4, 4))
    assert fb.dequantize_block(codes, scale, block=(4, 4)).shape == (8, 8)
    cfg = MiMoConfig.tiny()
    per = (256 + 64 + 32) // 2  # tiny GA fused segments per shard (n_ranks=2)
    shard_w = np.zeros((per, 8), np.uint8)
    shard_s = np.ones((per // 4 + 2, 2), np.float32)  # padded grid, block (4, 4)
    parts = reconstruct_layer_qkv(cfg, 0, [shard_w, shard_w], [shard_s, shard_s],
                                  block=(4, 4), n_ranks=2)
    assert set(parts) == {"q", "k", "v"}
    assert parts["q"].weight.shape == (256, 8) and parts["k"].weight.shape == (64, 8)
    assert parts["v"].weight.shape == (32, 8)
    assert parts["v"].scale_per_row.shape == (32, 2)
