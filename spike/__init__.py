"""spike/__init__.py — I1 spike CPU twin (throwaway-by-default).

Modules: quant (T1/T2 codec), loader (ckpt_tp=4 split + name audit), attn (T3/T5),
kv (SWA eviction), model (T9 start_pos).  Env ``MIMO26_SPIKE_NAIVE=1`` selects
every wrong implementation at once so the negative suite fails.
"""
