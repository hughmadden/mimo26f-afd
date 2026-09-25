"""spike/shard_index.py — P-102: read-only shard index + fail-loud name audit.

Entry points mirrored (map §2): ``mimo26/loader.py`` — ``canonical_name`` (:28,
``model.mtp.`` prefix tested BEFORE the generic ``model.`` strip — order
matters), ``is_backbone_weight`` (:43), expert regex (:23).

Live index (the coordinator, 22 Sep 2026 23:17 AEST): 73 081 tensors / 65 shards / 48 fused-QKV /
 48 mtp / 462 audio_encoder.*+visual.* (non-backbone, expected and dropped
from the backbone dict — is_backbone_weight :43 class).

Fail-loud audit: anything that classifies as UNCLASSIFIED raises; losing the
``mtp.`` root in canonicalisation raises.  Nothing writes to the weights tree.
"""
from __future__ import annotations

import json
import re
from collections import Counter

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale)?$")
NON_BACKBONE_PREFIXES = (
    "audio_encoder.", "model.audio_encoder.", "visual.", "model.visual.",
    "speech_embeddings.", "model.speech_embeddings.")


def canonical_name(raw: str) -> str:
    """mimo26/loader.py:28 — MTP prefix first (order matters), then generic strip."""
    if raw.startswith("model.mtp."):
        return "mtp." + raw[len("model.mtp."):]
    if raw.startswith("model."):
        return raw[len("model."):]
    return raw


def is_backbone_weight(raw: str) -> bool:
    """mimo26/loader.py:43 — backbone only; drops mtp/dflash/audio/visual."""
    c = canonical_name(raw)
    return bool(re.match(r"^layers\.\d+\.", c)) or c in (
        "embed_tokens.weight", "norm.weight", "lm_head.weight")


def classify(raw: str) -> str:
    if EXPERT_RE.match(raw):
        return "expert"
    if raw.startswith("model.mtp."):
        return "mtp"
    if "dflash" in raw or "draft" in raw:
        return "dflash"
    if raw.startswith(NON_BACKBONE_PREFIXES):
        return "audio/visual/speech"  # expected non-backbone (dropped by :43)
    return "backbone" if is_backbone_weight(raw) else "UNCLASSIFIED"


def audit(index: dict) -> dict:
    """Fail-loud audit of a safetensors index {weight_map: {tensor: shard}}."""
    wm = index["weight_map"]
    kinds = Counter()
    bad: list[str] = []
    for raw in wm:
        k = classify(raw)
        kinds[k] += 1
        if k == "UNCLASSIFIED":
            bad.append(raw)
        if k == "mtp" and not canonical_name(raw).startswith("mtp."):
            bad.append(f"{raw}: mtp prefix lost by canonicalisation")
    qkv = sum(1 for n in wm
              if re.match(r"^model\.layers\.\d+\.self_attn\.qkv_proj\.weight$", n))
    if qkv != 48:
        bad.append(f"fused-QKV count {qkv} != 48")
    if bad:
        raise ValueError(f"shard_index audit: {len(bad)} problems: {bad[:8]}")
    return {"tensors": len(wm), "shards": len(set(wm.values())),
            "qkv": qkv, "kinds": dict(kinds)}


def load_index(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
