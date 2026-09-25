"""mimo26 — CPU reference + stub package for MiMo-V2.6-Flash on the DS41RT AFD engine.

See ../../ARCHITECTURE.md for the design contract. numpy only; no GPU, no weights required.
"""

from .config import MiMoConfig, GA, SWA, DENSE_FFN, MOE_FFN  # noqa: F401

__all__ = ["MiMoConfig", "GA", "SWA", "DENSE_FFN", "MOE_FFN"]
