"""Small shared helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_json_lenient(path_or_text) -> dict:
    """JSON with trailing commas tolerated.

    Xiaomi's shipped `dflash/config.json` ends `... "use_cache": true,\\n  \\n}` — strict
    `json.loads` rejects it (verified 2026-09-22), so tolerant loaders (DFlashConfig.from_file) route through this helper.
    Accepts a Path, an existing filename, or raw text (anything containing a newline or a
    NUL-less long string is treated as text).
    """
    if isinstance(path_or_text, Path):
        text = path_or_text.read_text()
    elif isinstance(path_or_text, str) and "\n" not in path_or_text and Path(path_or_text).is_file():
        text = Path(path_or_text).read_text()
    else:
        text = str(path_or_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)
