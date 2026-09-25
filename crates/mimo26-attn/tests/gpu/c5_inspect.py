#!/usr/bin/env python3
"""Read-only package/source inventory. No torch/FlashInfer imports, CUDA or JIT."""
import importlib.metadata as md
import json
import re
from pathlib import Path

print('C5_SCOPE exact 64/4 heads, QK192/V128, BF16 Q, E4M3 or BF16 KV; reference only')
for name in ('torch', 'flashinfer-python', 'flashinfer', 'flashinfer-cubin', 'flashinfer-jit-cache', 'vllm'):
    try:
        d = md.distribution(name)
    except md.PackageNotFoundError:
        print('PACKAGE_MISSING', name)
        continue
    print('PACKAGE', name, d.version, str(d.locate_file('')))
    files = list(d.files or [])
    native = [str(f) for f in files if str(f).endswith(('.so', '.cubin', '.fatbin'))]
    print('NATIVE_FILES', name, len(native), json.dumps(native[:30]))
    if name == 'flashinfer-python':
        aot = Path(d.locate_file('flashinfer/data/aot'))
        print('FLASHINFER_AOT', str(aot), 'exists', aot.exists(),
              'libraries', json.dumps([str(x) for x in aot.rglob('*.so')][:40]))
    for rel in files:
        p = str(rel)
        selected = (p in ('flashinfer/decode.py', 'flashinfer/prefill.py', 'flashinfer/jit/env.py', 'flashinfer/jit/core.py',
                          'vllm/attention/ops/paged_attn.py', 'vllm/v1/attention/ops/paged_attn.py',
                          'vllm/vllm_flash_attn/flash_attn_interface.py', 'vllm/v1/attention/backends/flash_attn.py') or p.endswith('flashinfer/jit/attention/variants.py'))
        if not selected:
            continue
        path = Path(d.locate_file(rel))
        lines = path.read_text().splitlines()
        print('SOURCE', path)
        hits = set()
        for i, line in enumerate(lines):
            if re.search(r'def (plan|run|get_batch_decode|gen_batch_decode)|head_dim|head_size|use_tensor_cores|jit|prebuilt|prebuilt_ops|backend|SUPPORTED_HEAD|FLASHINFER_AOT_DIR|def flash_attn|def forward_decode|empty_like', line, re.I):
                hits.update(range(max(0, i-1), min(len(lines), i+3)))
        if p == 'vllm/vllm_flash_attn/flash_attn_interface.py':
            hits = set(range(min(len(lines), 430)))
        for i in sorted(hits)[:480]:
            print(f'{i+1}: {lines[i]}')
for root in ('/root/.cache/flashinfer', '/opt/flashinfer', '/workspace/flashinfer'):
    p = Path(root)
    if p.exists():
        matches = [str(x) for x in p.rglob('*192*')][:40]
        print('BAKED_CACHE_192_PATHS', root, json.dumps(matches))
print('RESULT: PASS inspection only; no decode availability or timing claim')
