# oracle/

CPU numerics contract. At I1 this holds **an import** of the first-attempt
`mimo26` package + `tests/golden/` (or a pin/reference to it). We consume it;
we do not fork it.

- `mimo26/` — numpy reference (loader codecs, attention, KV, draft, sample)
- `tests/` — L0/L1 suite + golden corpus
- External oracle receipts live in `../runs/`, not here.

Until I1, the source of truth is:
`port-workspace/code/`

- `lattice/` — **project-owned** (not part of the imported twin): the lattice-v1 reference codecs
  (`quant_v1.py`: quantizer `e4m3fn-k32-v1` + BF16 RNE, numpy codec of record and a bit-equal torch
  fake-quant twin). Contract: `docs/design/lattice-v1.md`; tests: `tests/test_lattice_quant_v1.py`.
