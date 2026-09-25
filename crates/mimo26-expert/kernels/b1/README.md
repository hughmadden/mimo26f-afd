# B1 prep and quantizer handoff — 24 September 2026 AEST

**Numerical mode: E-W4A8-v1. Codec: e4m3fn-k32-v1. CPU/compile qualification only.**

Scope: the new prepare/staging/group-plan/route-reduce/quantizer files listed
below, their supporting headers/licenses, and `tests/b1_prep.rs` + `tests/b1/`.
Existing build dispatch, Cargo dependencies, wire implementation and kernel-lead
compute files are unchanged. Integration is pull-style from `b1/prep`.

The worktree was refreshed to `e0b7bdc` after the direct P-LATTICE instruction.
R5 (`dd8de19`) and `docs/design/lattice-v1.md` govern numerical boundaries; R8
supersedes the earlier wire bound. `grouped.cu`, `mxfp4_ptx.cuh` and
`LICENSE.b1-compute` arrived from main and remain **kernel-lead-owned**.
This change does not qualify their numerical compute or add E-W4A8-i16.

## Units and proof

| Unit | Contract | CPU evidence |
|---|---|---|
| `prepare.cu`, `prepared.cuh` | Canonical v2 -> distinct B1P1 N 256/K 128 representation. W13 is **up then gate**. Byte-preserving, I-local 512, exactly 3,342,336 B, 64-bit offsets. | 36 real Rust rank images, 54 pinned full-source hashes, independent full-source rank slicing and forward-coordinate inverse/bijection. v1 JSON refusal, corruption, swap, shape, pointer, overlap and poison negatives. |
| `staging.cuh` | FC1 N 64/128/192, K 128; FC2 N 128, K 64/128/192. Physical slots, 16-B copies, zero-only packed scale tails. | 7,371 canonical-rectangle comparisons; invalid tails/alignment/extent/tag/rank/aliases, inactive poison and >4 GiB integer offsets. |
| `group_plan.cu`, `group_plan.cuh` | Stable expert-major groups of at most 16 rows; actual IDs, inverse/original maps, unchanged weights; classes 256/2048/4096. | 72 replays against independent `stable_sort`, sparse/full IDs, M 64 splitting, shrinking/empty replay, metadata and no-write-on-rejection tests. GPU planner is a **serial correctness scaffold**, not tuned. |
| `route_reduce.cu`, `route_reduce.cuh` | Unweighted, complete FP32 routes in original order. Weight once after FC2; separate RN multiply/add in slots 0–7; FP32 rank sum -> BF16 RNE. | 49,152 FP32 outputs, 48,911 double-weight mismatches; order/FMA/subnormal/RNE and rejection traps. Actual wire types are used unchanged. No atomics: device faults use uniquely owned invalid markers and a subsequent deterministic sweep. |
| `quant_v1.cu`, `quant_v1.cuh` | Independently written from lattice-v1 §4. Integer-bit scale selection and exact-grid RNE; floor FP32(1e-4), signed zero, nonfinite/reconstruction faults. Four full warps/CTA, one K32 block/warp; no atomics or BF16 pre-round. | First reference comparison: 13,939 blocks / 446,048 FP32 inputs; 13,219 valid blocks byte-exact; 686 input faults and 34 reconstruction faults. All 143 scale bytes and 254 finite payload codes covered. |

Quantizer corpus covers code centers and tie neighbours over every scale,
amax transitions, floor neighbours, signed zero, FP32/E4M3 subnormals, max
finite/overflow, nonfinite values at every lane, and fixed-seed bit/exponent
samples. Harness-only wrong variants trigger these mismatch counts:
truncation **98,079**; nearest scale **76,421**; no floor **66,072**;
BF16 pre-round **39,322**; E4M3 subnormal flush **11,654**; K16 **79,437**.
That flush mutant is **not hardware FP32 FTZ/DAZ**: the scale floor already maps
FP32 input subnormals to signed zero. The route test checks an actual CPU FP32
subnormal product separately; no device FTZ execution proof is claimed.
Whole-lattice traps such as dropping intermediate rounding remain separate
compute/oracle gates; these codec counts do not claim those gates.

**Independence:** `quant-v1-independence.md` records the source freeze at
23:47:21 AEST, before reading/importing the builder's reference. The first
reference comparison required **no quantizer source correction**. The test uses
`oracle/lattice/quant_v1.py` unchanged (SHA256
`414d72abeab24fdf3cf796e1f610b3d6547a51ffe6ebc76334b545f4def91333`).
For valid blocks it compares payload and scale bytes. For faults it checks the
reference's **encode then decode** contract: the reference encoder alone emits
bytes for FLT_MAX, but its decoder faults on the out-of-FP32 reconstruction;
this CUDA/host API refuses to publish that block's bytes and reports fault 2.

CUDA 12.8.93 compiles our four implementation TUs and the staging instantiation
for **sm_89, compile only**. No CUDA initialization, kernel execution, GPU timing,
Spark access, sanitizer, graph execution or sm_121 qualification was performed
by this prep owner. The CPU comparison executes shared scalar codec logic,
**not the CUDA warp collectives**. GPU correctness/performance still needs its
own target-SM receipt. Python is harness/oracle only, never serving code.

## Integration obligations

- Loader authenticates actual manifest/version/shape/rank/SHA before upload.
  `CanonicalInfo` and `PreparedInfo` describe geometry, not content authenticity.
  Canonical and prepared allocations remain distinct; publish only after stream
  completion and successful status. Prepared bytes are never row-major v2.
- Caller owns immutable extents/metadata and stream lifetimes, and retains the
  engine's device-owner/AOT gates. These helpers are not serving entry points.
- Staging predicates and metadata are block-uniform. Provide separate aligned
  shared regions of the declared sizes, then commit/wait and block sync.
  Inactive groups return before metadata-dependent addressing. A physical slot
  is not the ordinal of an active expert.
- Complete FC2 plane reduction and inverse scatter before route reduction.
  Neither weights nor BF16 conversion may have been applied to raw route inputs.
  On a numerical fault all reduction outputs are invalid, possibly partially
  written. Check the host enqueue result and final device fault after completion.
- Quantizer input is consecutive K32 blocks. Preserve logical row boundaries
  (H 4096 or I-local 512). Output is row-major activation payload/scales, **not**
  the lane-major prepared weight format. Allocate suitably aligned payload
  storage before exposing it as the compute core's `ActivationView` words.
  Faults are per block: 0 = valid, 1 = nonfinite input, 2 = reconstruction overflow.
  Check every fault before publishing a batch. Bad blocks leave payload/scale
  untouched; structural rejection writes nothing. Empty batches are valid.
- `CoordinatorSum` adds in caller order. Feed buffered rank rows in **0,1,2,3**
  order; the test does so explicitly. Do not infer reordering from the wire type.
  The existing kernel-lead rank-order tests/adapter remain separate ownership.

## Wire bound: retained R3 failure, explicit R8 correction

Original R3 bound: `2.4e-6 + sum(abs(y_r))*2^-9`.
The synthetic position 8,195 has FP32 rank values
`[-0.4443359375, -0.3076171875, -0.1708984375, -0.0341796875]` and decoded BF16
values `[-0.4453125, -0.30859375, -0.1708984375, -0.0341796875]`.
Error **0.001953125** exceeds the old bound **0.00187160166015625**.
The same partials are realizable using normalized top-8 weights of 1/8.
The strict initial failure is retained in
`target/mimo26f-builds/b1-route-596758-1790159808847792038/`.
The fixed corpus has **241 old-bound failures / 12,288 positions**.

After the refresh, **ADVISOR-I4 §9 R8** is authoritative:

`2.4e-6 + sum(abs(y_r))*2^-8 + 3*2^-24*sum(abs(decoded_y_r))`.

The test now reports old R3 failure and corrected R8 separately; it does not
silently replace the failed receipt. `b1_wire_bound_counterexample` preserves
the old negative. `wire-bound.json` names both versions. CPU seam success is
not a model-quality, GPU, or full B1 candidate promotion.

## Reproduce (dev-host CPU / compilation only)

From this worktree, load the existing ignored machine-local config with `ROOT`
set correctly; do not change existing project files or invent toolchain pins.
The outer lock serializes against main's build owner; dev.sh owns the local lock.

```bash
ROOT="$PWD"
source ./configs/build.env
MIMO26_B1_REAL_DIR=./target/mimo26f-builds/expert-tp4.7VfBT7/oracle \
MIMO26_B1_COMPILE_CUDA=1 \
flock ./target/mimo26f-builds/.cargo.lock \
  scripts/dev.sh test expert-unit --test b1_prep -- \
  --include-ignored --nocapture --test-threads=1
```

To run only the quantizer, put `b1_quantizer_v1` before `--` and use `--ignored`.
The real prepare cell needs 36 Rust images, nine manifests and 54 full-source
payload/scale files; missing data fails, not skips. Each cell retains unique
build slots, command logs, Sydney timestamp, commit/dirty state, source/artifact
hashes and result JSON. Five heavy cells are ignored by default; the lightweight
old-wire-bound counterexample remains in normal discovery. Omit
`MIMO26_B1_COMPILE_CUDA=1` for pure CPU checks.

Retained historical receipts (worktree-relative build root):
- Prep before R5 label update: `b1-final.4ZFUwz.log`, 5 tests passed at 20:51 AEST.
- Scoped expert/repack regression: `b1-regression.pVE8sx.log`, 143 passed,
  13 ignored, before the later main refresh. Not a full-workspace gate.
- First quantizer/reference: `b1-quant-first.tEqbIl.log`, 1 passed at 23:56 AEST;
  `b1-quant-1837491-1790171804349282963/quant-reference.json` has exact counts.
- Earlier FC1 alignment-negative failure: `b1-staging-662529-1790160454640503956/`.
  It found `bounds` instead of `geometry`, not an accepted invalid access; fixed.
- The first invocation lacked Cargo on PATH and stopped before compilation.
  All failed cells were retained, not overwritten.

## Final verification — 24 September, 00:02 AEST

At base `e0b7bdc` plus these scoped files:
- Full explicit B1 suite: **6 passed, 0 failed**; all five CUDA TUs compiled.
- Default scoped expert/repack regression: **145 passed (68 + 77), 0 failed,
  14 ignored**. This is not the full-workspace merge gate.
- Old R3 wire negative remains detected; corrected R8 CPU seam **PASS**.
- Quantizer source hashes remain identical to the pre-reference freeze.

Aggregate log: `target/mimo26f-builds/b1-release-check.KrOOZ4.log`.
Per-cell source/artifact hashes and status are under that build root:
`b1-plan-1867127-1790172151705375756`,
`b1-prepare-1867127-1790172155827550068`,
`b1-quant-1867127-1790172162448908001`,
`b1-route-1867127-1790172166766051221`,
`b1-staging-1867127-1790172170839207556`.
Each has `result.json`; quantizer has `quant-reference.json`, route has
`wire-bound.json`. Read-only peer reviews found no production correctness defect
under documented caller contracts. Missing regression locks were added; the
E4M3-flush/FP32-FTZ distinction is explicit. Neither review approved GPU execution.

## Approved provenance

Four pre-existing B1 REUSE rows; supporting headers belong to those same units:
- ds41rt prepare (`v41_expert_pack.cu`) at `e2a6f2ca5af56b8b567fb5086f82597427f28477`,
  SHA256 `c72c6cee143fba9522b7485573eeb2889d49eef848a3b7b809bb7434edfbe23e`.
- b12x exact staging (`w4a8_staging.py`) at approved gitlink
  `3882b935ede761d6c73a5d6fd68e690f1e3f5380`, SHA256
  `ab95db03abdc06152ab47878cfa4e2e7b7bb19f450516d76ad472f04f1e147f0`.
- b12x stable grouping/inverse-map portion (`v41_route_plan.py`), same gitlink,
  SHA256 `f5ee61e9e0e3c8cbbf0c6aa50af74be57d33d32b88b09f69debe93c996175da4`.
- ds41rt compact reduction (`v41_route_reduce.cu`), same ds41rt commit,
  SHA256 `49fcb3270485505bfe57e91bc55e3eba005f68efb1c0b8b5d014f487095b3466`.

The inspected b12x checkout was `4d0e409475bee86ea95ff4828ce3461812d855ba`;
these files were verified identical to the approved gitlink. Original MIT and
Apache-2.0 licenses are retained as `LICENSE.ds41rt` and `LICENSE.b12x`.
No old rank-per-route reducer, shared expert or final coordinator cast was copied.
Quantizer source is new, spec-derived code, not upstream copy-in.
