# B1 primitive qualification

Owned by KERNEL-LEAD; separate from the lead's `tests/b1/` and preparation units.

Approved REUSE rows 116–117 extract only the E2M1 container transform and
block-scaled MMA PTX from pinned b12x `intrinsics.py`. The full-file SHA256 and
license are retained in `kernels/b1/mxfp4_ptx.cuh` and `LICENSE.b1-compute`.

- `scripts/dev.sh test gemm b1-primitive-host`: nvcc 12.8 compiles sm_120a;
  only the host fixture runs. **No dev-host GPU initialization or qualification.**
- `HOST=spark1 scripts/dev.sh test gemm spark-b1-primitive-dry`: no effects.
- `HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-b1-primitive`:
  nvcc 13 builds sm_121a; live architecture 121 and 48 physical SMs are required.
  The architecture-specific ISA is explicit, not a rewritten AOT identity.
  Use `--gpu-architecture=compute_121a --gpu-code=sm_121a`, not the `-arch`
  shorthand: NVCC 13 also adds base `compute_121` PTX for the shorthand, which
  rejects this instruction. The first compile failure is retained; no test
  kernel ran in that attempt. See [NVCC 13 code-generation options](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch).

The synthetic fixture exercises an asymmetric 16 × 8 × 128 product using four
K32 blocks and all four A/B scale-byte selectors. A lane mapping is independently
checked against a scalar FP64 dot product. Actual mutations swap nibbles, select
the wrong scale bytes, or use another column's scale lanes.

Container tests preserve the nibble in byte bits 5:2, including negative zero.
All 4,096 code/scale pairs are tested in eight byte positions; recovered nibbles
pass through the existing conforming MiMo decoder, compared bitwise with an
independent host codebook/FP64 saturation reference. The real K1 cell compares
55,296 fixture bits and 54 source hashes, with a real nibble-swap negative.

**Limits:** ordinary-scale synthetic MMA is not exceptional-scale MMA proof.
The final compute dispatcher still needs MiMo's special-scale/saturation handling.
This cell does not implement the prepared layout, quantizer, grouped FFN,
route reducer or bandwidth gate. It neither promotes B1 nor revisits B2's STOP.
All device buffers have redzones and preserve an 8 GiB device reserve; the
remote runner separately guards CUDA ownership and system MemAvailable.

## Independent FFN lattice oracle

`crates/mimo26-repack/tests/lattice_oracle.py` consumes the builder's
`oracle/lattice/quant_v1.py` and the independent spike MXFP4 decoder. It imports
no CUDA/Rust implementation. Run through:

- `scripts/dev.sh test gemm lattice-oracle-selftest`
- `scripts/dev.sh test gemm lattice-oracle-real`

The first runs the bounded checkpoint-reader regressions, builder codec CPU
suite and nine FFN/wire mutation detectors. Together they cover all nine required
lattice-v1 naive switches. Torch is deselected; no GPU is initialized.

The real cell generates scratch-only references for layer 1 experts
`[0, 7, 255, 1, 2, 3, 4, 5]`, four ranks and M = 1, 2, 4, 8. FC1 dots are FP64
then cast to FP32. Stable SiLU and the up product use explicit FP32 operations;
intermediate encoding uses the builder codec, unweighted, in 16 K32 blocks/rank.
FC2 dots are FP64. Separately, FC2 outputs round to FP32 before weight-once and
slot-ordered pre-sum; each rank rounds to BF16 before rank-ordered FP32 addition.

`b1-source.json` names every artifact's shape/dtype/SHA256 and hashes the oracle,
codec, decoder and reader sources. Input payload/scales and gate/up/product,
intermediate payload/scales/decoded values, raw FC2 partials and numerical wire
results are exported. Stage arrays use `[rank, token, route, channel]`; weights
use `[token, route]`. `b1-full.f64` is the ideal weighted full FC2 result before
FP32 route arithmetic; `b1-wire.f32` applies the declared FP32/BF16 boundaries.
This Python wire calculation is **not actual codec or transport execution**.

The reference does not emulate an MMA accumulation tree. Compare FC1/product
compute at the frozen componentwise bound and test codec bytes on identical
FP32 inputs. Characterize threshold crossings separately; never widen tolerance
or relabel a different lattice to hide them. A CPU oracle receipt is not a B1
GPU correctness, model-quality or performance qualification.

## Compute-only slice routines

`kernels/b1/grouped.cu` ports REUSE row 118's inner FC1/FC2 schedule. The caller
supplies validated activation rows and staged weight/scale tiles. It contains
no preparer, staging copies, group planner, quantizer or route reducer, and does
not yet export a production launch. Widths 64 and 128 use the same lattice.

The extended `spark-b1-primitive` cell tests full K4096 FC1 and separate
N4096 × K64/K128 FC2 projections for M = 1, 2, 4, 8, 16, 64 (M64 is four M16
math groups, **not optimized prefill qualification**). Host fixtures construct
tiles directly; no unpublished lead code is compiled. Input row permutations,
padding zeros, poisoned outputs and redzones are checked against scalar FP64.
The GPU negatives swap gate/up, apply native clamps or BF16 FC1 rounding.

Weight scales 0/1 and 253–255 take a warp-uniform scalar FMA arm using the
conforming MiMo decoder before multiplication. This preserves subnormal weights
and finite saturation; scale 255 is not passed to native UE8M0 as NaN. Five
separate K32 probes test both signs, signed-zero operands and scale-byte selector
2; a bypass-fallback negative exercises the actual wrong native MMA path.
Ordinary-scale MMA remains the normal arm. Activation decoding is not an encoder;
invalid FP8 codes/scales propagate NaN for the caller's numerical-fault check.

The projections are intentionally independent: their test inputs do not stand
in for the missing fused activation/quantizer connection. Full B1 integration and
real-model GPU lattice comparison still require the lead's published units.

### FC2 precision extension point (builder request, 22:35 AEST)

`fc2_slice<Width, OperandPolicy>` selects operand handling at compile time;
`Fc2Fp8` is the only implemented/default policy. Its `View`, `load` and K32
`step` isolate the activation representation and MMA from the shared FP4 tile
schedule. FC1 still emits FP32 intermediates, leaving the caller to pair the
selected FC2 policy with that lattice's declared intermediate conversion.

A future BF16 policy may use exact dequantized ordinary FP4 weights and a
conforming exceptional-scale fallback over the same tiles. **No i16 policy,
kernel, encoder or dispatch path is implemented now.** This is not a runtime
switch or per-M lattice mixing; E-W4A8-v1 remains the current build target.

### Compute-only sanitizer cell

`HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-b1-sanitize`
rebuilds the optimized sm_121a driver with line information and runs `--all-math`
under memcheck (full leak check), initcheck, synccheck and racecheck. Every tool
must exit zero, emit one clean summary and pass all 12 slice / 12 complete-rank /
5 scale cases. Nonzero exits, missing cases, dirty summaries and warnings fail closed.
`b1-sanitizer-selftest` exercises four parser positives, 52 failure controls and
an unknown-tool refusal without invoking CUDA or a sanitizer. The dry cell is
`spark-b1-sanitize-dry`; all remote runs retain the ownership/memory guards.

This checks only the current independent projections and exceptional-scale arm.
Racecheck is a shared-memory hazard checker, **not proof of global-memory race
freedom**. None of these results can close the later fused staging/quantizer/
reducer sanitizer gate. The parser negatives are CPU fixtures, not injected GPU
memory faults. Tool availability is required; the cell installs nothing.

### Complete rank-local projection boundary

The primitive cell additionally runs `--rank 0`: FC1 covers all 512 local
channels, with gate/up/product placed at row stride 512 in M16-padded group
scratch. This scratch must not use an unpadded `Group::route_base` as its physical
row index. FC2 keeps one FP32 accumulator across all K512, in four K128 or eight
K64 updates, and writes complete unweighted N4096 outputs. No intermediate slice
planes, numerical atomics, route weights or slice-reduction unit are introduced.
The compute routine's activation view is offset in both payload and scale
coordinates while retaining its row strides.

Independent scalar FP64 references cover both widths and M = 1, 2, 4, 8, 16, 64.
Actual negatives permute output channel slices, reset FC2 accumulators between
updates, or omit activation K offsets. They do not introduce races. These remain
independent projections on synthetic encoded inputs, not the connected
activation/quantizer/FC2 chain. The sanitizer cell runs both suites through
`--all-math`; its result must name complete-rank coverage explicitly. Neither
suite is optimized prefill qualification.

### Stage artifact comparison (CPU harness)

`MIMO26_B1_REFERENCE=... MIMO26_B1_CANDIDATE=... scripts/dev.sh test gemm lattice-compare`
compares candidate artifacts against the independent real-reference bundle. Both
use `b1-source.json` and the reference's flat artifact names, shapes, dtypes and
SHA256 metadata. Candidate FC2 is `b1-partial.f32` rather than the reference's
`b1-partial.f64`; the candidate need not provide `b1-full.f64`, `b1-x.f32` or
`b1-mid.f32`. Source projection identities/hashes and input wire bytes/weights
must match. M = 1/2/4/8 can consume prefixes of a larger reference. Axes are rank
0..3, token, original route slot 0..7, channel; each intermediate row has 512
values and 16 K32 scales. Hashes do not establish execution provenance.

Gate/up/product/FC2/rank comparisons retain `atol = rtol = 1e-5`. The CUDA
quantizer bytes must equal the builder codec applied to the **reported FP32 h**,
not to ideal-reference h. A separate count reports ideal/candidate intermediate
byte differences; it never waives any downstream bound. Ordered route weighting,
rank BF16 RNE and rank-order accumulation are checked bit-exactly using their
reported inputs. The pre-return sum must match the ideal full FFN within
`2.4e-6`; the return uses the unchanged R8 bound. Exit 3 is numerical failure,
exit 2 is malformed/incompatible artifacts. This is an offline comparison,
**not proof that a candidate ran on a GPU or used the claimed checkpoint**.

The existing `lattice-oracle-selftest` cell includes comparator controls: data
mutations, wrong reduction order/rounding, one-ULP intermediate bin crossing
(with no downstream tolerance waiver), nonfinite input, metadata/hash/size and
symlink refusals. They are CPU synthetic controls, not B1 GPU qualification.

### Connected v1 Spark cell

`HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-b1-ffn`
stages the published lead units without modification. The new compute caller
uses shared-memory weight staging with producer/consumer barriers, full K512
FC2 accumulation, group-padded FP32 intermediates, the independent CUDA v1
quantizer, original-route scatter, and the lead's once-weighted BF16 reducer.
Every plan/compute/quantizer/reducer fault and redzone is checked before output
acceptance. It tests M = 1 through 8, four ranks, eight real layer 1 experts and
a deliberate double-weight mutation. The dev-host comparator consumes actual GPU
stage dumps; binary data stays in scratch. Inputs are already encoded v1 wire
bytes and are not re-encoded on Spark. Coordinator summation is CPU FP32 rank
order; this cell does not execute wire transport, AOT capacity admission or a
bandwidth measurement. The original compute-only sanitizer remains distinct.
