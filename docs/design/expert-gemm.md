# MXFP4 expert GEMM: layout v2 and two execution regimes

2026-09-23 AEST. Owner: KERNEL-LEAD. **Design / implementation in progress.**
CPU compilation and trap repairs landed in `c77ad78`. No GPU bandwidth or
Spark promotion is claimed here. Hardware receipts, not this estimate, decide.

## 1. Mathematical and storage contract

H=4096, I=2048, 256 experts per MoE layer, 47 MoE layers. Weights are checkpoint
E2M1 low-even/high-odd nibbles with one E8M0 scale per 32 K elements (0.53125 B
per parameter). No requantization or NVFP4 conversion is allowed in repacking.

For rank r, J=[512r,512(r+1)):

- Gate/up retain all H inputs and **contiguous output rows J**: [512,4096].
- Down retains **all H output rows and contiguous input columns J**: [4096,512].
- h_r = SiLU(x G[J,:]^T) * (x U[J,:]^T).
- y_r = h_r D[:,J]^T. Sum y_0+y_1+y_2+y_3 in fixed rank order.
- Each partial has 4096 outputs: 8192 bytes if encoded BF16. There is no
  intermediate gather and no zero embedding. Routing weights must be applied
  exactly once by the integration layer, outside this raw expert primitive.

| Region | Offset | Bytes | Row-major local shape |
|---|---:|---:|---|
| gate payload | 0 | 1,048,576 | [512,2048] u8 |
| gate scales | 1,048,576 | 65,536 | [512,128] u8 |
| up payload | 1,114,112 | 1,048,576 | [512,2048] u8 |
| up scales | 2,162,688 | 65,536 | [512,128] u8 |
| down payload | 2,228,224 | 1,048,576 | [4096,256] u8 |
| down scales | 3,276,800 | 65,536 | [4096,16] u8 |

Total **3,342,336 B**, no reserved tail. The old v1 layout used strided OUTPUT
rows for down, [1024,2048], and required a gather. It has the SAME byte sizes!
Manifest version is now **2**. Reject version 1 before reading/uploading a slice;
regenerate slices and SHA manifests, never just relabel a manifest. A bare
byte array or its size is not a version discriminator. `identity::load_slice`
is the checked disk boundary; raw projection accessors assume already-validated
v2 input. Corruption, ambiguous entries and mismatched identities refuse loading.

## 2. Decode: M <= 8 is a streaming GEMV family

At M=1 one rank reads 3.34 MB/expert for 12.58 MFLOP. Weight-only arithmetic
intensity is 3.765 FLOP/B (NOT the WIP's 1.25 or 2). At M=8 it is ~30.1 FLOP/B
before activation traffic. Reuse each packed weight across the M local rows.
Do not expand resident FP4 weights to BF16/f32 globally: that destroys the byte
budget and the 40.2 GB/rank resident-weight model.

### Response to MiMo §2: chosen next implementation is B2, not scalar tuning

[Review f4af25e](../../runs/20260923-i4/reviews/f4af25e-mimo.md) is accepted:
fixing offsets alone cannot make the inherited kernel viable. The earlier
one-warp/row, byte-load sketch is superseded by this vectorized, pipelined map.
B2 is the next **qualification candidate**, not a claim that the performance
problem is solved. No prior-art code is copied; B1 remains the preferred
follow-up if measured B2 misses the target. Copy-in requires builder-owned
REUSE entries first. Do not start the scheduler on an unqualified B2 result.

1. **Row redundancy (§2.1 / F6):** 256 threads, 8 warps, four independent
   8-lane groups per warp, **32 output rows per CTA**. For lane l and warp w:
   `row = blockIdx.x*32 + 4*w + floor(l/8)`, `sub_lane = l%8`.
   Each 8-lane group cooperates on one dot product; ONLY sub_lane 0 stores it.
   Gate/up use 16 row CTAs/expert, down 128. The ownership test enumerates
   `(row,token)` stores and requires exactly one writer, not just equal answers.
   Remove the misleading `ROWS_PER_THREAD` constant and repin Rust/CUDA geometry.
2. **M-fold rereads (§2.2):** specialize M-capacity in {1,2,4,8}. The loop order
   is K-tile -> packed word -> nibble pair -> **token inner loop**. Keep two
   FP32 accumulator chains per token in registers; every decoded weight pair
   feeds all active token accumulators before it is discarded. Reduce chains
   and the eight lanes in a fixed order. Mask ragged token tails, and return
   empty/padded groups BEFORE their expert address or token interval is loaded.
   M>8 uses a separately labelled fallback until the prefill kernel is qualified.
3. **Divergent constant LUT (§2.3):** remove hot-path `__constant__` indexing.
   Compose E2M1 sign/exponent/mantissa with integer operations, fold the E8M0
   exponent into it, and handle zero/subnormal/saturation explicitly (§4).
   Decode each pair once, outside the token loop. Exhaustively compare all
   4096 nibble/scale combinations against the independent scalar reference.
4. **Scalar strided loads and latency (§2.4):** K tile = **256 elements**.
   Every lane loads one aligned `uint4` (16 packed bytes = 32 weights) at
   `row_payload + k0/2 + 16*sub_lane`. The eight lanes collectively load a
   contiguous, aligned 128-byte row segment, rather than eight unrelated rows.
   Each row-group leader loads its eight scale bytes as an aligned 64-bit word;
   shuffle the two 32-bit halves within that group, select one scale per lane.
   v2 row strides/offsets and K256 boundaries preserve these alignments.

**Two-stage pipeline:** prefetch next tile's packed vector and scales into
registers before arithmetic consumes the current tile. Double-buffer activation
K256 tiles in shared memory using `cp.async` (available to both selected CUDA
architectures); at M-capacity 8 this is `2*8*256*4 = 16,384` bytes, not 128 KiB.
Wait for the stage being consumed, synchronize the CTA before consumption and
before buffer reuse, and drain all outstanding copies in the tail. Do not use
`wait_group 1` for a tail with only one pending group: it need not wait for that
copy. No full-K shared staging and no zero-byte "enabled" fallback remain.

Activation staging is vector-aligned and swizzled: for local K index k, store
at `((k%32)/4)*32 + (k/32)*4 + k%4` within its token's 256-float tile. The copy
unit is four adjacent floats; eight row-group lanes accessing a fixed k-within-
32 then hit eight distinct banks, while different row groups broadcast the
same activation addresses. A plain K-major tile would create 8-way bank
conflicts for this lane map. The permutation must get a bijection/bounds test.

At M8 the two accumulator chains cost 16 registers/lane and two packed vectors
cost eight, before scales, addresses and temporaries. Compiler register/spill
reports and SASS must confirm vector loads and load/compute overlap; source
`uint4` or "prefetch" comments alone are not evidence. Start with two stages;
do not increase to four without evidence that latency, not register pressure,
is the remaining bottleneck. Gate/up may share the activation tile, but fusion
must not silently double the quoted register budget. First qualify separate
GEMMs plus SiLU/multiply, including all their time in the FFN receipt.

**Qualifications to the review (not excuses to retain the old code):** the
16-entry LUT has at most 16 distinct lookup addresses, not 32; serialization
is still unacceptable. Identical duplicate stores are still a data race, so I
disagree with calling F6 correctness-neutral. Finally, 4x and Mx are duplicated
work/load factors, not established DRAM-traffic factors: caches require a real
counter/working-set check. None of these changes the CHANGES verdict.

## 3. Prefill: M around 64 needs reuse and matrix instructions

For a 2048-token, top-8 prefill uniformly routed over 256 experts, M=64, with
16,384 routed rows. This is not M=8. Arithmetic intensity rises to roughly
241 FLOP per weight byte: scalar GEMV repetition is a correctness fallback,
not an adequate prefill optimization.

Target prefill path: bounded M/N/K tiles (initial candidates 16x64x32 and
32x64x64), packed global loads -> on-chip E2M1/E8M0 decode -> shared/register
BF16 tiles -> tensor-core MMA with FP32 accumulation. Double buffering and
async copies are optimizations only after the baseline has passed parity.
Scale boundaries are K32 and every v2 K split is aligned to them. Choose tiles
from measured register/shared-memory occupancy, not theoretical FLOP peak.

The input dtype is part of the numerical contract. For BF16 or dequantized
FP8 values exactly representable in BF16, this can use BF16 MMA directly.
A general f32-input API must NOT silently cast x to BF16 and claim f32 parity:
retain the f32 fallback, or explicitly implement/test a multi-part BF16 expansion
before enabling that fast path. Exceptional MXFP4 values outside the finite
BF16 range also require the f32 fallback. Until this path is implemented and
qualified, label prefill performance as the fallback, never as tensor-core speed.

## 4. Decoder correctness and load path

E2M1 LUT by nibble: [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3,
-4, -6]. Preserve negative zero for bitwise unpack. Scale = 2^(min(byte,254)-127);
reserved 255 clamps to 2^127. Saturate the product to finite f32 range. E8M0
byte 0 is **2^-127**, not zero. The 27 real fixture blocks have no 255 scales
and no saturation, so exhaustive synthetic decoder boundary tests remain needed.

Device decoder should use a register/bit-based LUT equivalent, not divergent
constant-memory table loads on every lane and not FP64 arithmetic on GB10.
Fast normal exponent composition may handle the observed real scale range;
explicit subnormal/saturation handling must agree with the CPU oracle for all
16 x 256 code/scale pairs. Do not enable FTZ or fast-math in a bitwise proof.

Load order: validate v2 manifest + shape/source identity + SHA -> allocate/upload
v2 packed bytes -> validate live arch/SM count against the bake -> launch valid
expert groups. Keep offsets 64-bit for the >40 GB resident expert pool. Timing
must exclude repack/upload but include all expert computation, activation,
intermediate and reduction kernels required by the claimed path.

## 5. Bandwidth expectation and measurement gate

GB10 LPDDR5x model peak: **273 decimal GB/s**. Target >=191.1 GB/s (70%) for
EVERY M in {1,2,4,8}; <163.8 GB/s (60%) is a documented STOP. The 60–70% band
is below target, not a promotion. M={16,64} is also reported, without pretending
the decode bandwidth criterion is the right compute-regime throughput limit.

**Revised planning estimate after MiMo review, not a measurement:**

| Candidate | Expected effective GB/s at M<=8 | Meaning |
|---|---:|---|
| Inherited kernel, even after isolated correctness patches | review predicts <163.8 at every small M | STOP; do not spend a bandwidth window on it |
| B2 rewrite specified in §2 | **136.5–204.75 (50–75%)**, low confidence | spans STOP through marginal target; no promised 70% |
| Required acceptance | **>=191.1 for every M=1,2,4,8** | only actual sm_121 receipts can establish this |

This replaces my initial optimistic 70–80% planning range. Vector coalescing,
once-per-M-tile decoding and overlapping loads remove the known structural
multipliers; they do not prove a DRAM-limited kernel. At 191.1 GB/s and M8,
weight-only compute demand is about **5.76 TFLOP/s** plus integer decode,
shared-memory and reduction instructions. Payload bytes are 94.1% of stored
weight bytes; scale traffic, activation staging and instruction issue are not
free. Register spills or bank conflicts can keep B2 below 60% despite low
arithmetic intensity. Test the SASS/pipeline first, not register-count tweaks
to the old scalar structure.

One slice's weight-only floor is 12.24 us at peak, 17.49 us at target. Do not
compare these floors with a tiny cache-hot launch or infer serving tok/s.
**Exit decision:** any M<=8 below 163.8 means STOP; 163.8–191.1 means
BELOW-TARGET. Neither is eligible for I5 promotion. Present the receipt and
pivot to the allowlisted native MXFP4 family/B1 rather than call B2 "close
enough". That pivot needs a concrete source unit and REUSE row, not an assumed
Marlin throughput guarantee.

Benchmark gate: M={1,2,4,8,16,64}, real shape, fixed weights/inputs, correctness
first. Cover **256 active experts** and the model's **57.4 expected distinct
experts** with actual 57- and 58-expert rows (never allocate 57.4 experts).
If a 57.4 aggregate is reported, use a declared 60:40 mixture and divide summed
bytes by summed time, not average the GB/s values. Rotate disjoint address sets
from the 256-expert pool if a small active set fits L2; record its actual size.
Use CUDA events, warmup, repeated samples, synchronization, finite-output
checks and an active weight working set larger than L2. Print GPU, compute
capability, physical SMs, dtype, experts, local shapes, unique weight bytes,
logical activation/output traffic, elapsed time, commit/dirty state and Sydney
time. Distinguish **effective** GB/s from hardware-measured DRAM GB/s; a cache-hit
rate or counter is needed to attribute the latter. Reject apparently
super-peak DRAM claims until byte accounting/cache residency is checked.

dev-host RTX4090 sm_89 numbers are sanity proxies only. Never compare them to 273
GB/s as a Spark acceptance verdict. The builder runs sm_121 windows; the
coordinator/5090 and all fleet services remain untouched by this agent.

## 6. Prior art and differences

- ds41rt's official native expert format is already FP4 E2M1 + E8M0 K32,
  not its converted NVFP4/EXL3 path. Source map:
  `port-workspace/PORT-SURFACE.md` §0.1 and §5.
- Read-only `ds41rt-native-grouped-wob/docs/ds41-expert-grouped-slices.md`
  records sm_121 grouped M<=16 synthetic qualification: width64 wins one/two-row
  cases, width192 wins mixed/shared larger rows. Timings include fused FC1/
  activation/FC2 and ordered reduction, but exclude routing/transport and lack
  a DRAM/cache-pressure proof. They motivate regime-dependent tiling, not a
  transferable MiMo performance number. Do not inherit its padded-640 geometry.
- The WO-B split1/split2 worktree documents are coordinator FP8 experiments,
  not direct Spark MXFP4 bandwidth receipts. They demonstrate that a changed
  split-K association can alter BF16 rounding; replay numerical oracles before
  promoting a split-K schedule.
- Marlin-style weight-only kernels use packed coalesced loads, on-chip unpack,
  tensor-core reuse and workload-dependent tiling. Those principles apply;
  its integer/NVFP4 layouts, scale formats, activation dtypes and shuffle
  conventions do not automatically match checkpoint MXFP4. Our default layout
  is a pure byte partition with no opaque Marlin permutation. Any future
  permutation needs another explicit version and proof, not a silent loader fix.

No prior-art code body is copied by this design. Copy-in would require a REUSE
row from the builder before entering a build.

## 7. F1–F7 closure plan and acceptance tests

The B2 rewrite now compiles on sm_89/CUDA12.8; F1–F6 implementation changes
have host checks but **still need GPU qualification**. The new F7 native unpack
and real TP4/GEMM drivers are host-qualified and staged for GPU execution under
the current visibility policy. Bandwidth and remote GPU cells remain closed. Order remains:
v2 green -> design response -> rewrite -> correctness closures -> remote mode.

Compile-only receipt (2026-09-23 17:32 AEST): correct M-capacity1/2/4/8 uses
48/48/57/64 registers and2/4/8/16KiB shared memory, with **zero spills/stack**.
SASS contains128-bit global loads,128-bit async global-to-shared copies and
0/1 dependency waits; this establishes emitted instructions, not effective
latency hiding or bandwidth. Host tests compare12,288 arithmetic decode cases
(4096 correct +4096 unclamped +4096 force-one) to independent FP64/LUT math,
and pin row ownership, K coverage, swizzle/banks and BF16 RNE. A host-linked
CUDA-TU plan validator accepts sparse/empty groups and rejects9 malformed
plans without initializing a GPU. This is NOT L3 or a sm_121 compile receipt.

The new explicit v2 C ABI carries layout/identity/capacity, allocation lengths,
resident expert count separate from group count, device IDs/offsets and a fault
word. Prepare immutable grouping arrays from validated host metadata; initialize
the fault word before execution and check it on completion. FFN scratch is two
[T,512] f32 arrays (gate and up), then gate is reused for the local intermediate;
down reads exactly512 values/row and writes4096 outputs. M>8 is still an
M8-tiled SIMT fallback. The tensor-core prefill implementation is outstanding.

| Finding | Decision / implementation | Required negative or proof |
|---|---|---|
| F1: stale offsets | Repack `geom.rs` is authoritative. Checked-in `mimo26_slice_layout.h` is mechanically pinned: repack tests assert its definitions against canonical constants and expert tests assert them against the consumer. CUDA `geometry` derives from that header, not prose literals. Sparks receive the pinned header and need no Rust. | Perturb an offset and require the pin to fail; compare GPU unpack of actual Rust-produced slices against full source coordinates for every rank. |
| F2: contradictory layout statements | Remove v1-as-current tables/comments in Rust, CUDA ABI, tests and bench. All state gate/up [512,4096], down [4096,512], no reserved tail. | Geometry pins include shapes AND offsets AND v2, not only unchanged byte sizes. |
| F3: down scratch OOB | Down K=512, output N=4096. Validate sizes with overflow-safe arithmetic. Scratch contains only the rank-local intermediate. | Guarded scratch/output allocations, memcheck at M=1/8/64, and end-to-end FFN parity, not only three isolated GEMMs. |
| F4: wrong seam | Builder-approved v2 contiguous split in both crates; manifest-gated load rejects v1. | Existing new v1 load negatives; actual repack -> CPU expert -> independent full-expert sum at all nine fixture experts, then GPU equivalent. CPU real-identity cell passed 27/27 cases (max_abs 2.384185791e-6); GPU identity remains pending. |
| F5: zero-byte staged path | Delete full-M/full-K staging. Bounded two-stage K256 allocation, <=16 KiB at M8. If a distinct direct-global fallback exists, it has no shared access, not a boolean contradicted by launch bytes. | Validate stage bytes/limits, exercise M8/16/64 tails, sanitizer shared-memory/race checks, barrier-uniformity review. |
| F6: duplicate writers | Eight cooperating lanes/row, one leader store, 32 rows/CTA. | Exact coverage test for every (row,token); sparse-ID/padded-group cases; deterministic reductions. No duplicate-store exception. |
| F7: impossible real proof | Separate full-checkpoint sample decode from repacked-layout decode. Direct full-tensor addresses for all fixture samples; no synthetic quarter, no skipped rows. | Exactly 27 x 2048 GPU output bits, both source hashes, oracle comparison; GPU real-layout coordinate proof separately. |

Additional T10 mutation: **force scale to 1.0** literally, alongside off-by-one
and byte-255 unclamping. Each mutation must fail independently; an ALL-flags
failure is not evidence that each trap was detected. Preserve existing CPU
trap classifications and add flags consistently across Rust and CUDA.

Sparse group IDs need an explicit validated device ID table, distinct from
resident slice count; zero/padded groups must not index it. Host launch metadata
contains checked total rows, maximum local M, allocation lengths and layout
version. Do not copy device offsets back to the host on each timed launch.
Pointers alone cannot prove sizes: validated descriptors own that boundary.

**Numerics:** parallel K reduction changes association relative to the CPU
sequential dot. Require componentwise `atol=1e-5, rtol=1e-5` and finite results
for the declared f32 cell, including cancellation-sensitive random inputs.
Do not weaken tolerance after a failure. FP32 FMA, SiLU and fixed-rank summation
must each be covered. BF16 output conversion is tested separately with RNE.

**Bench verdict:** CUDA prints raw timings/identity, not an independent PASS
policy. After local execution or remote result retrieval, the dev-host-side Rust
reporter uses the unit-tested `bench::Verdict` and `bench::report` logic. Thus
Spark stays nvcc-only without duplicating thresholds. Unsupported identity,
missing rows, nonfinite/zero timings and unverified correctness refuse a report;
60–70% says BELOW-TARGET, never PASS.

## 8. Spark transport contract (after local correctness)

`run_gpu_gemm.sh` defaults to the dev host/sm_89/CUDA12.8. `HOST=sparkN` stages kernel
sources, pinned layout header, fixture metadata and standalone C++/CUDA
drivers below `/var/tmp/i4-expert/`, builds with `/usr/local/cuda/bin/nvcc`
CUDA13.0 for **sm_121**, and reads existing remote weights at
`/var/tmp/models/MiMo-V2.6-Flash-RL`. No Rust or Python is required on Spark.
Use a unique run subdirectory so stale binaries/results cannot be mistaken
for a new cell. Verify transferred source hashes and record the exact compiler.

`--dry` must print every rsync/ssh/nvcc/run/retrieval command without creating
remote state, compiling or launching. Results return under
`runs/20260923-i4/expert/`; retain failure receipts and exit status. There are
no container start/stop/restart commands: builder owns the window and restores
and probes the service. Refuse concurrent CUDA owners and inadequate memory.
Exact runnable window #2/#3 commands will be in the packet only after the
remote harness and dry-run behavior are tested; this section is not readiness.

Native FP4 MMA is an optional later prefill candidate. Do **not** assume the
review's `tcgen05` suggestion applies to compute capability 12.1 merely because
GB10 is Blackwell: distinguish architecture-specific instruction families,
check compiler/ISA support, and qualify the matching sm_121 path separately.
This does not dispute F1–F7 or delay the portable CUDA12.8/13.0 B2 baseline.

### Builder execution contract (2026-09-23 update)

Latest policy (builder17:59AEST) permits direct pg4090 iteration **only after
nvidia-smi succeeds**. Check visibility, ownership and free memory before each
GPU run; leave at least4GiB free including our buffers. While visibility fails,
queue the same committed cell for builder. Spark/5090 remain builder-owned:
no SSH to the coordinator/Sparks and no container operations by this agent. Packet heading
`## BUILDER: RUN THESE` carries those commands, estimates and stop criteria.
The legacy `MIMO26_BUILDER_GPU` opt-in flag is retained for command compatibility.
The native `gemm_parity.cu` driver reads safetensors directly with bounded JSON,
range/shape validation and first-party SHA256. Its host audit verifies all54
source tensor hashes and rejects6 deliberately malformed fixtures. The source
hash checks precede each tensor's GPU decode. No source bytes come from answers.

`MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm unpack` stages the first proof:
full-checkpoint output for all27 blocks,55,296 sampled bits, external comparator,
and individually wrong nibble/scale-index/force-one implementations. A separate
GPU synthetic exhaustive decoder check covers signed zero, subnormals, all256
scale bytes and the unclamped255 negative. Exit3 is only numerical mismatch;
CUDA/input/timeout/guard failures cannot satisfy a negative. The cell requires
a clean worktree, one free RTX4090 and no competing CUDA owner. The explicitly
authorized permanent Python modules `service.cotenant_a.cotenant_a` and
`service.cotenant_b.cotenant_b` may coexist; match exact live argv, never PIDs or substrings.
The shared expert-side helper rejects all other or uninspectable owners. Memory
reserve checks remain unchanged. It logs all output to its unique workspace slot. **Await GPU receipt before claiming F7.**

### Real TP4/GEMM proof driver (staged, not GPU-qualified)

`tp4-driver-selftest` builds native code, regenerates independent FP64 goldens
from full checkpoint tensors, runs the actual Rust repacker, and audits all36
images against manifest v2/SHA and literal full-source coordinates. Gate/up,
rank partial, and isolated-down goldens are additional outputs of the existing
external NumPy oracle, not reconstructions from device answers. Ten tamper
negatives include v1 with intact bytes, stale offsets, a forged new SHA for a
wrong rectangle, nonfinite references, and a wrong-but-finite partial sum.

`tp4-gpu` is the queued hardware proof. It covers nine real experts at
M1/2/4/8/16/64:54 full TP4 sums,216 rank partials,648 isolated GEMMs, and all
55,296 real fixture sample bits decoded from Rust-generated slices across all
ranks. It also exercises27 sparse/empty/ragged plans,9 zero-token plans and18
deliberately corrupted device metadata arrays. The four-entry resident pool
contains the four rank images of one real expert; **this is not a256-expert
resident or full-prefill proof**. Input/scratch/output/weight redzones preserve
128-byte vector-load alignment. Every allocation preflights the4GiB reserve.

Separate wrong paths: nibble swap, shifted scale index, literal scale1, actual
padding read, and BF16 accumulation. The BF16 detector uses a clearly synthetic
W=1, x=f32(0.001), K4096 matrix with independent exact-real sum4096*x; its FP32
positive must pass before the wrong accumulation runs. Padding must produce
its actual nonresident-byte fault (typed exit5); numerical wrong paths require
exit3. CUDA, loader, guard and timeout failures never qualify as negatives.
Class2048 live identity and wrong arch/SM/capacity refusals are included; other
capacity classes, sanitizer and full-prefill routing are still separate gates.

### Capacity-boundary routing supplement

`capacity-host` compiles class256/2048/4096 separately and host-tests the linked
plan validator plus a synthetic256-expert fixture. `capacity-gpu` exercises the
actual FFN at2048/16384/32768 routed rows (M8/64/128), with256 unique full-size
images, permuted resident IDs and all top8 routes per source. An independent
analytic/token-first oracle checks all outputs and weighted scatter. A wrong
ordinal-for-ID upload must fail numerically. Live arch/SM/capacity refusals run
in each executable; compilation and CPU fixtures alone never satisfy AOT.

This is functional full-shape coverage, not real-checkpoint256 or a benchmark.
The deliberately sparse synthetic matrices are unsuitable for bandwidth claims.
The largest device buffers total approximately1.93GiB; allocator checks preserve
a further4GiB free. Resident eye/ear services may coexist; other owners block.

### Executed dev-host receipt and subsequent fleet ruling (23 September AEST)

The full unpack and TP4 proofs above passed on dev-host sm89/128SM at0701318
(re-cut as0035956 with identical expert/repack sources; binary receipts removed):55,296 real bits,54 TP4 sums,216 partials,
648 isolated GEMMs and every independent negative. TP4 max_abs2.384185791e-7.
All three live AOT classes and full256 **synthetic** routing also pass. See
`runs/20260923-i4/expert/local-0701318/RESULT.md`; these are not bandwidth or
Spark receipts. Sanitizer and actual-checkpoint256 remain open.

ADVISOR-I4§9 now authorizes KERNEL-LEAD to run directly on Spark1–4 under its
own-process/scratch guardrails, superseding prior builder-only remote wording.
The next gate is sm121 nibble proof, then M1..8 bandwidth, then sm121 AOT.
Below163.8GB/s STOP;163.8–191.1 means pivot to B1, not polishB2. The accepted
V2-F2 geometry binding is a landing condition; V2-F1 weighted BF16 ReturnRow
through CoordinatorSum is a separate required pre-I5 gate, not certified by
our FP32 TP4 proof.

### First Spark gate and geometry landing condition (20:05–20:09 AEST)

Spark1 sm121/48SM, nvcc13.0.88 at e173a04: real unpack55,296/55,296,
54 source hashes,131,072 decoder outputs and four independent negatives pass;
external dev-host comparator agrees. Native-only receipt:
`runs/20260923-i4/expert/spark1-20260923-200519-e173a04e7a7d-expert-spark-unpack.DSf1rE/RESULT.md`.
No Spark FFN/bandwidth/all-class AOT promotion yet.

V2-F2 is now mechanically covered by `mimo26-repack/tests/cuda_layout.rs`:
not just the header values against `geom.rs`, but the actual CUDA projection
function must use that pinned macro table, without local overrides. Stale-table
negatives leave the header intact. Every projection row/K256 tile is checked
for128-byte payload and8-byte scale alignment (V2-F3b). The five-test
`geometry-selftest` cell passes; no production CUDA changes were necessary.

### Current B2 verdict: STOP (2026-09-23 20:33 AEST)

At f74ed91, the real256-expert Spark1 E-FP32 benchmark passed its correctness
checks but **failed the all-M≤8 performance gate**. M1–4 reached218.81/215.92/
196.38/192.46GB/s; M5–8 reached136.89/135.95/134.40/132.70GB/s, all below163.8.
Exit6 STOP is retained, not retried away. No B2 tuning or AOT promotion follows.
Canonical receipt/table:
`runs/20260923-i4/expert/spark1-20260923-203239-f74ed91e021b-expert-spark-bench.bHamDv/RESULT.md`.

Updated at dd8de19: P-LATTICE is ruled in `docs/design/lattice-v1.md` and B1
E-W4A8-v1 belongs to lead on `b1/prep`; the compute REUSE rows are released.
KERNEL-LEAD does not duplicate B1 work. B2 remains the E-FP32 correctness/reference
implementation; the failed bandwidth gate is not erased by the lattice ruling.

### Pre-committed Spark timing cell

`HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-bench`
loads256 distinct real layer1 experts directly on Spark, packing rank0 with
bounded native reads. All source hashes are recorded; only experts0/7/255
have independently pinned fixture hashes. No model weights cross hosts. The
external NumPy/FP64 x/partial goldens for those three experts are generated on
the dev host and copied as test inputs into scratch (not receipts).

For **each M1..8**, check those three full FFN partials, finite outputs for all
residents, fault word and redzones before timing; the actual nibble-swap GPU
implementation must fail first. Then10 warmups and31 individual CUDA-event
samples bracket the complete three-GEMM+SiLU FFN. Recheck after timing. Report
raw samples and the median. Numerator is exactly **3,342,336 × resident count**
unique weight+scale bytes, once for M≤8; divide by milliseconds×1e6 for decimal
GB/s. This is effective weight bandwidth, **not a measured DRAM counter**; I/O,
SiLU and launch costs remain in the denominator. Require resident bytes>2×L2.
Clocks stay unlocked. No throughput number above273GB/s can qualify.

The bounded batch measures all eight rows before its final decision; any
row<163.8 returns6 **STOP**, otherwise any row<191.1 returns7 **PIVOT B1**.
No failed threshold is silently treated as success.57/58 distinct-real-resident
variants are selectable, but are not claims about an actual router distribution.
Full256 numerical goldens, sanitizer and all-class Spark AOT remain separate.

### V2-F1 bound counterexample (2026-09-23 20:52 AEST)

The CPU-only `wire-boundary` cell now links actual `mimo26-expert` and
`mimo26-wire` libraries, without copying codec bodies or changing Cargo.lock.
`rank_sum::rank_pre_sum` is a finite-checked CPU reference: unweighted
[token,8,4096] FC2 outputs → FP32 weight once → slot-ordered rank pre-sum.
It is not a GPU reducer or a final real-checkpoint seam qualification.

An exact dyadic case disproves the adopted wire bound: every rank pre-sum is
1.00390625; compute error is zero; BF16 RNE rounds each to1.0. The actual
ReturnRow encode/decode and rank0→3 CoordinatorSum produce4 instead of4.015625.
**Wire error0.015625 > prescribed0.007845417578125**, all4096 coordinates.
The strict probe exits3; the bound is unchanged and V2-F1 remains open pending
builder/reviewer correction. See
`runs/20260923-i4/expert/wire-boundary-20260923-205237/RESULT.md`.

CoordinatorSum accumulates in caller order, not internally sorted rank order.
Its actual BF16 inputs [2^24,1,-2^24,1] yield1 in order0,1,2,3 versus2 in
order0,2,1,3. The serving integration must buffer/order ranks explicitly.

### R6/R8 update (23 September 2026, 21:06 AEST)

The builder explicitly corrected the wire bound in R8 to
`2.4e-6 + Σ |y_r| × 2^-8 + 3 × 2^-24 × Σ |decoded_y_r|`.
The same tie fixture passes at **0.015625 ≤ 0.015689150411987**; its legacy
failure is retained. The integration probe buffers decoded rank frames and
verifies all **24 arrival permutations**, including the order-sensitive
cancellation fixture. It refuses duplicate/out-of-range ranks. This is CPU
integration scaffolding, not the outstanding real GPU V2-F1 receipt.
Evidence: `runs/20260923-i4/expert/wire-r8-20260923-210613/RESULT.md`.

R6 assigns the B1 compute core, lattice-oracle FFN reference and integration to
KERNEL-LEAD after V2-F1. lead owns prepare/staging/group-plan/route-reduce and
the CUDA quantizer on `b1/prep`; integrate those units pull-style. Builder owns
the Python codec in `oracle/lattice/`. No per-M lattice mixing or B2 polishing.

### V2-F1 real reference closure (23 September 2026, 21:37 AEST)

Source `b650ad4`, Spark1 GB10 sm_121 / 48 SMs, real layer 1 experts
`[0, 7, 255, 1, 2, 3, 4, 5]`, all four logical rank slices, M = 1, 2, 4, 8:
**61,440 coordinates pass R8** after GPU FC2 → CPU FP32 weight-once slot
pre-sum → actual BF16 ReturnRow encode/decode → buffered CoordinatorSum.
Worst compute error is **3.475963220034e-8**; worst wire error/bound ratio is
**0.822500675**. Independent FP64 oracle and all 48 source hashes agree;
actual GPU nibble-swap and real-output weight-twice negatives fail as required.
The tie, legacy bound failure and all 24 order permutations remain pinned.

This closes **V2-F1 numerical identity for the E-FP32 reference path**, not
GPU route reduction, four-host transport, serving integration or B1 qualification.
Receipt: `runs/20260923-i4/expert/spark1-20260923-213715-b650ad40c2b5-expert-spark-wire.hvpOVm/RESULT.md`.
B2 remains stopped for performance; the next critical-path work is B1 compute.

### B1 primitive gate (23 September 2026, 22:06 AEST)

Source `a06161f`, Spark1 GB10, architecture 121 / 48 SMs, explicit
`compute_121a → sm_121a`: approved E2M1 container and block-scaled MMA wrappers
pass GPU tests. Four K32 blocks in a 16 × 8 × 128 asymmetric product match
FP64 exactly at all 128 outputs. All 4,096 code/scale pairs pass eight-position
container recovery and conforming decode; **55,296 real fixture bits / 54 hashes**
also pass. Actual nibble, scale-byte and scale-lane negatives are detected.
The initial shorthand-codegen compile failure is retained, not replaced.

This does **not** qualify exceptional-scale MMA, the full B1 FFN or performance.
Those gates still require the released slice schedule, v1 activation boundaries,
conforming special-scale handling, builder-codec FFN oracle and lead integration.
Receipt: `runs/20260923-i4/expert/spark1-20260923-220639-a06161f841ac-expert-spark-b1-primitive.eYdNUI/RESULT.md`.

### Independent E-W4A8-v1 FFN reference (23 September 2026, 22:23 AEST)

Source `13b0fac`, `scripts/dev.sh test gemm lattice-oracle-real`: CPU reference
uses builder quantizer v1 and the spike MXFP4 decoder, never CUDA/Rust logic.
FP64 dots surround the explicit FP32 FC1/activation boundaries; intermediate
encoding is unweighted K32, followed by FP32 FC2 output/weight/slot sum and
rank-local BF16 return. Real eight-expert full/TP4 and M = 1, 2, 4, 8 prefix
identities pass, with 48 source hashes. The 22 codec tests and nine FFN/wire
mutation detectors pass. Stage arrays and source hashes are available for the
future GPU comparator, not a GPU or model-quality qualification by themselves.
Receipt: `runs/20260923-i4/expert/lattice-oracle-20260923-222252-13b0fac/RESULT.md`.

### B1 slice-math gate (23 September 2026, 22:48 AEST)

Source `5353d76`, Spark1 GB10 architecture 121 / 48 SMs: released compute-only
width 64/128 FC1 and FC2 routines pass M = 1, 2, 4, 8, 16, 64 projections:
**832,960 active + 429,632 padded coordinates**. Stable activation is FP32;
no BF16 cast, clamps, route weight or atomics occur in this unit. Exceptional
weight scales 0/1 and 253–255 use conforming decode-before-FMA; 640 probe
coordinates pass, and bypassing that arm fails the high-scale cases. Actual
gate/up swap, clamp and BF16 FC1 negatives are detected.

This is not the fused quantized FFN. Staging, quantizer v1, ordered reduction and
production launch remain integration work; M64 math grouping is not optimized
prefill qualification. No performance result is promoted.
Receipt: `runs/20260923-i4/expert/spark1-20260923-224709-5353d76f4667-expert-spark-b1-primitive.V5dQJf/RESULT.md`.

### Compute-only sanitizer gate (23 September 2026, 23:11 AEST)

Source `0da1c794`, Spark1 GB10 architecture 121 / 48 SMs, Compute Sanitizer
2025.3.1.0: memcheck/full leak check, initcheck, synccheck and racecheck pass all
12 projection and 5 special-scale cases. This includes the FP8-only compile-time
FC2 policy refactor. No errors/leaks or reported hazards/warnings. The 32 harness
failure controls are CPU log fixtures, not injected GPU faults.

The fused staging/quantizer/reducer sanitizer gate remains open; racecheck does
not establish global-memory race freedom. No performance result is inferred.
Receipt: `runs/20260923-i4/expert/spark1-20260923-230949-0da1c79477a5-expert-spark-b1-sanitize.QIkCDP/RESULT.md`.

### Complete rank-local projection gate (23 September 2026, 23:34 AEST)

Source `0edc3e1`, Spark1 GB10 architecture 121 / 48 SMs: full N512/K4096 FC1
placement and N4096/K512 FC2 accumulation pass widths 64/128 and
M = 1, 2, 4, 8, 16, 64. **1,070,080 active + 551,936 padded coordinates**
pass scalar-reference checks. Channel-placement, accumulator-reset and K-offset
negatives fail as intended. FC2 retains its accumulators across K slices, so it
can supply complete unweighted routes without a separate slice-plane reducer.
Group scratch is padded to 16 rows; do not index it by an unpadded route base.

No quantizer-connected FFN, real B1 parity, new sanitizer or performance claim.
Receipt: `runs/20260923-i4/expert/spark1-20260923-233144-0edc3e188f1f-expert-spark-b1-primitive.6VRxsJ/RESULT.md`.

### Complete-rank sanitizer extension (23 September 2026, 23:45 AEST)

Source `1196d27`, Spark1 GB10 architecture 121 / 48 SMs: all four Compute
Sanitizer tools now pass the complete-rank projection cases as well as the prior
slice and scale cases (**12 + 12 + 5** per tool). No errors/leaks or reported
hazards/warnings. This supersedes the earlier math-only coverage gap, not the
still-open connected staging/quantizer/reducer gate. No performance inference.
Receipt: `runs/20260923-i4/expert/spark1-20260923-234244-1196d2732e94-expert-spark-b1-sanitize.d4MmfV/RESULT.md`.

### Connected v1 FFN (24 September 2026, 01:58 AEST)

Source `c3945b9`, Spark1 architecture 121 / 48 SMs: actual lead preparation,
plan, staging, independent CUDA quantizer and ordered BF16 reducer connected to
B1 compute. M = 1 through 8, four ranks, eight real layer 1 experts pass every
fixed-bound stage comparison. Same-input codec bytes and ordered reduction/
return are exact; no ideal intermediate crossings. Actual double weighting is
rejected. Receipt:
`runs/20260923-i4/expert/spark1-20260924-015423-c3945b919ddb-expert-spark-b1-ffn.6NSzkl/RESULT.md`.
This does not close connected sanitizers, AOT classes, optimized M16/64 or
57/58-resident qualification.

### Frozen B1 bandwidth cell (pre-run)

`HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-b1-bench`.
Width 128 for every M, all 256 distinct layer 1 rank-0 images, unique weight+
scale bytes = 855,638,016 (must exceed twice L2). Independent FP64-derived
reference checks **every expert/output**, before and after timing. Actual
premature-weighting control must fail. Input payload/scales stay encoded v1.
Timed scope is FC1 gate/up + SiLU + intermediate quantizer + FC2; as for the
B2 math gate, preparation, plan construction, upload, route reduction and wire
are outside the event interval. This is not end-to-end serving latency.
Ten warmups, 31 CUDA-event samples, median; unchanged decimal byte/time metric,
191.1 target and 163.8 STOP boundary. Record every M = 1 through 8; any STOP
or PIVOT prevents promotion. No per-M lattice mixing or retry-and-replace.

### Initial connected B1 bandwidth verdict: STOP (24 September 2026, 02:23 AEST)

Source `1e2d716` ran the frozen cell above on Spark1. All 256 rank-0 experts
pass correctness before/after timing, but **every M is below 163.8 GB/s**:
M1–8 = 160.654472, 160.048658, 158.563299, 157.134807, 155.228259,
153.684747, 152.501437, 152.782905. Native exit 6; all samples retained.
No performance promotion, tuning, per-M replacement or lattice mixing. The
remaining promotion ladder stops pending the next design decision. The i16
hedge has no implied bandwidth qualification.
Receipt: `runs/20260923-i4/expert/spark1-20260924-021949-1e2d71613a8f-expert-spark-b1-bench.gNsh9t/RESULT.md`.

### R12 diagnostic profiles (24 September 2026, 03:13 AEST)

R12 lifts the blanket B2 M5–8-work ban but requires evidence before redesign.
Spark1 full-set NCU captures now cover B1 M1/M8 and B2 M4/M5; all capture and
native correctness checks pass. This does **not** replace either frozen STOP
receipt. [Counter/stall handoff](../../runs/20260923-i4/expert/r12-profiles-20260924-0313/RESULT.md)
records every stage, source identity, limitations and retained raw exports.

B1 has severe issue starvation (FC1 eligible warps/scheduler about 0.13),
long-scoreboard dominance and shared-bank conflicts. B2 changes from a four-row
to an eight-row compiled tile at M5: registers 48 → 64, register block limit
5 → 4, theoretical occupancy 83.33% → 66.67%, with padded arithmetic. Its L2
hit rates increase rather than collapse. These observations motivate review,
not a selected redesign or promised GB/s.

Counter completeness remains open: this GB20B/NCU catalog exposes no direct
`dram__` counters, and several uncontrolled-replay achieved-occupancy readings
are impossible or exceed static limits. No surrogate DRAM claim, clamping,
clock-policy change or threshold waiver was made. Builder decision requested
for MiMo Pro's ranked options/predictions and these profiling limitations.
Per-M lattice mixing remains forbidden; no redesign has started.

**03:37 AEST supplement:** those first captures used instrumented rebuilds.
The actual retained frozen B1/B2 executables have now also been profiled without
rebuilding, with before/after executable hashes and verified launch selectors.
[Exact-binary receipt](../../runs/20260923-i4/expert/spark1-20260924-032632-fa24cd35280d-expert-frozen-profile.yljaDq/RESULT.md)
adds the explicit issue-slot/register/spill/tensor table and one attribution
paragraph for each of 14 captured kernels. B1 main-kernel tensor activity is
3.62–4.85%, issue-slot use 11.54–13.90%; observed spilling requests are zero
in every kernel. B2 issue-slot use is 71.10–73.13% at M4 versus 65.08–67.23%
at M5. Direct DRAM remains unavailable and inconsistent occupancy remains
flagged. NCU-emitted event timings are not replacement bandwidth gates.

## 9. Evidence required before promotion

1. CPU correct suite green; naive run fails only designated traps.
2. Real bytes: 27 blocks x 2048 samples, both source SHA256s, bitwise GPU dump;
   swapped-nibble/scale negatives fail. CPU proof is only a preliminary check.
3. Real full FFN identity: actual Rust repacker -> actual expert implementation,
   sum four rank partials vs independent full-expert oracle; M=1/8/64, all nine
   fixture experts, random inputs, finite outputs and componentwise tolerance.
4. GPU grouped L3 includes sparse IDs, zero/inactive/padded groups, diverse M,
   bounds guards and the full grouped-prefill routing shape.
5. AOT classes {256,2048,4096}: validate baked and live arch AND physical SM
   count, capacity and layout version. Reject foreign SM and corrupted manifest
   BEFORE launch; CPU policy tests are not a real-device positive gate.
6. Bounded dev-host proxy, then builder Spark bandwidth window; retain failed rows.

## 10. R13 diagnostic findings and next prototype (24 September 2026 AEST)

R13 explicitly assigns this work to KERNEL-LEAD on **spark1**, with the lead's B1
MLP work on **spark2**. This supersedes the older builder-relay wording above.
No cross-scheduling or per-M lattice mixing.

- [M2/M3 PC receipt](../../runs/20260923-i4/expert/spark1-20260924-035324-df3a92852767-expert-frozen-profile.K1ARJx/RESULT.md):
  template 2 → 4, registers 40 → 48, shared 4 → 8 KiB, static FFMA 64 → 128.
  All 48 selected PC totals match raw aggregates. F2's recurring-per-row
  +394 µs premise is not established; the frozen delta remains unchanged.
- [Forced-four M5 receipt](../../runs/20260923-i4/expert/spark1-20260924-041927-ab8e05b3312e-expert-spark-b2-force4.g3ayjt/RESULT.md):
  resources really return to 48 registers / 8 KiB / 83.33% theoretical
  occupancy, but the unprofiled diagnostic is **101.148811 GB/s, STOP**.
  Adjacent CTAs duplicate weight/decode work. Higher eligibility/issue use is
  not a speedup. This rejects profile restoration as sufficient, not uniquely
  identifies a staging-map fault. It is not the staged-once option-1 design.

The next low-risk prototype is exact M5/M6/M7 specialization, with M8 as an
unchanged-work control, preserving the existing eight-lane K partition and
per-row FP32 operation order. Equal times across the old padded-eight cases
do not establish that padding is free. This experiment alone is not expected
to qualify M8; it must not be presented as a completed all-M solution.

A follow-up worth evaluating is a once-per-scale guarded normal-range decoder
path, motivated by the PC attribution, with exhaustive bit checks and the full
exceptional-scale path retained. It must neither clip nor change E-FP32. Any
combined candidate needs its own parity and frozen all-256 gate; no predicted
speedup is accepted before that gate.

A dual-four-row design also needs an honest state/traffic ledger: eight parked
FP32 accumulator values per 256 threads cost **8 KiB** before activation or
weight staging. Holding a full gate/up weight tile costs **64 KiB payload plus
4 KiB scales** at 32 output rows. Neither can silently be called the original
8 KiB shared-memory profile. Shared/register time-sharing must preserve per-row
FP32 order and account for swap traffic, pipeline depth, register pressure and
spills. No version of that design is yet implemented or promoted.

The builder-requested 04:50 AEST [packed-FP32 compile probe](../../runs/20260923-i4/expert/spark1-20260924-045006-7f3e7fb1dcb5-expert-ffma2.FMFNx2/RESULT.md)
finds that CUDA13.0.88 accepts both `fma.rn.f32x2` and `__ffma2_rn` for
compute_121a/sm_121a, but emits **two scalar FFMA**, not FFMA2. The identical
sm_100a probes emit one FFMA2. Do not budget a packed-instruction doubling on
GB10; no numerical or throughput claim follows from this compile-only test.
The ~80.7% issue activity was forced-four **M5**, not its M4 control (~72.8–72.9%).

The 05:07 AEST [exact-M control](../../runs/20260923-i4/expert/spark1-20260924-050704-769610ae82b4-expert-spark-b2-exact.1WSBkd/RESULT.md)
(`769610a`) passes its benchmark numerical/negative checks but returns **STOP**:
M5/6/7/8 **171.078331 / 156.830664 / 143.735518 / 131.730007 GB/s**.
Padding removal helps; it is not an all-M solution. sm121 M5 uses62 registers,
M6–8 use64, all without static spills. The four-block resource class persists,
with added-row costs454/497/543µs. Those costs concern this instruction mix,
not a proven universal E-FP32 ceiling. No default-policy change or promotion.

R14's [completed model pin, 05:42 AEST](../../runs/20260923-i4/expert/spark1-20260924-053953-19b2296730f1-expert-frozen-profile.V7ZVFE/RESULT.md)
fails at M4: actual executed-warp count ×32 / stored byte is59.103537 rather
than49±15%; actual M8 is71.145204 versus65±15%. The original M8 SASS already
contains64 LDS.128 instructions per GEMM. Re-anchor before crediting further
packed-load savings or the214–229GB/s prediction. **Implementation stopped
pending builder re-anchoring**, as explicitly requested; no decoder prototype
started. The170GB/s recast-prototype kill line is not a profiler-data threshold.

R15 (06:05 AEST) subsequently freezes B2 best-effort and makes B1 sole primary,
applying the kill consequence to the re-anchored **prediction**, not a measured
decode2 prototype. No decode2 implementation is authorized.

The [06:15 AEST reference exit receipt](../../runs/20260923-i4/expert/spark1-20260924-061059-924924ff2c7f-expert-spark-i4-reference.AyGwGH/RESULT.md)
(`924924f`) qualifies native B2 **sm121 AOT256/2048/4096** on GB10: full256
synthetic routing at M8/64/128, wrong-manifest zero-node launch refusals and
powered bypass/ordinal negatives. Connected **E-W4A8-v1** passes all four
sanitizers at M1/M8 across four sequential rank slices and eight real experts,
plus independent numerical/byte-boundary comparisons. Initcheck uses
uninitialized payloads and a real failing read-before-write control. The
57/58 single-layer/rank0 resident pools pass correctness but retain native
performance **STOP** (M8 129.015208/127.257882GB/s). This is not full-model
residency, optimized-prefill qualification, future-B1 qualification, or I4 close.
