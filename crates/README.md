# crates/

| Crate | What it is |
|---|---|
| `mimo26-coordinator` | The coordinator binary: model forward, scheduler, KV cache and snapshots, DFlash, expert-wire client |
| `mimo26-api` | OpenAI-compatible HTTP API (A8) and the MiMo tool-call parser |
| `mimo26-attn` | Attention kernels (GQA + SWA + sink), paged KV, split-KV decode, AOT gates |
| `mimo26-expert` | MXFP4 expert GEMM kernels (B1) and their reference harness |
| `mimo26-spark` | The Spark expert-rank daemon |
| `mimo26-wire` | DS41RTE3 v3 frames and CRC32C |
| `mimo26-rdma` | RDMA RC transport (rdma-core verbs) |
| `mimo26-repack` | Expert slice staging for the ranks (`mimo26-repack --rank r`) |
| `mimo26-load` | Checkpoint loading (fused QKV, scale grids) and the name audit |
| `mimo26-lanesim` | Discrete-event simulator of the expert path |

Module map and REUSE allowlist: `../ARCHITECTURE.md` §4–§5.
