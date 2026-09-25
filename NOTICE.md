# Third-party notices

This repository is MIT-licensed ([LICENSE](LICENSE)). The components below come from other projects and keep their own licences. [docs/REUSE.md](docs/REUSE.md) records the source, commit and file digest of every copied unit.

## DS41RT: MIT

- **Source:** <https://github.com/tpurtell/ds41rt>. Copyright (c) 2026 T.J. Purtell.
- **What was taken:**
  - the expert prepare and route-reduce units in `crates/mimo26-expert/kernels/b1/` (`prepare.cu`, `prepared.cuh`, `route_reduce.cu`);
  - the designs of the wire format (`DS41RTE3` v3), the RDMA transport and the host snapshot cache. Per `docs/REUSE.md`, the code for these was written in this project, not copied.
- **Licence text:** `crates/mimo26-expert/kernels/b1/LICENSE.ds41rt`.

## b12x: Apache License 2.0

- **Source:** <https://github.com/local-inference-lab/b12x>. Copyright (c) 2025 the b12x authors.
- **What was taken:** the block-scaled MMA primitive and the W4A8 slice schedule behind the B1 expert kernel.
  - Files: `crates/mimo26-expert/kernels/b1/` (`mxfp4_ptx.cuh`, `grouped.cu`, `group_plan.cu`, `staging.cuh`, `mlp_staging.cuh`).
  - Changes are stated in each file's header.
- **Licence text:** `crates/mimo26-expert/kernels/b1/LICENSE.b12x` and `LICENSE.b1-compute`.

## tonyd2wild MiMo-V2.6-Flash DGX Spark recipe: MIT

- **Source:** <https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe> at `13621bb`. Copyright (c) 2026 Tech2wild.
- **What was taken:** the bench and probe tools in `harness/fleet/tonyd2wild/`, byte-for-byte: `mimobench.py`, `mimo_needle.py`, `replay_exact.py`, `stress-corrupt.py` and `toolcap-proxy.cjs`.
- **Licence text:** `harness/fleet/tonyd2wild/LICENSE`.

## rdma-core headers: GPL-2.0 or OpenIB.org BSD

- **Source:** the verbs headers in `crates/mimo26-rdma/native/include/` come from linux-rdma/rdma-core.
- **Licence:** they are dual-licensed, and are used here under the OpenIB.org BSD licence.
- **Copyright:** each file's header carries its notices.

## Not included

- **Image decoding:** the vision input reimplements the behaviour of libjpeg-turbo and Pillow. No code from either is included; `docs/REUSE.md` records the provenance.
- **The model:** the MiMo-V2.6-Flash-RL weights and drafters are distributed by Xiaomi under the model's own licence.
- **CUDA:** the toolkit and cuBLAS are linked at build and run time.
