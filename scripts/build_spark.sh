#!/usr/bin/env bash
# Build the mimo26-spark serving crate with the CUDA kernel FFI linked, then run
# the device smoke (m26x_device_identity + AOT gate). Invoked by
# `scripts/dev.sh build spark`.
#
# Local dev proxy defaults: the dev host's RTX 4090 (sm_89 / baked arch 89 / 128 SMs /
# class 2048). A Spark build bakes sm_121a (arch 121 / 48 SMs) — set the
# MIMO26F_*_ env vars for that (the remote Spark cell drives it), or pass
# through. One CUDA build at a time: the caller holds no lock; the builder
# serialises GPU batches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export MIMO26F_CUDA_ARCH="${MIMO26F_CUDA_ARCH:-sm_89}"
export MIMO26F_BAKED_ARCH="${MIMO26F_BAKED_ARCH:-89}"
export MIMO26F_BAKED_SMS="${MIMO26F_BAKED_SMS:-128}"
export MIMO26F_CAPACITY_CLASS="${MIMO26F_CAPACITY_CLASS:-2048}"
export MIMO26F_CUDA_LIB="${MIMO26F_CUDA_LIB:-/usr/local/cuda/lib64}"
printf 'build spark: arch=%s baked=%s/%s capacity=%s\n' \
  "$MIMO26F_CUDA_ARCH" "$MIMO26F_BAKED_ARCH" "$MIMO26F_BAKED_SMS" "$MIMO26F_CAPACITY_CLASS"
exec cargo test --manifest-path "$ROOT/Cargo.toml" -p mimo26-spark --features cuda -- --nocapture
