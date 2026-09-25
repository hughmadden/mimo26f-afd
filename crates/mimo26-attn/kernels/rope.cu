/* RoPE — T19/T7: FP32 on-the-fly angles (`float(pos) × inv_freq`, no cos/sin
 * table), partial rotary 64 of 192 dims (0.334), dual θ (1e7 GA / 1e4 SWA).
 *
 * Pair layout matches `oracle/mimo26/nn/layers.py apply_rotary`: pair j joins
 * channels (j, j + rot_dim/2) within the FIRST rot_dim channels; channels
 * beyond rot_dim pass through untouched (T7's "partial, not full width").
 *
 * Naive switches (the CPU suite's T19/T7 negatives):
 *   M26_NAIVE_ROPE_TRUNC_TABLE   angles from `pos % 32768` (a 32K cos table)
 *   M26_NAIVE_ROPE_FULL_WIDTH    rotate all 192 dims (partial ignored)
 *   M26_NAIVE_ROPE_SINGLE_THETA  SWA's 1e4 θ everywhere (dual-θ ignored)
 */

#include "include/mimo26_attn_device.cuh"

namespace {

__global__ void rope_kernel(double theta, double partial_rotary_factor,
                            const float* x, float* y, const int64_t* pos,
                            int T, int H, int d, uint32_t naive) {
  const int rot_dim = (int)(partial_rotary_factor * (double)d + 1e-6); /* 64 of 192 */
  const int half = rot_dim / 2;
  const int pairs = half;
  /* one thread per (t, h, pair) */
  const int total = T * H * pairs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int j = idx % pairs;
    const int t = (idx / pairs) / H;
    const int h = (idx / pairs) % H;
    int64_t p = pos[t];
    if (naive & M26_NAIVE_ROPE_TRUNC_TABLE) p = p % 32768; /* T19 bug */
    const double th = (naive & M26_NAIVE_ROPE_SINGLE_THETA) ? 1e4 : theta;
    /* inv_freq[j] = theta^(-2j/rot_dim): the denominator is the ROTARY width
     * rd = rot_dim (64 of 192), NOT the full d — mirrors `src/rope.rs:36-41`
     * `inv_freq_f64` (`theta^(-2j/rd)`). Dividing by the full d put every
     * rotated pair's angle at the wrong frequency (O(1) misrotation). */
    const float inv = (float)exp(-2.0 * (double)j / (double)rot_dim * log(th));
    const float ang = (float)p * inv; /* FP32 angle, computed on the fly */
    const float c = cosf(ang);
    const float s = sinf(ang);
    float* row = y + ((int64_t)t * H + h) * d;
    const float* xrow = x + ((int64_t)t * H + h) * d;
    const int i0 = j;
    const int i1 = j + half;
    const float a = xrow[i0];
    const float b = xrow[i1];
    row[i0] = a * c - b * s;
    row[i1] = b * c + a * s;
  }
}

__global__ void rope_fullwidth_kernel(double theta, const float* x, float* y,
                                      const int64_t* pos, int T, int H, int d) {
  /* T7 bug oracle: treat partial_rotary_factor as 1.0 (every channel pairs) */
  const int total = T * H * (d / 2);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int j = idx % (d / 2);
    const int t = (idx / (d / 2)) / H;
    const int h = (idx / (d / 2)) % H;
    const float inv = (float)exp(-2.0 * (double)j / (double)d * log(theta));
    const float ang = (float)pos[t] * inv;
    const float c = cosf(ang);
    const float s = sinf(ang);
    float* row = y + ((int64_t)t * H + h) * d;
    const float* xrow = x + ((int64_t)t * H + h) * d;
    const float a = xrow[j];
    const float b = xrow[j + d / 2];
    row[j] = a * c - b * s;
    row[j + d / 2] = b * c + a * s;
  }
}

__global__ void copy_kernel(const float* x, float* y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * (size_t)blockDim.x) {
    y[i] = x[i];
  }
}

} /* namespace */

extern "C" cudaError_t m26_rope_apply(
    double theta, double partial_rotary_factor,
    const float* x, float* y, const int64_t* pos,
    int32_t T, int32_t H, int32_t d, uint32_t naive, m26_stream_t stream) {
  const size_t n = (size_t)T * H * d;
  cudaError_t err;
  if (y != x) {
    const int blocks = n < (1u << 22) ? (int)((n + 255) / 256) : 4096;
    copy_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(x, y, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  /* pass-through channels beyond rot_dim: copy_kernel copied them already
   * (and the aliasing case never touches them below) */
  const int rot_dim = (int)(partial_rotary_factor * (double)d + 1e-6);
  const int total = T * H * (rot_dim / 2);
  const int blocks = total < 4096 ? (total ? total : 1) : 4096;
  if (naive & M26_NAIVE_ROPE_FULL_WIDTH) {
    rope_fullwidth_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        theta, x, y, pos, T, H, d);
  } else {
    rope_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        theta, partial_rotary_factor, x, y, pos, T, H, d, naive);
  }
  (void)n;
  return cudaGetLastError();
}
