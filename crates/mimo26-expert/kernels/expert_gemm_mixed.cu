/* R18c phase-wise mixed-M B2 decode dispatch — production translation unit.
 * The kernel, the launch/dispatch helpers, the validation, and the extern "C"
 * entry `m26x_expert_ffn_mixed_v2` all live in mixed_dispatch.cuh (verbatim
 * from the R18c test adapter, A1 R3: no body edits). This TU instantiates them
 * into a linkable object beside the frozen per-M expert_gemm.cu.
 */
#include "mixed_dispatch.cuh"
