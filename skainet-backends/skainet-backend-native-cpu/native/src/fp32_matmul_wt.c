#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Native `x · Wᵀ` for a dense row-major `[out, in]` weight (SKEEP-007 §T0.2/§T1).
 *
 *   out[i, o] = Σ_p in[i, p] · w[o, p]
 *
 * Every output is one contiguous dot product over the weight's own row, so no
 * transposed copy of the weight is ever made — the shape a Linear layer stores
 * its weight in is the shape this kernel wants. Four weight rows per tile share
 * one pass over the activation row; each output accumulates its products in
 * ascending p in a single accumulator, so the result is bit-identical to the
 * transpose-then-GEMM path and to the Kotlin `denseRowDotTiled` reference.
 *
 * Threads over output rows (disjoint out[] slices) when output_dim is large
 * enough, through skainet_row_threads.h — on Emscripten that is the pthread
 * pool when the page is cross-origin isolated and a plain call otherwise.
 */
typedef struct {
    const float* in;        /* in + in_offset, m rows of k */
    const float* w;         /* w + w_offset, n rows of k */
    float* out;             /* out + out_offset, m rows of n */
    int32_t m, n, k;
} skainet_fp32_wt_ctx;

static inline float skainet_dot_f32(const float* SKAINET_RESTRICT a, const float* SKAINET_RESTRICT b, int32_t k) {
    float s = 0.0f;
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    int32_t p = 0;
    for (; p + 4 <= k; p += 4) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(a + p), wasm_v128_load(b + p)));
    }
    s = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1)
      + wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
    for (; p < k; ++p) s += a[p] * b[p];
#elif defined(SKAINET_HAVE_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int32_t p = 0;
    for (; p + 4 <= k; p += 4) acc = vfmaq_f32(acc, vld1q_f32(a + p), vld1q_f32(b + p));
    s = skainet_neon_hadd_f32(acc);
    for (; p < k; ++p) s += a[p] * b[p];
#else
    for (int32_t p = 0; p < k; ++p) s += a[p] * b[p];
#endif
    return s;
}

static void skainet_fp32_wt_rows(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_fp32_wt_ctx* c = (const skainet_fp32_wt_ctx*) vctx;
    for (int32_t i = 0; i < c->m; ++i) {
        const float* in_row = c->in + (size_t) i * c->k;
        float* out_row = c->out + (size_t) i * c->n;
        for (int32_t o = o_start; o < o_end; ++o) {
            out_row[o] = skainet_dot_f32(in_row, c->w + (size_t) o * c->k, c->k);
        }
    }
}

SKAINET_API void skainet_fp32_matmul_wt(
    const float* in, int32_t in_offset,
    const float* w, int32_t w_offset,
    float* out, int32_t out_offset,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;
    if (k <= 0) {
        for (int32_t i = 0; i < (int32_t) ((size_t) m * n); ++i) out[out_offset + i] = 0.0f;
        return;
    }
    skainet_fp32_wt_ctx ctx = { in + in_offset, w + w_offset, out + out_offset, m, n, k };
    skainet_run_rows(skainet_fp32_wt_rows, &ctx, n);
}
