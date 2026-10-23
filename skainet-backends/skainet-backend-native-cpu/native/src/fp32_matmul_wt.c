#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"
#include "skainet_cpu_features.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Native `x · Wᵀ` for a dense row-major `[out, in]` weight (SKEEP-007 §T0.2/§T1).
 *
 *   out[i, o] = Σ_p in[i, p] · w[o, p]
 *
 * Every output is one contiguous dot product over the weight's own row, so no
 * transposed copy of the weight is ever made — the shape a Linear layer stores
 * its weight in is the shape this kernel wants.
 *
 * Numerics: the order in which an output accumulates its products is fixed per
 * build (the portable loop, the four-lane wasm/NEON sums, or the eight-lane AVX2
 * tile below) and never depends on which thread or which row slice computes
 * it, so a threaded run, a sliced run (SKEEP-005 schedules split the weight's
 * rows across calls) and a single call are bit-identical to each other. Lanes
 * differ between builds the way they already do between NEON and wasm.
 *
 * Threads over output rows (disjoint out[] slices) when output_dim is large
 * enough, through skainet_row_threads.h — on Emscripten that is the pthread
 * pool when the page is cross-origin isolated and a plain call otherwise.
 *
 * x86-64: the shared library is built without -march (it has to run on any
 * x86-64), so the portable loop is SSE2 code. Hosts with AVX2+FMA — every
 * x86 CPU since 2013 — take skainet_fp32_wt_rows_avx2, compiled per function
 * with the avx2+fma target (MSVC emits VEX intrinsics without /arch) and
 * selected once per call by the cached cpuid probe in skainet_cpu_features.h,
 * the same per-function dispatch the Apple arm64 dotprod kernels use.
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

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__wasm__)
#define SKAINET_FP32_WT_AVX2 1
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#  define SKAINET_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#  define SKAINET_AVX2_TARGET
#endif

/*
 * One output's reduction, shared by every tile variant: the eight lanes of an
 * FMA chain over p < (k & ~7), then this fixed horizontal sum, then the scalar
 * tail in ascending p. Keeping the three steps identical in the 4×2, 4×1 and
 * 1-row variants is what makes the result independent of tile placement.
 */
SKAINET_AVX2_TARGET
static inline float skainet_avx2_finish(__m256 acc, const float* a, const float* b, int32_t k8, int32_t k) {
    __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);
    for (int32_t p = k8; p < k; ++p) s += a[p] * b[p];
    return s;
}

/* Single weight row against one activation row (the o tail of a slice). */
SKAINET_AVX2_TARGET
static inline float skainet_avx2_dot(const float* a, const float* b, int32_t k8, int32_t k) {
    __m256 acc = _mm256_setzero_ps();
    for (int32_t p = 0; p < k8; p += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    }
    return skainet_avx2_finish(acc, a, b, k8, k);
}

/*
 * Four weight rows against activation rows two at a time: per p-step six loads
 * feed eight FMAs, and the 4 × k floats of weight stay in the core's own cache
 * while every activation row streams past them — the weight is read from
 * memory once per call, not once per activation row.
 */
SKAINET_AVX2_TARGET
static void skainet_fp32_wt_rows_avx2(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_fp32_wt_ctx* c = (const skainet_fp32_wt_ctx*) vctx;
    const int32_t m = c->m, n = c->n, k = c->k;
    const int32_t k8 = k & ~7;
    int32_t o = o_start;
    for (; o + 4 <= o_end; o += 4) {
        const float* w0 = c->w + (size_t) o * k;
        const float* w1 = w0 + k;
        const float* w2 = w1 + k;
        const float* w3 = w2 + k;
        int32_t i = 0;
        for (; i + 2 <= m; i += 2) {
            const float* a0 = c->in + (size_t) i * k;
            const float* a1 = a0 + k;
            __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
            __m256 s02 = _mm256_setzero_ps(), s03 = _mm256_setzero_ps();
            __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
            __m256 s12 = _mm256_setzero_ps(), s13 = _mm256_setzero_ps();
            for (int32_t p = 0; p < k8; p += 8) {
                const __m256 x0 = _mm256_loadu_ps(a0 + p);
                const __m256 x1 = _mm256_loadu_ps(a1 + p);
                __m256 y = _mm256_loadu_ps(w0 + p);
                s00 = _mm256_fmadd_ps(x0, y, s00);
                s10 = _mm256_fmadd_ps(x1, y, s10);
                y = _mm256_loadu_ps(w1 + p);
                s01 = _mm256_fmadd_ps(x0, y, s01);
                s11 = _mm256_fmadd_ps(x1, y, s11);
                y = _mm256_loadu_ps(w2 + p);
                s02 = _mm256_fmadd_ps(x0, y, s02);
                s12 = _mm256_fmadd_ps(x1, y, s12);
                y = _mm256_loadu_ps(w3 + p);
                s03 = _mm256_fmadd_ps(x0, y, s03);
                s13 = _mm256_fmadd_ps(x1, y, s13);
            }
            float* out0 = c->out + (size_t) i * n + o;
            float* out1 = out0 + n;
            out0[0] = skainet_avx2_finish(s00, a0, w0, k8, k);
            out0[1] = skainet_avx2_finish(s01, a0, w1, k8, k);
            out0[2] = skainet_avx2_finish(s02, a0, w2, k8, k);
            out0[3] = skainet_avx2_finish(s03, a0, w3, k8, k);
            out1[0] = skainet_avx2_finish(s10, a1, w0, k8, k);
            out1[1] = skainet_avx2_finish(s11, a1, w1, k8, k);
            out1[2] = skainet_avx2_finish(s12, a1, w2, k8, k);
            out1[3] = skainet_avx2_finish(s13, a1, w3, k8, k);
        }
        if (i < m) {
            const float* a0 = c->in + (size_t) i * k;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            for (int32_t p = 0; p < k8; p += 8) {
                const __m256 x0 = _mm256_loadu_ps(a0 + p);
                s0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w0 + p), s0);
                s1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w1 + p), s1);
                s2 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w2 + p), s2);
                s3 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(w3 + p), s3);
            }
            float* out0 = c->out + (size_t) i * n + o;
            out0[0] = skainet_avx2_finish(s0, a0, w0, k8, k);
            out0[1] = skainet_avx2_finish(s1, a0, w1, k8, k);
            out0[2] = skainet_avx2_finish(s2, a0, w2, k8, k);
            out0[3] = skainet_avx2_finish(s3, a0, w3, k8, k);
        }
    }
    for (; o < o_end; ++o) {
        const float* wo = c->w + (size_t) o * k;
        for (int32_t i = 0; i < m; ++i) {
            c->out[(size_t) i * n + o] = skainet_avx2_dot(c->in + (size_t) i * k, wo, k8, k);
        }
    }
}
#endif /* x86-64 */

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
    skainet_row_range_fn rows = skainet_fp32_wt_rows;
#ifdef SKAINET_FP32_WT_AVX2
    if (skainet_cpu_has_avx2_fma()) rows = skainet_fp32_wt_rows_avx2;
#endif
    skainet_run_rows(rows, &ctx, n);
}
