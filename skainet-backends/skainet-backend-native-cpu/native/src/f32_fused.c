#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Fused dense FP32 row kernels: softmax, layer norm, RMS norm and scaled dot-product attention
 * (SKEEP-007 resident activations, step 3). Each is one pass over its rows with the intermediates
 * in registers or a per-worker scratch, where the ops-layer decomposition runs 3 to 12 allocating
 * passes (a layer norm is two reductions, two broadcasts, a square, a sqrt and a divide; attention
 * is a `[seqQ, seqKV]` score matrix, a mask add, a softmax and a second product). They exist for
 * the wasm tier first — Kotlin/Wasm has no SIMD and no shared-memory threads — and are compiled into
 * every native build, so the JVM binds the same symbols through FFM and both tiers run one code
 * base for these ops.
 *
 * Contracts (all operands contiguous, row-major):
 *
 *   skainet_f32_softmax_rows(in, out, rows, cols, log_form)
 *     out[r, :] = softmax(in[r, :]) — max-subtracted, exp, divided by the row sum; with log_form != 0
 *     out[r, :] = in[r, :] - (log Σ exp(in[r, :] - max) + max).
 *
 *   skainet_f32_layer_norm(in, gamma, beta, out, rows, cols, eps)
 *     out[r, c] = (in[r, c] - mean_r) / sqrt(var_r + eps) · gamma[c] + beta[c], with mean and the
 *     biased variance over the row; gamma and/or beta may be NULL (the affine is skipped).
 *
 *   skainet_f32_rms_norm(in, gamma, out, rows, cols, eps)
 *     out[r, c] = in[r, c] / sqrt(mean(in[r, :]²) + eps) · gamma[c]; gamma may be NULL.
 *
 *   skainet_f32_sdpa(q, k, v, out, batch, heads, kv_heads, seq_q, seq_kv, head_dim, scale,
 *                    causal, window, right_context, lengths, mask, mask_b, mask_h, mask_q, mask_k)
 *     out = softmax(q · kᵀ · scale + rules + mask) · v, the ops-layer `scaledDotProductAttention`:
 *     q is [batch, heads, seq_q, head_dim], k and v are [batch, kv_heads, seq_kv, head_dim] with
 *     query head h reading K/V head h / (heads / kv_heads) (grouped-query attention), out is
 *     [batch, heads, seq_q, head_dim]. `scale` is applied as given (the caller resolves the
 *     1/sqrt(head_dim) default). The structural rules are the ops layer's AttentionMask: a query at
 *     absolute position absQ = seq_kv - seq_q + qi attends to keys ki with
 *       ki <= absQ                             when causal != 0,
 *       absQ - window + 1 <= ki <= absQ + right_context   when window > 0,
 *       ki < lengths[b]                        when lengths != NULL ([batch] int32),
 *     intersected; a key outside the set scores -1e30 (finite, so a fully masked row is uniform)
 *     and its score is never computed. `mask` (or NULL) is a dense additive mask broadcast
 *     trailing-aligned over [batch | 1, heads | kv_heads | 1, seq_q | 1, seq_kv | 1] as the
 *     mask_* dims say, added to every score. Scores and probabilities are float; the dot products
 *     accumulate in four lanes where a SIMD body exists, the probability-weighted sum of v
 *     accumulates per output element in ascending key order (the reference's order).
 *
 * The exp calls go through skainet_expf (skainet_simd.h): a plain scalar libm call per element,
 * never a libmvec vector variant the shared library would fail to resolve.
 *
 * Threading: rows (a query row of one head for attention) go to the shared row pool
 * (skainet_row_threads.h), so a 12-head × 150-token attention threads on the mt sidecar and the JVM
 * pool, and a decode step with a handful of rows stays on the caller. Attention needs a
 * `seq_kv`-float scratch per worker; it is heap-allocated once per worker range.
 */

#if defined(SKAINET_HAVE_WASM_SIMD128)
#define SKAINET_FZ_VEC v128_t
#define SKAINET_FZ_LOAD(p) wasm_v128_load(p)
#define SKAINET_FZ_STORE(p, v) wasm_v128_store((p), (v))
#define SKAINET_FZ_SPLAT(x) wasm_f32x4_splat(x)
#define SKAINET_FZ_ADD wasm_f32x4_add
#define SKAINET_FZ_SUB wasm_f32x4_sub
#define SKAINET_FZ_MUL wasm_f32x4_mul
#define SKAINET_FZ_MAX wasm_f32x4_max
#define SKAINET_FZ_HADD skainet_wasm_hadd_f32
#define SKAINET_FZ_HAVE_VEC 1
static inline float skainet_fz_hmax(v128_t v) {
    const float a = wasm_f32x4_extract_lane(v, 0), b = wasm_f32x4_extract_lane(v, 1);
    const float c = wasm_f32x4_extract_lane(v, 2), d = wasm_f32x4_extract_lane(v, 3);
    const float ab = a > b ? a : b, cd = c > d ? c : d;
    return ab > cd ? ab : cd;
}
#elif defined(SKAINET_HAVE_NEON)
#define SKAINET_FZ_VEC float32x4_t
#define SKAINET_FZ_LOAD(p) vld1q_f32(p)
#define SKAINET_FZ_STORE(p, v) vst1q_f32((p), (v))
#define SKAINET_FZ_SPLAT(x) vdupq_n_f32(x)
#define SKAINET_FZ_ADD vaddq_f32
#define SKAINET_FZ_SUB vsubq_f32
#define SKAINET_FZ_MUL vmulq_f32
#define SKAINET_FZ_MAX vmaxq_f32
#define SKAINET_FZ_HADD skainet_neon_hadd_f32
#define SKAINET_FZ_HAVE_VEC 1
static inline float skainet_fz_hmax(float32x4_t v) { return vmaxvq_f32(v); }
#endif

/* The masked score of the ops layer's AttentionMask.MASKED: finite, so exp underflows to 0. */
#define SKAINET_SDPA_MASKED (-1.0e30f)

/* ------------------------------------------------------------ row helpers */

/* Σ x[i] over [0, n), four lanes then the tail. */
static inline float skainet_fz_sum(const float* SKAINET_RESTRICT x, int32_t n) {
    float s = 0.0f;
    int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
    SKAINET_FZ_VEC acc = SKAINET_FZ_SPLAT(0.0f);
    for (; i + 4 <= n; i += 4) acc = SKAINET_FZ_ADD(acc, SKAINET_FZ_LOAD(x + i));
    s = SKAINET_FZ_HADD(acc);
#endif
    for (; i < n; ++i) s += x[i];
    return s;
}

/* Σ (x[i] - c)² over [0, n). */
static inline float skainet_fz_sum_sq_centered(const float* SKAINET_RESTRICT x, int32_t n, float c) {
    float s = 0.0f;
    int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
    const SKAINET_FZ_VEC vc = SKAINET_FZ_SPLAT(c);
    SKAINET_FZ_VEC acc = SKAINET_FZ_SPLAT(0.0f);
    for (; i + 4 <= n; i += 4) {
        const SKAINET_FZ_VEC d = SKAINET_FZ_SUB(SKAINET_FZ_LOAD(x + i), vc);
        acc = SKAINET_FZ_ADD(acc, SKAINET_FZ_MUL(d, d));
    }
    s = SKAINET_FZ_HADD(acc);
#endif
    for (; i < n; ++i) { const float d = x[i] - c; s += d * d; }
    return s;
}

/* max x[i] over [0, n), n >= 1. */
static inline float skainet_fz_max(const float* SKAINET_RESTRICT x, int32_t n) {
    float m = x[0];
    int32_t i = 1;
#ifdef SKAINET_FZ_HAVE_VEC
    if (n >= 4) {
        SKAINET_FZ_VEC vm = SKAINET_FZ_LOAD(x);
        for (i = 4; i + 4 <= n; i += 4) vm = SKAINET_FZ_MAX(vm, SKAINET_FZ_LOAD(x + i));
        m = skainet_fz_hmax(vm);
    }
#endif
    for (; i < n; ++i) if (x[i] > m) m = x[i];
    return m;
}

/* Σ a[i] · b[i] over [0, n), four accumulators then the tail. */
static inline float skainet_fz_dot(const float* SKAINET_RESTRICT a, const float* SKAINET_RESTRICT b, int32_t n) {
    float s = 0.0f;
    int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
    SKAINET_FZ_VEC acc0 = SKAINET_FZ_SPLAT(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (; i + 16 <= n; i += 16) {
        acc0 = SKAINET_FZ_ADD(acc0, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(a + i), SKAINET_FZ_LOAD(b + i)));
        acc1 = SKAINET_FZ_ADD(acc1, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(a + i + 4), SKAINET_FZ_LOAD(b + i + 4)));
        acc2 = SKAINET_FZ_ADD(acc2, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(a + i + 8), SKAINET_FZ_LOAD(b + i + 8)));
        acc3 = SKAINET_FZ_ADD(acc3, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(a + i + 12), SKAINET_FZ_LOAD(b + i + 12)));
    }
    SKAINET_FZ_VEC acc = SKAINET_FZ_ADD(SKAINET_FZ_ADD(acc0, acc1), SKAINET_FZ_ADD(acc2, acc3));
    for (; i + 4 <= n; i += 4) acc = SKAINET_FZ_ADD(acc, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(a + i), SKAINET_FZ_LOAD(b + i)));
    s = SKAINET_FZ_HADD(acc);
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* out[i] = (x[i] - shift) · scale over [0, n), then · gamma[i] + beta[i] when given. */
static inline void skainet_fz_affine_row(const float* SKAINET_RESTRICT x, float shift, float scale,
                                         const float* SKAINET_RESTRICT gamma, const float* SKAINET_RESTRICT beta,
                                         float* SKAINET_RESTRICT out, int32_t n) {
    int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
    const SKAINET_FZ_VEC vshift = SKAINET_FZ_SPLAT(shift), vscale = SKAINET_FZ_SPLAT(scale);
    for (; i + 4 <= n; i += 4) {
        SKAINET_FZ_VEC y = SKAINET_FZ_MUL(SKAINET_FZ_SUB(SKAINET_FZ_LOAD(x + i), vshift), vscale);
        if (gamma) y = SKAINET_FZ_MUL(y, SKAINET_FZ_LOAD(gamma + i));
        if (beta) y = SKAINET_FZ_ADD(y, SKAINET_FZ_LOAD(beta + i));
        SKAINET_FZ_STORE(out + i, y);
    }
#endif
    for (; i < n; ++i) {
        float y = (x[i] - shift) * scale;
        if (gamma) y *= gamma[i];
        if (beta) y += beta[i];
        out[i] = y;
    }
}

/* out[i] += p · v[i] over [0, n). */
static inline void skainet_fz_axpy(float p, const float* SKAINET_RESTRICT v, float* SKAINET_RESTRICT out, int32_t n) {
    int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
    const SKAINET_FZ_VEC vp = SKAINET_FZ_SPLAT(p);
    for (; i + 4 <= n; i += 4) SKAINET_FZ_STORE(out + i, SKAINET_FZ_ADD(SKAINET_FZ_LOAD(out + i), SKAINET_FZ_MUL(vp, SKAINET_FZ_LOAD(v + i))));
#endif
    for (; i < n; ++i) out[i] += p * v[i];
}

/* Softmax of one row over [0, n) (n >= 1): exp(x - max) / Σ, or the log form. `out` may be `x` (attention runs it in place). */
static inline void skainet_fz_softmax_row(const float* x, float* out, int32_t n, int32_t log_form) {
    const float m = skainet_fz_max(x, n);
    float sum = 0.0f;
    if (log_form) {
        for (int32_t i = 0; i < n; ++i) sum += skainet_expf(x[i] - m);
        const float lse = logf(sum) + m;
        int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
        const SKAINET_FZ_VEC vl = SKAINET_FZ_SPLAT(lse);
        for (; i + 4 <= n; i += 4) SKAINET_FZ_STORE(out + i, SKAINET_FZ_SUB(SKAINET_FZ_LOAD(x + i), vl));
#endif
        for (; i < n; ++i) out[i] = x[i] - lse;
        return;
    }
    for (int32_t i = 0; i < n; ++i) { const float e = skainet_expf(x[i] - m); out[i] = e; sum += e; }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        int32_t i = 0;
#ifdef SKAINET_FZ_HAVE_VEC
        const SKAINET_FZ_VEC vi = SKAINET_FZ_SPLAT(inv);
        for (; i + 4 <= n; i += 4) SKAINET_FZ_STORE(out + i, SKAINET_FZ_MUL(SKAINET_FZ_LOAD(out + i), vi));
#endif
        for (; i < n; ++i) out[i] *= inv;
    }
}

/* ---------------------------------------------------------------- softmax */

typedef struct {
    const float* in;
    float* out;
    int32_t cols;
    int32_t log_form;
} skainet_f32_softmax_ctx;

static void skainet_f32_softmax_rows_fn(void* vctx, int32_t r0, int32_t r1) {
    const skainet_f32_softmax_ctx* c = (const skainet_f32_softmax_ctx*) vctx;
    for (int32_t r = r0; r < r1; ++r) {
        skainet_fz_softmax_row(c->in + (size_t) r * c->cols, c->out + (size_t) r * c->cols, c->cols, c->log_form);
    }
}

SKAINET_API void skainet_f32_softmax_rows(const float* in, float* out, int32_t rows, int32_t cols, int32_t log_form) {
    if (rows <= 0 || cols <= 0) return;
    skainet_f32_softmax_ctx ctx;
    ctx.in = in; ctx.out = out; ctx.cols = cols; ctx.log_form = log_form;
    skainet_run_rows(skainet_f32_softmax_rows_fn, &ctx, rows);
}

/* -------------------------------------------------------------- layer norm */

typedef struct {
    const float* in;
    const float* gamma;
    const float* beta;
    float* out;
    int32_t cols;
    float eps;
    int32_t rms;
} skainet_f32_norm_ctx;

static void skainet_f32_norm_rows_fn(void* vctx, int32_t r0, int32_t r1) {
    const skainet_f32_norm_ctx* c = (const skainet_f32_norm_ctx*) vctx;
    const int32_t n = c->cols;
    const float inv_n = 1.0f / (float) n;
    for (int32_t r = r0; r < r1; ++r) {
        const float* x = c->in + (size_t) r * n;
        float* out = c->out + (size_t) r * n;
        if (c->rms) {
            const float ms = skainet_fz_sum_sq_centered(x, n, 0.0f) * inv_n;
            skainet_fz_affine_row(x, 0.0f, 1.0f / sqrtf(ms + c->eps), c->gamma, NULL, out, n);
        } else {
            const float mean = skainet_fz_sum(x, n) * inv_n;
            const float var = skainet_fz_sum_sq_centered(x, n, mean) * inv_n;
            skainet_fz_affine_row(x, mean, 1.0f / sqrtf(var + c->eps), c->gamma, c->beta, out, n);
        }
    }
}

SKAINET_API void skainet_f32_layer_norm(const float* in, const float* gamma, const float* beta, float* out,
                                        int32_t rows, int32_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    skainet_f32_norm_ctx ctx;
    ctx.in = in; ctx.gamma = gamma; ctx.beta = beta; ctx.out = out; ctx.cols = cols; ctx.eps = eps; ctx.rms = 0;
    skainet_run_rows(skainet_f32_norm_rows_fn, &ctx, rows);
}

SKAINET_API void skainet_f32_rms_norm(const float* in, const float* gamma, float* out, int32_t rows, int32_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    skainet_f32_norm_ctx ctx;
    ctx.in = in; ctx.gamma = gamma; ctx.beta = NULL; ctx.out = out; ctx.cols = cols; ctx.eps = eps; ctx.rms = 1;
    skainet_run_rows(skainet_f32_norm_rows_fn, &ctx, rows);
}

/* --------------------------------------------------------------- attention */

typedef struct {
    const float* q;
    const float* k;
    const float* v;
    float* out;
    int32_t heads, kv_heads, seq_q, seq_kv, head_dim;
    float scale;
    int32_t causal, window, right_context;
    const int32_t* lengths;
    const float* mask;
    int32_t mask_b, mask_h, mask_q, mask_k;
} skainet_f32_sdpa_ctx;

/* One query row (b, h, qi): scores into `scores` (seq_kv floats), softmax, then Σ p · v. */
static void skainet_f32_sdpa_row(const skainet_f32_sdpa_ctx* c, int32_t b, int32_t h, int32_t qi, float* SKAINET_RESTRICT scores) {
    const int32_t seq_kv = c->seq_kv, d = c->head_dim;
    const int32_t kvh = h / (c->heads / c->kv_heads);
    const float* q = c->q + (((size_t) b * c->heads + h) * c->seq_q + qi) * d;
    const float* kbase = c->k + ((size_t) b * c->kv_heads + kvh) * seq_kv * d;
    const float* vbase = c->v + ((size_t) b * c->kv_heads + kvh) * seq_kv * d;
    float* out = c->out + (((size_t) b * c->heads + h) * c->seq_q + qi) * d;

    /* The attended key range [first, end) of this row under the structural rules. */
    const int32_t abs_q = seq_kv - c->seq_q + qi;
    int32_t first = 0, end = seq_kv;
    if (c->window > 0) {
        const int32_t f = abs_q - c->window + 1;
        if (f > first) first = f;
        const int32_t e = abs_q + c->right_context + 1;
        if (e < end) end = e;
    }
    if (c->causal && abs_q + 1 < end) end = abs_q + 1;
    if (c->lengths && c->lengths[b] < end) end = c->lengths[b];
    if (end < 0) end = 0;
    if (first > end) first = end;

    /* Scores: q · k · scale inside the range, MASKED outside; never computed outside. */
    if (first > 0 || end < seq_kv) {
        for (int32_t ki = 0; ki < first; ++ki) scores[ki] = SKAINET_SDPA_MASKED;
        for (int32_t ki = end; ki < seq_kv; ++ki) scores[ki] = SKAINET_SDPA_MASKED;
    }
    for (int32_t ki = first; ki < end; ++ki) scores[ki] = skainet_fz_dot(q, kbase + (size_t) ki * d, d) * c->scale;

    /* The dense additive mask, broadcast trailing-aligned. */
    int32_t pv_first = first, pv_end = end;
    if (c->mask) {
        const int32_t bi = c->mask_b == 1 ? 0 : b;
        const int32_t hi = c->mask_h == 1 ? 0 : (c->mask_h == c->heads ? h : kvh);
        const size_t row = ((size_t) bi * c->mask_h + hi) * c->mask_q + (c->mask_q == 1 ? 0 : qi);
        const float* m = c->mask + row * c->mask_k;
        if (c->mask_k == 1) {
            const float mv = m[0];
            for (int32_t ki = 0; ki < seq_kv; ++ki) scores[ki] += mv;
        } else {
            for (int32_t ki = 0; ki < seq_kv; ++ki) scores[ki] += m[ki];
        }
        /* A mask can raise a key the rules excluded: weigh every key below. */
        pv_first = 0; pv_end = seq_kv;
    }
    if (first >= end) {
        /* Every key masked: a uniform row, as the dense -1e30 mask yields. */
        pv_first = 0; pv_end = seq_kv;
    }

    skainet_fz_softmax_row(scores, scores, seq_kv, 0);

    /* out = Σ_ki p[ki] · v[ki, :], ascending ki per output element. Keys the rules excluded have
     * p == 0 exactly (exp underflow against a finite max), so skipping them changes nothing. */
    for (int32_t x = 0; x < d; ++x) out[x] = 0.0f;
    for (int32_t ki = pv_first; ki < pv_end; ++ki) skainet_fz_axpy(scores[ki], vbase + (size_t) ki * d, out, d);
}

static void skainet_f32_sdpa_rows_fn(void* vctx, int32_t r0, int32_t r1) {
    const skainet_f32_sdpa_ctx* c = (const skainet_f32_sdpa_ctx*) vctx;
    float* scores = (float*) malloc((size_t) c->seq_kv * sizeof(float));
    if (!scores) return;
    const int32_t per_batch = c->heads * c->seq_q;
    for (int32_t r = r0; r < r1; ++r) {
        const int32_t b = r / per_batch;
        const int32_t rem = r - b * per_batch;
        skainet_f32_sdpa_row(c, b, rem / c->seq_q, rem % c->seq_q, scores);
    }
    free(scores);
}

SKAINET_API void skainet_f32_sdpa(
    const float* q, const float* k, const float* v, float* out,
    int32_t batch, int32_t heads, int32_t kv_heads,
    int32_t seq_q, int32_t seq_kv, int32_t head_dim,
    float scale, int32_t causal, int32_t window, int32_t right_context,
    const int32_t* lengths,
    const float* mask, int32_t mask_b, int32_t mask_h, int32_t mask_q, int32_t mask_k
) {
    if (batch <= 0 || heads <= 0 || kv_heads <= 0 || seq_q <= 0 || head_dim <= 0) return;
    const int32_t rows = batch * heads * seq_q;
    if (seq_kv <= 0) {
        for (int32_t i = 0; i < rows * head_dim; ++i) out[i] = 0.0f;
        return;
    }
    skainet_f32_sdpa_ctx ctx;
    ctx.q = q; ctx.k = k; ctx.v = v; ctx.out = out;
    ctx.heads = heads; ctx.kv_heads = kv_heads; ctx.seq_q = seq_q; ctx.seq_kv = seq_kv; ctx.head_dim = head_dim;
    ctx.scale = scale; ctx.causal = causal; ctx.window = window; ctx.right_context = right_context;
    ctx.lengths = lengths;
    ctx.mask = mask; ctx.mask_b = mask_b; ctx.mask_h = mask_h; ctx.mask_q = mask_q; ctx.mask_k = mask_k;
    skainet_run_rows(skainet_f32_sdpa_rows_fn, &ctx, rows);
}
