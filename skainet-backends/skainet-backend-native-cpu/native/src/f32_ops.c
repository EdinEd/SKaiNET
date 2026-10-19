#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Dense FP32 elementwise, activation and reduction kernels (SKEEP-007 resident activations).
 *
 * These exist for the wasm tier first: Kotlin/Wasm has no SIMD and no shared-memory threads, so
 * everything between two matmuls of a browser forward pass — bias add, GELU, residual add, the
 * reductions of a norm — ran as scalar Kotlin on one thread while the matmuls ran in the SIMD128
 * sidecar. With these exports the activations can stay in the sidecar's linear memory from one
 * op to the next (`Storage.WasmLinear`) and only cross the boundary when something outside the
 * sidecar needs them. They are compiled into every native build as well, so a JVM/Android/K-N
 * binding is a few lines when wanted; the JVM has Vector API bodies for all of them today.
 *
 * Contracts (all operands contiguous):
 *
 *   skainet_f32_binary(op, a, a_mode, b, b_mode, out, n, inner)
 *     out[i] = a[ia] (op) b[ib] for i in [0, n), where an operand's index follows its mode:
 *       SKAINET_F32_FULL   ia = i             (same shape as the output)
 *       SKAINET_F32_SCALAR ia = 0             (a single value broadcast over the output)
 *       SKAINET_F32_BIAS   ia = i % inner     (a vector over the last dimension, `inner` long)
 *     — the three broadcast shapes the common ops layer takes on its fast path (same shape,
 *     scalar, last-dim "bias"); anything else is the caller's generic path.
 *
 *   skainet_f32_unary(op, in, out, n)
 *     out[i] = f(in[i]) with f one of relu, sigmoid, silu, gelu (tanh form), gelu (erf form),
 *     erf, tanh. The formulas are those of the Kotlin reference ops, so results agree to the
 *     last-ulp differences between libm and the Kotlin math functions. The transcendental ones are
 *     scalar libm calls (no vector libm in wasm or portable C); relu has SIMD bodies.
 *
 *   skainet_f32_reduce_all(in, n, mean, out)      *out = Σ in[i], divided by n when mean != 0
 *   skainet_f32_reduce_axis(in, outer, red, inner, mean, out)
 *     out[o * inner + x] = Σ_k in[(o * red + k) * inner + x]   (/ red when mean != 0)
 *     — the (outer, reduced, inner) decomposition of a reduction over one axis.
 *
 * Threading: the work is split into fixed-size chunks of output elements (rows for the axis
 * reduction) and handed to the shared row pool when there are enough of them
 * (skainet_row_threads.h), so a 1M-element op uses the mt sidecar's pthreads and a 768-wide bias
 * add stays on the caller. Per-element results do not depend on the partition. The full
 * reduction accumulates in four lanes, so its rounding differs from a sequential sum in the last
 * bits, as the JVM Vector API path's does.
 */

#define SKAINET_F32_FULL   0
#define SKAINET_F32_SCALAR 1
#define SKAINET_F32_BIAS   2

#define SKAINET_F32_ADD 0
#define SKAINET_F32_SUB 1
#define SKAINET_F32_MUL 2
#define SKAINET_F32_DIV 3

#define SKAINET_F32_RELU       0
#define SKAINET_F32_SIGMOID    1
#define SKAINET_F32_SILU       2
#define SKAINET_F32_GELU_TANH  3
#define SKAINET_F32_GELU_EXACT 4
#define SKAINET_F32_ERF        5
#define SKAINET_F32_TANH       6

/* Output elements per pool "row": the pool threads at 512 rows, so 1024 → 512K elements. */
#define SKAINET_F32_BINARY_CHUNK 1024
/* The transcendental unaries cost ~20 ns a lane in scalar libm; thread earlier. */
#define SKAINET_F32_UNARY_CHUNK 256

/* ---------------------------------------------------------------- binary */

typedef struct {
    const float* a;
    const float* b;
    float* out;
    int32_t n, inner;
    int32_t op, a_mode, b_mode;
} skainet_f32_binary_ctx;

/* The two SIMD shapes: lane-wise a[i] (op) b[i], and lane-wise a[i] (op) splat. */
#if defined(SKAINET_HAVE_WASM_SIMD128)
#define SKAINET_F32_VEC v128_t
#define SKAINET_F32_LOAD(p) wasm_v128_load(p)
#define SKAINET_F32_STORE(p, v) wasm_v128_store((p), (v))
#define SKAINET_F32_SPLAT(x) wasm_f32x4_splat(x)
#define SKAINET_F32_VADD wasm_f32x4_add
#define SKAINET_F32_VSUB wasm_f32x4_sub
#define SKAINET_F32_VMUL wasm_f32x4_mul
#define SKAINET_F32_VDIV wasm_f32x4_div
#define SKAINET_F32_VMAX wasm_f32x4_max
#define SKAINET_F32_HAVE_VEC 1
#elif defined(SKAINET_HAVE_NEON)
#define SKAINET_F32_VEC float32x4_t
#define SKAINET_F32_LOAD(p) vld1q_f32(p)
#define SKAINET_F32_STORE(p, v) vst1q_f32((p), (v))
#define SKAINET_F32_SPLAT(x) vdupq_n_f32(x)
#define SKAINET_F32_VADD vaddq_f32
#define SKAINET_F32_VSUB vsubq_f32
#define SKAINET_F32_VMUL vmulq_f32
#define SKAINET_F32_VDIV vdivq_f32
#define SKAINET_F32_VMAX vmaxq_f32
#define SKAINET_F32_HAVE_VEC 1
#endif

static inline float skainet_f32_apply(int32_t op, float x, float y) {
    switch (op) {
        case SKAINET_F32_ADD: return x + y;
        case SKAINET_F32_SUB: return x - y;
        case SKAINET_F32_MUL: return x * y;
        default:              return x / y;
    }
}

#ifdef SKAINET_F32_HAVE_VEC
static inline SKAINET_F32_VEC skainet_f32_vapply(int32_t op, SKAINET_F32_VEC x, SKAINET_F32_VEC y) {
    switch (op) {
        case SKAINET_F32_ADD: return SKAINET_F32_VADD(x, y);
        case SKAINET_F32_SUB: return SKAINET_F32_VSUB(x, y);
        case SKAINET_F32_MUL: return SKAINET_F32_VMUL(x, y);
        default:              return SKAINET_F32_VDIV(x, y);
    }
}
#endif

/* out[i0..i1) = a[i0..i1) (op) b[i0..i1): both full. */
static void skainet_f32_binary_full(int32_t op, const float* SKAINET_RESTRICT a, const float* SKAINET_RESTRICT b,
                                    float* SKAINET_RESTRICT out, int32_t i0, int32_t i1) {
    int32_t i = i0;
#ifdef SKAINET_F32_HAVE_VEC
    for (; i + 4 <= i1; i += 4) {
        SKAINET_F32_STORE(out + i, skainet_f32_vapply(op, SKAINET_F32_LOAD(a + i), SKAINET_F32_LOAD(b + i)));
    }
#endif
    for (; i < i1; ++i) out[i] = skainet_f32_apply(op, a[i], b[i]);
}

/* out[i0..i1) = a[i0..i1) (op) s, or s (op) a[i0..i1) when `swap`. */
static void skainet_f32_binary_scalar(int32_t op, const float* SKAINET_RESTRICT a, float s, int swap,
                                      float* SKAINET_RESTRICT out, int32_t i0, int32_t i1) {
    int32_t i = i0;
#ifdef SKAINET_F32_HAVE_VEC
    const SKAINET_F32_VEC vs = SKAINET_F32_SPLAT(s);
    for (; i + 4 <= i1; i += 4) {
        const SKAINET_F32_VEC va = SKAINET_F32_LOAD(a + i);
        SKAINET_F32_STORE(out + i, swap ? skainet_f32_vapply(op, vs, va) : skainet_f32_vapply(op, va, vs));
    }
#endif
    for (; i < i1; ++i) out[i] = swap ? skainet_f32_apply(op, s, a[i]) : skainet_f32_apply(op, a[i], s);
}

static void skainet_f32_binary_chunks(void* vctx, int32_t c0, int32_t c1) {
    const skainet_f32_binary_ctx* c = (const skainet_f32_binary_ctx*) vctx;
    const int32_t i0 = c0 * SKAINET_F32_BINARY_CHUNK;
    const int32_t i1 = (c1 * SKAINET_F32_BINARY_CHUNK < c->n) ? c1 * SKAINET_F32_BINARY_CHUNK : c->n;
    if (c->a_mode == SKAINET_F32_FULL && c->b_mode == SKAINET_F32_FULL) {
        skainet_f32_binary_full(c->op, c->a, c->b, c->out, i0, i1);
        return;
    }
    if (c->a_mode == SKAINET_F32_FULL && c->b_mode == SKAINET_F32_SCALAR) {
        skainet_f32_binary_scalar(c->op, c->a, c->b[0], 0, c->out, i0, i1);
        return;
    }
    if (c->a_mode == SKAINET_F32_SCALAR && c->b_mode == SKAINET_F32_FULL) {
        skainet_f32_binary_scalar(c->op, c->b, c->a[0], 1, c->out, i0, i1);
        return;
    }
    if (c->a_mode == SKAINET_F32_FULL && c->b_mode == SKAINET_F32_BIAS) {
        /* Walk the chunk row by row of the last dimension so the bias is read contiguously. */
        int32_t i = i0;
        while (i < i1) {
            const int32_t x = i % c->inner;
            const int32_t run = (c->inner - x < i1 - i) ? c->inner - x : i1 - i;
            skainet_f32_binary_full(c->op, c->a + i, c->b + x, c->out + i, 0, run);
            i += run;
        }
        return;
    }
    if (c->a_mode == SKAINET_F32_BIAS && c->b_mode == SKAINET_F32_FULL) {
        int32_t i = i0;
        while (i < i1) {
            const int32_t x = i % c->inner;
            const int32_t run = (c->inner - x < i1 - i) ? c->inner - x : i1 - i;
            skainet_f32_binary_full(c->op, c->a + x, c->b + i, c->out + i, 0, run);
            i += run;
        }
        return;
    }
    /* Remaining mode pairs (scalar×scalar, bias×bias, scalar×bias): index each side by its mode. */
    for (int32_t i = i0; i < i1; ++i) {
        const int32_t ia = c->a_mode == SKAINET_F32_FULL ? i : (c->a_mode == SKAINET_F32_SCALAR ? 0 : i % c->inner);
        const int32_t ib = c->b_mode == SKAINET_F32_FULL ? i : (c->b_mode == SKAINET_F32_SCALAR ? 0 : i % c->inner);
        c->out[i] = skainet_f32_apply(c->op, c->a[ia], c->b[ib]);
    }
}

SKAINET_API void skainet_f32_binary(
    int32_t op,
    const float* a, int32_t a_mode,
    const float* b, int32_t b_mode,
    float* out, int32_t n, int32_t inner
) {
    if (n <= 0) return;
    skainet_f32_binary_ctx ctx;
    ctx.a = a; ctx.b = b; ctx.out = out;
    ctx.n = n;
    ctx.inner = inner > 0 ? inner : 1;
    ctx.op = op; ctx.a_mode = a_mode; ctx.b_mode = b_mode;
    skainet_run_rows(skainet_f32_binary_chunks, &ctx, (n + SKAINET_F32_BINARY_CHUNK - 1) / SKAINET_F32_BINARY_CHUNK);
}

/* ----------------------------------------------------------------- unary */

typedef struct {
    const float* in;
    float* out;
    int32_t n;
    int32_t op;
} skainet_f32_unary_ctx;

static inline float skainet_f32_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void skainet_f32_unary_chunks(void* vctx, int32_t c0, int32_t c1) {
    const skainet_f32_unary_ctx* c = (const skainet_f32_unary_ctx*) vctx;
    const float* SKAINET_RESTRICT in = c->in;
    float* SKAINET_RESTRICT out = c->out;
    const int32_t i0 = c0 * SKAINET_F32_UNARY_CHUNK;
    const int32_t i1 = (c1 * SKAINET_F32_UNARY_CHUNK < c->n) ? c1 * SKAINET_F32_UNARY_CHUNK : c->n;
    switch (c->op) {
        case SKAINET_F32_RELU: {
            int32_t i = i0;
#ifdef SKAINET_F32_HAVE_VEC
            const SKAINET_F32_VEC zero = SKAINET_F32_SPLAT(0.0f);
            for (; i + 4 <= i1; i += 4) SKAINET_F32_STORE(out + i, SKAINET_F32_VMAX(SKAINET_F32_LOAD(in + i), zero));
#endif
            for (; i < i1; ++i) out[i] = in[i] < 0.0f ? 0.0f : in[i];
            break;
        }
        case SKAINET_F32_SIGMOID:
            for (int32_t i = i0; i < i1; ++i) out[i] = skainet_f32_sigmoid(in[i]);
            break;
        case SKAINET_F32_SILU:
            for (int32_t i = i0; i < i1; ++i) out[i] = in[i] * skainet_f32_sigmoid(in[i]);
            break;
        case SKAINET_F32_GELU_TANH:
            for (int32_t i = i0; i < i1; ++i) {
                const float x = in[i];
                const float inner = x + 0.044715f * (x * x * x);
                out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * inner));
            }
            break;
        case SKAINET_F32_GELU_EXACT:
            for (int32_t i = i0; i < i1; ++i) {
                const double x = (double) in[i];
                out[i] = (float) (0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
            }
            break;
        case SKAINET_F32_ERF:
            for (int32_t i = i0; i < i1; ++i) out[i] = (float) erf((double) in[i]);
            break;
        default: /* SKAINET_F32_TANH */
            for (int32_t i = i0; i < i1; ++i) out[i] = tanhf(in[i]);
            break;
    }
}

SKAINET_API void skainet_f32_unary(int32_t op, const float* in, float* out, int32_t n) {
    if (n <= 0) return;
    skainet_f32_unary_ctx ctx;
    ctx.in = in; ctx.out = out; ctx.n = n; ctx.op = op;
    skainet_run_rows(skainet_f32_unary_chunks, &ctx, (n + SKAINET_F32_UNARY_CHUNK - 1) / SKAINET_F32_UNARY_CHUNK);
}

/* ------------------------------------------------------------- reductions */

SKAINET_API void skainet_f32_reduce_all(const float* in, int32_t n, int32_t mean, float* out) {
    float s = 0.0f;
    if (n > 0) {
        int32_t i = 0;
#ifdef SKAINET_F32_HAVE_VEC
        SKAINET_F32_VEC acc0 = SKAINET_F32_SPLAT(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (; i + 16 <= n; i += 16) {
            acc0 = SKAINET_F32_VADD(acc0, SKAINET_F32_LOAD(in + i));
            acc1 = SKAINET_F32_VADD(acc1, SKAINET_F32_LOAD(in + i + 4));
            acc2 = SKAINET_F32_VADD(acc2, SKAINET_F32_LOAD(in + i + 8));
            acc3 = SKAINET_F32_VADD(acc3, SKAINET_F32_LOAD(in + i + 12));
        }
        const SKAINET_F32_VEC acc = SKAINET_F32_VADD(SKAINET_F32_VADD(acc0, acc1), SKAINET_F32_VADD(acc2, acc3));
#if defined(SKAINET_HAVE_WASM_SIMD128)
        s = skainet_wasm_hadd_f32(acc);
#else
        s = skainet_neon_hadd_f32(acc);
#endif
#endif
        for (; i < n; ++i) s += in[i];
    }
    *out = (mean && n > 0) ? s / (float) n : s;
}

typedef struct {
    const float* in;
    float* out;
    int32_t red, inner;
    int32_t mean;
} skainet_f32_reduce_axis_ctx;

static void skainet_f32_reduce_axis_rows(void* vctx, int32_t o0, int32_t o1) {
    const skainet_f32_reduce_axis_ctx* c = (const skainet_f32_reduce_axis_ctx*) vctx;
    const int32_t inner = c->inner;
    const int32_t red = c->red;
    for (int32_t o = o0; o < o1; ++o) {
        float* SKAINET_RESTRICT out = c->out + (size_t) o * inner;
        const float* SKAINET_RESTRICT base = c->in + (size_t) o * red * inner;
        for (int32_t x = 0; x < inner; ++x) out[x] = 0.0f;
        for (int32_t k = 0; k < red; ++k) {
            const float* SKAINET_RESTRICT row = base + (size_t) k * inner;
            skainet_f32_binary_full(SKAINET_F32_ADD, out, row, out, 0, inner);
        }
        if (c->mean && red > 0) {
            const float inv = 1.0f / (float) red;
            skainet_f32_binary_scalar(SKAINET_F32_MUL, out, inv, 0, out, 0, inner);
        }
    }
}

SKAINET_API void skainet_f32_reduce_axis(const float* in, int32_t outer, int32_t red, int32_t inner, int32_t mean, float* out) {
    if (outer <= 0 || inner <= 0) return;
    if (red <= 0) {
        for (int32_t i = 0; i < outer * inner; ++i) out[i] = 0.0f;
        return;
    }
    skainet_f32_reduce_axis_ctx ctx;
    ctx.in = in; ctx.out = out; ctx.red = red; ctx.inner = inner; ctx.mean = mean;
    skainet_run_rows(skainet_f32_reduce_axis_rows, &ctx, outer);
}
