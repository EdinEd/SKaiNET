#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Native row-major SGEMM matching the
 * sk.ainet.backend.api.kernel.Fp32MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 * Strides are in floats (not bytes); for a contiguous parent matrix
 * `aStride == k`, `bStride == n`, `cStride == n`. Sub-block scenarios
 * pass larger strides and the corresponding offsets.
 *
 * Portable and NEON bodies: iteration order is i-p-j (outer product into
 * rows of C). The inner loop is `c[j] += a_ip * b[j]` over a contiguous run
 * of `n` floats for both b's row and c's row — auto-vectorizes cleanly under
 * -O3 -ffast-math into vfmadd231ps / fmla.
 *
 * SIMD128 body (the Emscripten sidecar): a 4×8 register-blocked micro-kernel.
 * Eight f32x4 accumulators hold a 4-row × 8-column tile of C for a whole
 * k-block, so per p the loop issues two B loads, four A broadcasts and eight
 * multiply-adds instead of a load/multiply-add/store triple per four
 * outputs. k is blocked so the B panel a tile streams (8 columns × K_BLOCK
 * rows) stays in L1 while the tile walks down the rows of A. Every C element
 * still accumulates its products in ascending p — the partial sum is carried
 * in C between k-blocks — so the result is bit-identical to the portable
 * body (SIMD128 has no fused multiply-add; the portable body is compiled
 * without FMA contraction on that target too).
 *
 * Threading (#1195 pattern, skainet_row_threads.h): the output is split over
 * the longer of its two sides — rows of C when m ≥ n, columns otherwise —
 * into disjoint ranges; workers never share a C element, so the partition
 * does not change any accumulation order. On Emscripten the pool is the
 * pthread pool of the mt sidecar; the st sidecar and MSVC run on the caller.
 *
 * Caller contract:
 *  - C is FULLY OVERWRITTEN in the m×n block (zero-then-accumulate).
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; the Kotlin wrapper rejects
 *    them. The C kernel still treats negatives as no-op (via the
 *    `<=` loop bounds) defensively.
 */

typedef struct {
    const float* a;      /* a + a_offset */
    const float* b;      /* b + b_offset */
    float* c;            /* c + c_offset */
    int32_t a_stride, b_stride, c_stride;
    int32_t m, n, k;
    int split_rows;      /* 1: worker ranges are rows of C; 0: columns */
} skainet_fp32_gemm_ctx;

#ifdef SKAINET_HAVE_WASM_SIMD128

#define SKAINET_FP32_K_BLOCK 256

/* C[i0..i0+4, j0..j0+8] over p in [p0, p1): eight accumulators, carried in C. */
static inline void skainet_fp32_tile_4x8(
    const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t j0, int32_t p0, int32_t p1
) {
    const float* a0 = c->a + (size_t) i0 * c->a_stride;
    const float* a1 = a0 + c->a_stride;
    const float* a2 = a1 + c->a_stride;
    const float* a3 = a2 + c->a_stride;
    float* c0 = c->c + (size_t) i0 * c->c_stride + j0;
    float* c1 = c0 + c->c_stride;
    float* c2 = c1 + c->c_stride;
    float* c3 = c2 + c->c_stride;
    v128_t acc00 = wasm_v128_load(c0), acc01 = wasm_v128_load(c0 + 4);
    v128_t acc10 = wasm_v128_load(c1), acc11 = wasm_v128_load(c1 + 4);
    v128_t acc20 = wasm_v128_load(c2), acc21 = wasm_v128_load(c2 + 4);
    v128_t acc30 = wasm_v128_load(c3), acc31 = wasm_v128_load(c3 + 4);
    const float* b_row = c->b + (size_t) p0 * c->b_stride + j0;
    for (int32_t p = p0; p < p1; ++p, b_row += c->b_stride) {
        const v128_t b0 = wasm_v128_load(b_row);
        const v128_t b1 = wasm_v128_load(b_row + 4);
        const v128_t va0 = wasm_f32x4_splat(a0[p]);
        const v128_t va1 = wasm_f32x4_splat(a1[p]);
        const v128_t va2 = wasm_f32x4_splat(a2[p]);
        const v128_t va3 = wasm_f32x4_splat(a3[p]);
        acc00 = wasm_f32x4_add(acc00, wasm_f32x4_mul(va0, b0));
        acc01 = wasm_f32x4_add(acc01, wasm_f32x4_mul(va0, b1));
        acc10 = wasm_f32x4_add(acc10, wasm_f32x4_mul(va1, b0));
        acc11 = wasm_f32x4_add(acc11, wasm_f32x4_mul(va1, b1));
        acc20 = wasm_f32x4_add(acc20, wasm_f32x4_mul(va2, b0));
        acc21 = wasm_f32x4_add(acc21, wasm_f32x4_mul(va2, b1));
        acc30 = wasm_f32x4_add(acc30, wasm_f32x4_mul(va3, b0));
        acc31 = wasm_f32x4_add(acc31, wasm_f32x4_mul(va3, b1));
    }
    wasm_v128_store(c0, acc00); wasm_v128_store(c0 + 4, acc01);
    wasm_v128_store(c1, acc10); wasm_v128_store(c1 + 4, acc11);
    wasm_v128_store(c2, acc20); wasm_v128_store(c2 + 4, acc21);
    wasm_v128_store(c3, acc30); wasm_v128_store(c3 + 4, acc31);
}

/* Edge: rows [i0, i1) × columns [j0, j1) over p in [p0, p1), i-p-j with 4-lane runs and a scalar tail. */
static inline void skainet_fp32_edge(
    const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j0, int32_t j1, int32_t p0, int32_t p1
) {
    for (int32_t i = i0; i < i1; ++i) {
        const float* a_row = c->a + (size_t) i * c->a_stride;
        float* c_row = c->c + (size_t) i * c->c_stride;
        const float* b_row = c->b + (size_t) p0 * c->b_stride;
        for (int32_t p = p0; p < p1; ++p, b_row += c->b_stride) {
            const float a_ip = a_row[p];
            const v128_t va = wasm_f32x4_splat(a_ip);
            int32_t j = j0;
            for (; j + 4 <= j1; j += 4) {
                wasm_v128_store(c_row + j, wasm_f32x4_add(wasm_v128_load(c_row + j),
                                                          wasm_f32x4_mul(va, wasm_v128_load(b_row + j))));
            }
            for (; j < j1; ++j) c_row[j] += a_ip * b_row[j];
        }
    }
}

/* Rows [i0, i1) × columns [j0, j1) of C, k-blocked, tiles first and edges after. */
static void skainet_fp32_block(const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j0, int32_t j1) {
    for (int32_t i = i0; i < i1; ++i) {
        float* c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t j = j0; j < j1; ++j) c_row[j] = 0.0f;
    }
    if (c->k <= 0) return;
    const int32_t i_tiles_end = i0 + ((i1 - i0) / 4) * 4;
    const int32_t j_tiles_end = j0 + ((j1 - j0) / 8) * 8;
    for (int32_t p0 = 0; p0 < c->k; p0 += SKAINET_FP32_K_BLOCK) {
        const int32_t p1 = (c->k - p0) < SKAINET_FP32_K_BLOCK ? c->k : p0 + SKAINET_FP32_K_BLOCK;
        for (int32_t j = j0; j < j_tiles_end; j += 8) {
            for (int32_t i = i0; i < i_tiles_end; i += 4) skainet_fp32_tile_4x8(c, i, j, p0, p1);
        }
        /* Remaining rows (< 4) across the tiled columns, then the remaining columns (< 8) for all rows. */
        skainet_fp32_edge(c, i_tiles_end, i1, j0, j_tiles_end, p0, p1);
        skainet_fp32_edge(c, i0, i1, j_tiles_end, j1, p0, p1);
    }
}

#else /* portable + NEON: i-p-j outer product over the block */

static void skainet_fp32_block(const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j0, int32_t j1) {
    for (int32_t i = i0; i < i1; ++i) {
        float* SKAINET_RESTRICT c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t j = j0; j < j1; ++j) c_row[j] = 0.0f;
    }
    if (c->k <= 0) return;
    for (int32_t i = i0; i < i1; ++i) {
        const float* SKAINET_RESTRICT a_row = c->a + (size_t) i * c->a_stride;
        float* SKAINET_RESTRICT c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t p = 0; p < c->k; ++p) {
            const float a_ip = a_row[p];
            const float* SKAINET_RESTRICT b_row = c->b + (size_t) p * c->b_stride;
#ifdef SKAINET_HAVE_NEON
            const float32x4_t va = vdupq_n_f32(a_ip);
            int32_t j = j0;
            for (; j + 4 <= j1; j += 4) {
                float32x4_t cv = vld1q_f32(c_row + j);
                cv = vfmaq_f32(cv, va, vld1q_f32(b_row + j));
                vst1q_f32(c_row + j, cv);
            }
            for (; j < j1; ++j) {
                c_row[j] += a_ip * b_row[j];
            }
#else
            for (int32_t j = j0; j < j1; ++j) {
                c_row[j] += a_ip * b_row[j];
            }
#endif
        }
    }
}

#endif

static void skainet_fp32_gemm_range(void* vctx, int32_t start, int32_t end) {
    const skainet_fp32_gemm_ctx* c = (const skainet_fp32_gemm_ctx*) vctx;
    if (c->split_rows) skainet_fp32_block(c, start, end, 0, c->n);
    else skainet_fp32_block(c, 0, c->m, start, end);
}

SKAINET_API void skainet_fp32_matmul(
    const float* SKAINET_RESTRICT a, int32_t a_offset, int32_t a_stride,
    const float* SKAINET_RESTRICT b, int32_t b_offset, int32_t b_stride,
    float* SKAINET_RESTRICT c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;
    skainet_fp32_gemm_ctx ctx;
    ctx.a = a + a_offset;
    ctx.b = b + b_offset;
    ctx.c = c + c_offset;
    ctx.a_stride = a_stride;
    ctx.b_stride = b_stride;
    ctx.c_stride = c_stride;
    ctx.m = m;
    ctx.n = n;
    ctx.k = k < 0 ? 0 : k;
    ctx.split_rows = m >= n;
    skainet_run_rows(skainet_fp32_gemm_range, &ctx, ctx.split_rows ? m : n);
}
