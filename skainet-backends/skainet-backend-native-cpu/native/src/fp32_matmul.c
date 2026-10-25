#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"
#include "skainet_cpu_features.h"

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
 * Portable body: iteration order is i-p-j (outer product into rows of C).
 * The inner loop is `c[j] += a_ip * b[j]` over a contiguous run of `n`
 * floats for both b's row and c's row — auto-vectorizes under -O3
 * -ffast-math. It is what a target without one of the lanes below runs.
 *
 * AArch64 body (Apple silicon, Android arm64, Linux arm64): an 8×8 register
 * tile in sixteen float32x4 accumulators with lane-indexed FMAs, the packed
 * 8-column B panel and the tails of the AVX2 lane — see the NEON block.
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
 * x86-64 body: the shared library is built without -march (it has to run on
 * any x86-64), so the portable loop above is SSE2 code without FMA. Hosts
 * with AVX2+FMA take skainet_fp32_block_avx2 — compiled per function with
 * the avx2+fma target (MSVC emits VEX intrinsics without /arch) and selected
 * once per call by the cached cpuid probe in skainet_cpu_features.h, the
 * dispatch the x·Wᵀ kernel and the Apple arm64 dotprod kernels use. It is
 * the SIMD128 micro-kernel grown to the register file: a 6-row × 16-column
 * tile of C in twelve __m256 accumulators (two B loads and six broadcasts
 * feed twelve FMAs per p), the same 256-deep k-blocking, 2×16 and 1×16
 * tiles for the rows left under six and a lane-masked 1×w tile for the
 * columns left under sixteen. Every C element takes one fused
 * multiply-add per p in ascending p whichever tile holds it, so a split,
 * threaded or single call gives the same bits — the AVX2 lane differs from
 * the portable x86 lane (mul then add) the way NEON already does.
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

#elif defined(SKAINET_HAVE_NEON)

/*
 * AArch64 body: the same shape as the AVX2 lane below, sized for NEON's
 * 32 four-lane registers. An 8-row × 8-column tile of C lives in sixteen
 * float32x4 accumulators; four p-steps at a time, eight A loads (four
 * consecutive p of each row) and eight B loads feed sixty-four lane-indexed
 * FMAs (vfmaq_laneq_f32 takes the a-value straight from a lane, no
 * broadcast register). The same 256-deep k-blocking, the same packed
 * 8-column B panel (a power-of-two n otherwise maps every panel row to one
 * L1 set), 4×8 and 1×8 tiles for the rows left over and a zero-padded panel
 * plus C stripe for the columns left over. Every element takes one fused
 * step per p in ascending p whichever tile holds it.
 */

#define SKAINET_FP32_NEON_K_BLOCK 256

/* One p-step of the 8×8 tile with the a-values in lane LANE of a0..a7. */
#define SKAINET_FP32_NEON_8X8_STEP(LANE) \
    bl = vld1q_f32(b_row); bh = vld1q_f32(b_row + 4); b_row += b_stride; \
    s00 = vfmaq_laneq_f32(s00, bl, a0, LANE); s01 = vfmaq_laneq_f32(s01, bh, a0, LANE); \
    s10 = vfmaq_laneq_f32(s10, bl, a1, LANE); s11 = vfmaq_laneq_f32(s11, bh, a1, LANE); \
    s20 = vfmaq_laneq_f32(s20, bl, a2, LANE); s21 = vfmaq_laneq_f32(s21, bh, a2, LANE); \
    s30 = vfmaq_laneq_f32(s30, bl, a3, LANE); s31 = vfmaq_laneq_f32(s31, bh, a3, LANE); \
    s40 = vfmaq_laneq_f32(s40, bl, a4, LANE); s41 = vfmaq_laneq_f32(s41, bh, a4, LANE); \
    s50 = vfmaq_laneq_f32(s50, bl, a5, LANE); s51 = vfmaq_laneq_f32(s51, bh, a5, LANE); \
    s60 = vfmaq_laneq_f32(s60, bl, a6, LANE); s61 = vfmaq_laneq_f32(s61, bh, a6, LANE); \
    s70 = vfmaq_laneq_f32(s70, bl, a7, LANE); s71 = vfmaq_laneq_f32(s71, bh, a7, LANE)

/*
 * C[8 × 8] over p in [p0, p1). `a` is the first of the tile's rows of A, `b_row`
 * row p0 of an 8-column panel of B (the packed copy, or B itself with its
 * stride), `c0` the first of the tile's rows of C; accumulators are loaded from
 * C and stored back, so a tile continues the sum carried from the previous
 * k-block.
 */
static inline void skainet_fp32_neon_tile_8x8(
    const float* a, int32_t a_stride, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    const float* ar0 = a;
    const float* ar1 = ar0 + a_stride;
    const float* ar2 = ar1 + a_stride;
    const float* ar3 = ar2 + a_stride;
    const float* ar4 = ar3 + a_stride;
    const float* ar5 = ar4 + a_stride;
    const float* ar6 = ar5 + a_stride;
    const float* ar7 = ar6 + a_stride;
    float* c1 = c0 + c_stride;
    float* c2 = c1 + c_stride;
    float* c3 = c2 + c_stride;
    float* c4 = c3 + c_stride;
    float* c5 = c4 + c_stride;
    float* c6 = c5 + c_stride;
    float* c7 = c6 + c_stride;
    float32x4_t s00 = vld1q_f32(c0), s01 = vld1q_f32(c0 + 4);
    float32x4_t s10 = vld1q_f32(c1), s11 = vld1q_f32(c1 + 4);
    float32x4_t s20 = vld1q_f32(c2), s21 = vld1q_f32(c2 + 4);
    float32x4_t s30 = vld1q_f32(c3), s31 = vld1q_f32(c3 + 4);
    float32x4_t s40 = vld1q_f32(c4), s41 = vld1q_f32(c4 + 4);
    float32x4_t s50 = vld1q_f32(c5), s51 = vld1q_f32(c5 + 4);
    float32x4_t s60 = vld1q_f32(c6), s61 = vld1q_f32(c6 + 4);
    float32x4_t s70 = vld1q_f32(c7), s71 = vld1q_f32(c7 + 4);
    float32x4_t bl, bh;
    int32_t p = p0;
    for (; p + 4 <= p1; p += 4) {
        const float32x4_t a0 = vld1q_f32(ar0 + p), a1 = vld1q_f32(ar1 + p);
        const float32x4_t a2 = vld1q_f32(ar2 + p), a3 = vld1q_f32(ar3 + p);
        const float32x4_t a4 = vld1q_f32(ar4 + p), a5 = vld1q_f32(ar5 + p);
        const float32x4_t a6 = vld1q_f32(ar6 + p), a7 = vld1q_f32(ar7 + p);
        SKAINET_FP32_NEON_8X8_STEP(0);
        SKAINET_FP32_NEON_8X8_STEP(1);
        SKAINET_FP32_NEON_8X8_STEP(2);
        SKAINET_FP32_NEON_8X8_STEP(3);
    }
    for (; p < p1; ++p) {
        bl = vld1q_f32(b_row); bh = vld1q_f32(b_row + 4); b_row += b_stride;
        s00 = vfmaq_n_f32(s00, bl, ar0[p]); s01 = vfmaq_n_f32(s01, bh, ar0[p]);
        s10 = vfmaq_n_f32(s10, bl, ar1[p]); s11 = vfmaq_n_f32(s11, bh, ar1[p]);
        s20 = vfmaq_n_f32(s20, bl, ar2[p]); s21 = vfmaq_n_f32(s21, bh, ar2[p]);
        s30 = vfmaq_n_f32(s30, bl, ar3[p]); s31 = vfmaq_n_f32(s31, bh, ar3[p]);
        s40 = vfmaq_n_f32(s40, bl, ar4[p]); s41 = vfmaq_n_f32(s41, bh, ar4[p]);
        s50 = vfmaq_n_f32(s50, bl, ar5[p]); s51 = vfmaq_n_f32(s51, bh, ar5[p]);
        s60 = vfmaq_n_f32(s60, bl, ar6[p]); s61 = vfmaq_n_f32(s61, bh, ar6[p]);
        s70 = vfmaq_n_f32(s70, bl, ar7[p]); s71 = vfmaq_n_f32(s71, bh, ar7[p]);
    }
    vst1q_f32(c0, s00); vst1q_f32(c0 + 4, s01);
    vst1q_f32(c1, s10); vst1q_f32(c1 + 4, s11);
    vst1q_f32(c2, s20); vst1q_f32(c2 + 4, s21);
    vst1q_f32(c3, s30); vst1q_f32(c3 + 4, s31);
    vst1q_f32(c4, s40); vst1q_f32(c4 + 4, s41);
    vst1q_f32(c5, s50); vst1q_f32(c5 + 4, s51);
    vst1q_f32(c6, s60); vst1q_f32(c6 + 4, s61);
    vst1q_f32(c7, s70); vst1q_f32(c7 + 4, s71);
}

#define SKAINET_FP32_NEON_4X8_STEP(LANE) \
    bl = vld1q_f32(b_row); bh = vld1q_f32(b_row + 4); b_row += b_stride; \
    s00 = vfmaq_laneq_f32(s00, bl, a0, LANE); s01 = vfmaq_laneq_f32(s01, bh, a0, LANE); \
    s10 = vfmaq_laneq_f32(s10, bl, a1, LANE); s11 = vfmaq_laneq_f32(s11, bh, a1, LANE); \
    s20 = vfmaq_laneq_f32(s20, bl, a2, LANE); s21 = vfmaq_laneq_f32(s21, bh, a2, LANE); \
    s30 = vfmaq_laneq_f32(s30, bl, a3, LANE); s31 = vfmaq_laneq_f32(s31, bh, a3, LANE)

/* C[4 × 8]: the rows left under eight, four at a time. */
static inline void skainet_fp32_neon_tile_4x8(
    const float* a, int32_t a_stride, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    const float* ar0 = a;
    const float* ar1 = ar0 + a_stride;
    const float* ar2 = ar1 + a_stride;
    const float* ar3 = ar2 + a_stride;
    float* c1 = c0 + c_stride;
    float* c2 = c1 + c_stride;
    float* c3 = c2 + c_stride;
    float32x4_t s00 = vld1q_f32(c0), s01 = vld1q_f32(c0 + 4);
    float32x4_t s10 = vld1q_f32(c1), s11 = vld1q_f32(c1 + 4);
    float32x4_t s20 = vld1q_f32(c2), s21 = vld1q_f32(c2 + 4);
    float32x4_t s30 = vld1q_f32(c3), s31 = vld1q_f32(c3 + 4);
    float32x4_t bl, bh;
    int32_t p = p0;
    for (; p + 4 <= p1; p += 4) {
        const float32x4_t a0 = vld1q_f32(ar0 + p), a1 = vld1q_f32(ar1 + p);
        const float32x4_t a2 = vld1q_f32(ar2 + p), a3 = vld1q_f32(ar3 + p);
        SKAINET_FP32_NEON_4X8_STEP(0);
        SKAINET_FP32_NEON_4X8_STEP(1);
        SKAINET_FP32_NEON_4X8_STEP(2);
        SKAINET_FP32_NEON_4X8_STEP(3);
    }
    for (; p < p1; ++p) {
        bl = vld1q_f32(b_row); bh = vld1q_f32(b_row + 4); b_row += b_stride;
        s00 = vfmaq_n_f32(s00, bl, ar0[p]); s01 = vfmaq_n_f32(s01, bh, ar0[p]);
        s10 = vfmaq_n_f32(s10, bl, ar1[p]); s11 = vfmaq_n_f32(s11, bh, ar1[p]);
        s20 = vfmaq_n_f32(s20, bl, ar2[p]); s21 = vfmaq_n_f32(s21, bh, ar2[p]);
        s30 = vfmaq_n_f32(s30, bl, ar3[p]); s31 = vfmaq_n_f32(s31, bh, ar3[p]);
    }
    vst1q_f32(c0, s00); vst1q_f32(c0 + 4, s01);
    vst1q_f32(c1, s10); vst1q_f32(c1 + 4, s11);
    vst1q_f32(c2, s20); vst1q_f32(c2 + 4, s21);
    vst1q_f32(c3, s30); vst1q_f32(c3 + 4, s31);
}

/* C[1 × 8]: the last rows, one at a time. */
static inline void skainet_fp32_neon_tile_1x8(
    const float* a, const float* b_row, int32_t b_stride, float* c0, int32_t p0, int32_t p1
) {
    float32x4_t s0 = vld1q_f32(c0), s1 = vld1q_f32(c0 + 4);
    int32_t p = p0;
    for (; p + 4 <= p1; p += 4) {
        const float32x4_t av = vld1q_f32(a + p);
        s0 = vfmaq_laneq_f32(s0, vld1q_f32(b_row), av, 0); s1 = vfmaq_laneq_f32(s1, vld1q_f32(b_row + 4), av, 0); b_row += b_stride;
        s0 = vfmaq_laneq_f32(s0, vld1q_f32(b_row), av, 1); s1 = vfmaq_laneq_f32(s1, vld1q_f32(b_row + 4), av, 1); b_row += b_stride;
        s0 = vfmaq_laneq_f32(s0, vld1q_f32(b_row), av, 2); s1 = vfmaq_laneq_f32(s1, vld1q_f32(b_row + 4), av, 2); b_row += b_stride;
        s0 = vfmaq_laneq_f32(s0, vld1q_f32(b_row), av, 3); s1 = vfmaq_laneq_f32(s1, vld1q_f32(b_row + 4), av, 3); b_row += b_stride;
    }
    for (; p < p1; ++p, b_row += b_stride) {
        s0 = vfmaq_n_f32(s0, vld1q_f32(b_row), a[p]); s1 = vfmaq_n_f32(s1, vld1q_f32(b_row + 4), a[p]);
    }
    vst1q_f32(c0, s0); vst1q_f32(c0 + 4, s1);
}

/* Rows [0, rows) of A against the panel `b_row` into C rows `c0` (both 8 wide): 8×8 tiles, then 4×8, then 1×8. */
static inline void skainet_fp32_neon_column(
    const float* a, int32_t a_stride, int32_t rows, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    int32_t i = 0;
    for (; i + 8 <= rows; i += 8) {
        skainet_fp32_neon_tile_8x8(a + (size_t) i * a_stride, a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, c_stride, p0, p1);
    }
    for (; i + 4 <= rows; i += 4) {
        skainet_fp32_neon_tile_4x8(a + (size_t) i * a_stride, a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, c_stride, p0, p1);
    }
    for (; i < rows; ++i) {
        skainet_fp32_neon_tile_1x8(a + (size_t) i * a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, p0, p1);
    }
}

/*
 * B[p0..p1, j..j+w) copied into a contiguous 8-wide panel, columns at or past
 * w (the column tail, w < 8) zero-filled. See the AVX2 lane for why the copy
 * pays: in place, a power-of-two n puts every row of the panel in one L1 set.
 */
static inline void skainet_fp32_neon_pack_panel(
    float* panel, const float* b, int32_t b_stride, int32_t w, int32_t p0, int32_t p1
) {
    const float* b_row = b + (size_t) p0 * b_stride;
    if (w == 8) {
        for (int32_t p = p0; p < p1; ++p, b_row += b_stride, panel += 8) {
            vst1q_f32(panel, vld1q_f32(b_row));
            vst1q_f32(panel + 4, vld1q_f32(b_row + 4));
        }
    } else {
        for (int32_t p = p0; p < p1; ++p, b_row += b_stride, panel += 8) {
            int32_t j = 0;
            for (; j < w; ++j) panel[j] = b_row[j];
            for (; j < 8; ++j) panel[j] = 0.0f;
        }
    }
}

/*
 * The columns left under eight, from j, for rows [i0, i1): the C block is
 * copied into an 8-wide stripe, the full tiles run over the stripe and the
 * zero-padded tail panel, and the live columns are stored back — the same
 * fused steps a wider block would give them.
 */
static void skainet_fp32_neon_tail(
    const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j, int32_t w,
    const float* panel, int32_t p0, int32_t p1
) {
    float stripe[8 * 8];
    for (int32_t i = i0; i < i1; i += 8) {
        const int32_t rows = (i1 - i) < 8 ? (i1 - i) : 8;
        float* c_row = c->c + (size_t) i * c->c_stride + j;
        for (int32_t r = 0; r < rows; ++r) {
            int32_t jj = 0;
            for (; jj < w; ++jj) stripe[r * 8 + jj] = c_row[(size_t) r * c->c_stride + jj];
            for (; jj < 8; ++jj) stripe[r * 8 + jj] = 0.0f;
        }
        skainet_fp32_neon_column(c->a + (size_t) i * c->a_stride, c->a_stride, rows, panel, 8, stripe, 8, p0, p1);
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t jj = 0; jj < w; ++jj) c_row[(size_t) r * c->c_stride + jj] = stripe[r * 8 + jj];
        }
    }
}

/* Rows [i0, i1) × columns [j0, j1) of C, k-blocked: the 8-column panels first, the column tail after. */
static void skainet_fp32_block(const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j0, int32_t j1) {
    for (int32_t i = i0; i < i1; ++i) {
        float* c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t j = j0; j < j1; ++j) c_row[j] = 0.0f;
    }
    if (c->k <= 0) return;
    const int32_t rows = i1 - i0;
    const int32_t j_tiles_end = j0 + ((j1 - j0) / 8) * 8;
    const int32_t j_tail = j1 - j_tiles_end;
    /* The packed copy pays for itself from two row tiles up; a thinner block reads B in place. */
    const int pack = rows >= 16;
    float panel[SKAINET_FP32_NEON_K_BLOCK * 8];
    const float* a = c->a + (size_t) i0 * c->a_stride;
    for (int32_t p0 = 0; p0 < c->k; p0 += SKAINET_FP32_NEON_K_BLOCK) {
        const int32_t p1 = (c->k - p0) < SKAINET_FP32_NEON_K_BLOCK ? c->k : p0 + SKAINET_FP32_NEON_K_BLOCK;
        for (int32_t j = j0; j < j_tiles_end; j += 8) {
            float* c0 = c->c + (size_t) i0 * c->c_stride + j;
            if (pack) {
                skainet_fp32_neon_pack_panel(panel, c->b + j, c->b_stride, 8, p0, p1);
                skainet_fp32_neon_column(a, c->a_stride, rows, panel, 8, c0, c->c_stride, p0, p1);
            } else {
                skainet_fp32_neon_column(a, c->a_stride, rows, c->b + (size_t) p0 * c->b_stride + j, c->b_stride, c0, c->c_stride, p0, p1);
            }
        }
        if (j_tail > 0) {
            skainet_fp32_neon_pack_panel(panel, c->b + j_tiles_end, c->b_stride, j_tail, p0, p1);
            skainet_fp32_neon_tail(c, i0, i1, j_tiles_end, j_tail, panel, p0, p1);
        }
    }
}

#else /* portable: i-p-j outer product over the block */

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
            for (int32_t j = j0; j < j1; ++j) {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__wasm__)
#define SKAINET_FP32_GEMM_AVX2 1
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#  define SKAINET_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#  define SKAINET_AVX2_TARGET
#endif

#define SKAINET_FP32_AVX2_K_BLOCK 256

/*
 * The tiles below take their operands as plain pointers: `a` is the first of
 * the tile's rows of A (a_stride apart), `b_row` is row p0 of a 16-column
 * panel of B (b_stride apart — the packed copy, or B itself), `c0` the first
 * of the tile's rows of C (c_stride apart — C itself, or the 16-wide stripe
 * the column tail is computed in). Accumulators are loaded from C and stored
 * back, so a tile continues the sum carried in C from the previous k-block.
 */

/*
 * C[6 × 16] over p in [p0, p1): twelve accumulators. Per p two B loads and
 * six broadcasts feed twelve FMAs, so the loop is bound by the two FMA ports,
 * not by loads.
 */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_tile_6x16(
    const float* a, int32_t a_stride, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    const float* a0 = a;
    const float* a1 = a0 + a_stride;
    const float* a2 = a1 + a_stride;
    const float* a3 = a2 + a_stride;
    const float* a4 = a3 + a_stride;
    const float* a5 = a4 + a_stride;
    float* c1 = c0 + c_stride;
    float* c2 = c1 + c_stride;
    float* c3 = c2 + c_stride;
    float* c4 = c3 + c_stride;
    float* c5 = c4 + c_stride;
    __m256 s00 = _mm256_loadu_ps(c0), s01 = _mm256_loadu_ps(c0 + 8);
    __m256 s10 = _mm256_loadu_ps(c1), s11 = _mm256_loadu_ps(c1 + 8);
    __m256 s20 = _mm256_loadu_ps(c2), s21 = _mm256_loadu_ps(c2 + 8);
    __m256 s30 = _mm256_loadu_ps(c3), s31 = _mm256_loadu_ps(c3 + 8);
    __m256 s40 = _mm256_loadu_ps(c4), s41 = _mm256_loadu_ps(c4 + 8);
    __m256 s50 = _mm256_loadu_ps(c5), s51 = _mm256_loadu_ps(c5 + 8);
    for (int32_t p = p0; p < p1; ++p, b_row += b_stride) {
        const __m256 b0 = _mm256_loadu_ps(b_row);
        const __m256 b1 = _mm256_loadu_ps(b_row + 8);
        __m256 x = _mm256_broadcast_ss(a0 + p);
        s00 = _mm256_fmadd_ps(x, b0, s00); s01 = _mm256_fmadd_ps(x, b1, s01);
        x = _mm256_broadcast_ss(a1 + p);
        s10 = _mm256_fmadd_ps(x, b0, s10); s11 = _mm256_fmadd_ps(x, b1, s11);
        x = _mm256_broadcast_ss(a2 + p);
        s20 = _mm256_fmadd_ps(x, b0, s20); s21 = _mm256_fmadd_ps(x, b1, s21);
        x = _mm256_broadcast_ss(a3 + p);
        s30 = _mm256_fmadd_ps(x, b0, s30); s31 = _mm256_fmadd_ps(x, b1, s31);
        x = _mm256_broadcast_ss(a4 + p);
        s40 = _mm256_fmadd_ps(x, b0, s40); s41 = _mm256_fmadd_ps(x, b1, s41);
        x = _mm256_broadcast_ss(a5 + p);
        s50 = _mm256_fmadd_ps(x, b0, s50); s51 = _mm256_fmadd_ps(x, b1, s51);
    }
    _mm256_storeu_ps(c0, s00); _mm256_storeu_ps(c0 + 8, s01);
    _mm256_storeu_ps(c1, s10); _mm256_storeu_ps(c1 + 8, s11);
    _mm256_storeu_ps(c2, s20); _mm256_storeu_ps(c2 + 8, s21);
    _mm256_storeu_ps(c3, s30); _mm256_storeu_ps(c3 + 8, s31);
    _mm256_storeu_ps(c4, s40); _mm256_storeu_ps(c4 + 8, s41);
    _mm256_storeu_ps(c5, s50); _mm256_storeu_ps(c5 + 8, s51);
}

/* C[2 × 16]: the rows left under six, two at a time. */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_tile_2x16(
    const float* a, int32_t a_stride, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    const float* a0 = a;
    const float* a1 = a0 + a_stride;
    float* c1 = c0 + c_stride;
    __m256 s00 = _mm256_loadu_ps(c0), s01 = _mm256_loadu_ps(c0 + 8);
    __m256 s10 = _mm256_loadu_ps(c1), s11 = _mm256_loadu_ps(c1 + 8);
    for (int32_t p = p0; p < p1; ++p, b_row += b_stride) {
        const __m256 b0 = _mm256_loadu_ps(b_row);
        const __m256 b1 = _mm256_loadu_ps(b_row + 8);
        __m256 x = _mm256_broadcast_ss(a0 + p);
        s00 = _mm256_fmadd_ps(x, b0, s00); s01 = _mm256_fmadd_ps(x, b1, s01);
        x = _mm256_broadcast_ss(a1 + p);
        s10 = _mm256_fmadd_ps(x, b0, s10); s11 = _mm256_fmadd_ps(x, b1, s11);
    }
    _mm256_storeu_ps(c0, s00); _mm256_storeu_ps(c0 + 8, s01);
    _mm256_storeu_ps(c1, s10); _mm256_storeu_ps(c1 + 8, s11);
}

/* C[1 × 16]: the last odd row. */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_tile_1x16(
    const float* a, const float* b_row, int32_t b_stride, float* c0, int32_t p0, int32_t p1
) {
    __m256 s0 = _mm256_loadu_ps(c0), s1 = _mm256_loadu_ps(c0 + 8);
    for (int32_t p = p0; p < p1; ++p, b_row += b_stride) {
        const __m256 x = _mm256_broadcast_ss(a + p);
        s0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row), s0);
        s1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row + 8), s1);
    }
    _mm256_storeu_ps(c0, s0); _mm256_storeu_ps(c0 + 8, s1);
}

/* Rows [i0, i1) of A against the panel `b_row` into C rows `c0` (both 16 wide): 6×16 tiles, then 2×16, then 1×16. */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_column(
    const float* a, int32_t a_stride, int32_t rows, const float* b_row, int32_t b_stride,
    float* c0, int32_t c_stride, int32_t p0, int32_t p1
) {
    int32_t i = 0;
    for (; i + 6 <= rows; i += 6) {
        skainet_fp32_avx2_tile_6x16(a + (size_t) i * a_stride, a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, c_stride, p0, p1);
    }
    for (; i + 2 <= rows; i += 2) {
        skainet_fp32_avx2_tile_2x16(a + (size_t) i * a_stride, a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, c_stride, p0, p1);
    }
    if (i < rows) {
        skainet_fp32_avx2_tile_1x16(a + (size_t) i * a_stride, b_row, b_stride, c0 + (size_t) i * c_stride, p0, p1);
    }
}

/*
 * B[p0..p1, j..j+16) copied into a contiguous 16-wide panel. In place, the
 * panel's rows are b_stride floats apart, and when n is a multiple of 1024
 * that is a multiple of 4 KiB: every row maps to the same L1 set, only eight
 * of them fit, and the "resident" panel thrashes (measured: 63×1024×1024 at
 * half the speed of 63×1021×1023). Packed, the panel is one line per p in
 * consecutive sets. Lanes at or past `w` (the column tail) are masked off —
 * VMASKMOV neither reads nor faults on them — and come out as zeros, so the
 * full tiles can run over a tail panel; the matching C lanes are never
 * stored (see the tail stripe below).
 */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_pack_panel(
    float* panel, const float* b, int32_t b_stride, int32_t p0, int32_t p1, __m256i m0, __m256i m1
) {
    const float* b_row = b + (size_t) p0 * b_stride;
    for (int32_t p = p0; p < p1; ++p, b_row += b_stride, panel += 16) {
        _mm256_storeu_ps(panel, _mm256_maskload_ps(b_row, m0));
        _mm256_storeu_ps(panel + 8, _mm256_maskload_ps(b_row + 8, m1));
    }
}

/* Masks for the lanes below w (0..16) in each half of a 16-wide row. */
SKAINET_AVX2_TARGET
static inline void skainet_fp32_avx2_lane_masks(int32_t w, __m256i* m0, __m256i* m1) {
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    *m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(w), lane);
    *m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(w - 8), lane);
}

/*
 * The columns left under sixteen, from j, for rows [i0, i1): the C block is
 * copied (lanes past the tail width masked off) into a 16-wide stripe, the full tiles
 * run over the stripe and the zero-padded tail panel, and the live lanes are
 * stored back. Each live element takes exactly the fused steps it would in a
 * full tile, so the tail is as fast per row as the body and bit-identical to
 * what a wider block would have given it.
 */
SKAINET_AVX2_TARGET
static void skainet_fp32_avx2_tail(
    const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j,
    const float* panel, int32_t p0, int32_t p1, __m256i m0, __m256i m1
) {
    float stripe[6 * 16];
    for (int32_t i = i0; i < i1; i += 6) {
        const int32_t rows = (i1 - i) < 6 ? (i1 - i) : 6;
        float* c_row = c->c + (size_t) i * c->c_stride + j;
        for (int32_t r = 0; r < rows; ++r) {
            _mm256_storeu_ps(stripe + r * 16, _mm256_maskload_ps(c_row + (size_t) r * c->c_stride, m0));
            _mm256_storeu_ps(stripe + r * 16 + 8, _mm256_maskload_ps(c_row + (size_t) r * c->c_stride + 8, m1));
        }
        skainet_fp32_avx2_column(c->a + (size_t) i * c->a_stride, c->a_stride, rows, panel, 16, stripe, 16, p0, p1);
        for (int32_t r = 0; r < rows; ++r) {
            _mm256_maskstore_ps(c_row + (size_t) r * c->c_stride, m0, _mm256_loadu_ps(stripe + r * 16));
            _mm256_maskstore_ps(c_row + (size_t) r * c->c_stride + 8, m1, _mm256_loadu_ps(stripe + r * 16 + 8));
        }
    }
}

/* Rows [i0, i1) × columns [j0, j1) of C, k-blocked: the 16-column panels first, the column tail after. */
SKAINET_AVX2_TARGET
static void skainet_fp32_block_avx2(const skainet_fp32_gemm_ctx* c, int32_t i0, int32_t i1, int32_t j0, int32_t j1) {
    for (int32_t i = i0; i < i1; ++i) {
        float* c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t j = j0; j < j1; ++j) c_row[j] = 0.0f;
    }
    if (c->k <= 0) return;
    const int32_t rows = i1 - i0;
    const int32_t j_tiles_end = j0 + ((j1 - j0) / 16) * 16;
    const int32_t j_tail = j1 - j_tiles_end;
    /* The packed copy pays for itself from two row tiles up; a thinner block reads B in place. */
    const int pack = rows >= 12;
    __m256i all0, all1, tail0, tail1;
    skainet_fp32_avx2_lane_masks(16, &all0, &all1);
    skainet_fp32_avx2_lane_masks(j_tail, &tail0, &tail1);
    float panel[SKAINET_FP32_AVX2_K_BLOCK * 16];
    const float* a = c->a + (size_t) i0 * c->a_stride;
    for (int32_t p0 = 0; p0 < c->k; p0 += SKAINET_FP32_AVX2_K_BLOCK) {
        const int32_t p1 = (c->k - p0) < SKAINET_FP32_AVX2_K_BLOCK ? c->k : p0 + SKAINET_FP32_AVX2_K_BLOCK;
        for (int32_t j = j0; j < j_tiles_end; j += 16) {
            float* c0 = c->c + (size_t) i0 * c->c_stride + j;
            if (pack) {
                skainet_fp32_avx2_pack_panel(panel, c->b + j, c->b_stride, p0, p1, all0, all1);
                skainet_fp32_avx2_column(a, c->a_stride, rows, panel, 16, c0, c->c_stride, p0, p1);
            } else {
                skainet_fp32_avx2_column(a, c->a_stride, rows, c->b + (size_t) p0 * c->b_stride + j, c->b_stride, c0, c->c_stride, p0, p1);
            }
        }
        if (j_tail > 0) {
            skainet_fp32_avx2_pack_panel(panel, c->b + j_tiles_end, c->b_stride, p0, p1, tail0, tail1);
            skainet_fp32_avx2_tail(c, i0, i1, j_tiles_end, panel, p0, p1, tail0, tail1);
        }
    }
}

static void skainet_fp32_gemm_range_avx2(void* vctx, int32_t start, int32_t end) {
    const skainet_fp32_gemm_ctx* c = (const skainet_fp32_gemm_ctx*) vctx;
    if (c->split_rows) skainet_fp32_block_avx2(c, start, end, 0, c->n);
    else skainet_fp32_block_avx2(c, 0, c->m, start, end);
}
#endif /* x86-64 */

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
    skainet_row_range_fn range = skainet_fp32_gemm_range;
#ifdef SKAINET_FP32_GEMM_AVX2
    if (skainet_cpu_has_avx2_fma()) range = skainet_fp32_gemm_range_avx2;
#endif
    skainet_run_rows(range, &ctx, ctx.split_rows ? m : n);
}
