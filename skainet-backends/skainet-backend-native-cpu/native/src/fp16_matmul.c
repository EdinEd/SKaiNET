#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native row-major FP32 × FP16 matmul matching the
 * sk.ainet.backend.api.kernel.Fp16MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 *   A, C : FP32 (FloatArray on the JVM side; float* here).
 *          Strides in floats. `a_stride == k` for a contiguous parent.
 *   B    : packed IEEE binary16 (ByteArray on the JVM side; uint8_t* here).
 *          Strides in *bytes*. `b_byte_stride == n * 2` for a contiguous
 *          parent. Each FP16 value is 2 bytes little-endian.
 *
 * The BF16 sibling gets its dequant for free — BF16 is the high half of an
 * FP32, so the conversion is one shift. Binary16 has a narrower exponent and
 * a wider mantissa, so it needs rebiasing, and subnormals need renormalizing.
 * Doing that with branches would cost more than the multiply it feeds, so
 * fp16_to_float below is branch-free: the two special cases are folded in
 * with arithmetic masks, which keeps the inner loop a straight-line sequence
 * the vectorizer can widen (unpack to 32-bit lanes, integer ops, one FMA).
 * The SIMD128 body is that same sequence on i32x4 lanes, so it decodes
 * bit-for-bit what the scalar does; the NEON body uses the ISA's own
 * binary16 → binary32 convert (base AArch64, exact for every input).
 *
 * Deliberately *not* using _Float16 or F16C intrinsics on x86. The x86_64
 * build carries no -march flag, so F16C cannot be assumed, and _Float16
 * without it lowers to libgcc helper calls that are slower than this and
 * block vectorization outright.
 *
 * Iteration order depends on m.
 *
 * At m == 1 it is plain i-p-j: each B element is used exactly once, so there
 * is nothing to reuse and the best thing to do is stream B sequentially.
 * Tiling here is pure overhead — measured 15% slower on ffn_up 8B — and m == 1
 * is the decode step of inference.
 *
 * At m > 1, i-p-j would walk the whole of B once per row of A: for ffn_up 8B
 * at m=16 that is 16 passes over 90 MiB. Instead j is tiled, and within a tile
 * each B row is decoded once into a small stack buffer and multiplied into all
 * m rows of C. B is then read once in total rather than m times, and the
 * decode count drops from m*k*n to k*n. The tile is sized so the decoded row
 * and the m C rows it feeds stay resident together.
 *
 * Accumulation order into any given C element is p ascending on every path,
 * so they are bit-identical to each other and to the original i-p-j
 * formulation, not merely close.
 *
 * Threading (#1195 pattern): the columns of C are split into disjoint ranges
 * through skainet_row_threads.h when n is large enough; each worker runs the
 * whole m × k work for its columns, so no accumulation order changes. On
 * Emscripten that is the pthread pool of the mt sidecar.
 *
 * Caller contract (mirrors skainet_bf16_matmul):
 *  - C is FULLY OVERWRITTEN in the m×n block.
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; defensively treated as no-op.
 *
 * NaN note: a signaling binary16 NaN stays signaling on the scalar and
 * SIMD128 paths, matching the JVM Panama kernel; the NEON convert quiets it.
 * Neither is observable — the value is immediately multiplied, and that
 * quiets it anyway.
 */

/* binary16's exponent field once shifted into FP32 position (0x7C00 << 13). */
#define SKAINET_FP16_EXP_FIELD 0x0F800000u
/* (127 - 15) << 23 — the FP32/binary16 exponent bias difference. */
#define SKAINET_FP16_REBIAS 0x38000000u
/* 1 << 23 — one exponent step, lifting a subnormal into the magic binade. */
#define SKAINET_FP16_SUBNORMAL_BUMP 0x00800000u

/*
 * Columns decoded per pass. 512 floats is a 2 KiB stack buffer — small enough
 * to leave the C rows it feeds resident alongside it, large enough that the
 * per-tile loop overhead disappears against the k*m inner work.
 */
#define SKAINET_FP16_TILE 512

static inline float skainet_fp16_to_float(uint16_t h) {
    const uint32_t bits = (uint32_t) h;
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t shifted = (bits & 0x7FFFu) << 13;
    const uint32_t exp_field = shifted & SKAINET_FP16_EXP_FIELD;

    /* 0 or 0xFFFFFFFF, so the corrections below are masks rather than jumps. */
    const uint32_t is_inf_nan =
        (uint32_t) -(int32_t) (exp_field == SKAINET_FP16_EXP_FIELD);
    const uint32_t is_subnormal = (uint32_t) -(int32_t) (exp_field == 0u);

    /* Rebias; Inf/NaN takes a second rebias, which saturates the exponent. */
    const uint32_t biased =
        shifted + SKAINET_FP16_REBIAS + (SKAINET_FP16_REBIAS & is_inf_nan);

    /* Zero and subnormals: bump one exponent step and subtract 2^-14, which
     * makes the FPU renormalize. For a subnormal m * 2^-24 the bumped value is
     * 2^-14 * (1 + m * 2^-10), so the subtraction leaves exactly m * 2^-24;
     * for zero it leaves +0. The sign is reapplied afterwards either way. */
    const uint32_t bumped = biased + SKAINET_FP16_SUBNORMAL_BUMP;
    float renormalized;
    memcpy(&renormalized, &bumped, sizeof(float));
    renormalized -= 0x1p-14f;
    uint32_t renormalized_bits;
    memcpy(&renormalized_bits, &renormalized, sizeof(uint32_t));

    const uint32_t magnitude =
        (renormalized_bits & is_subnormal) | (biased & ~is_subnormal);
    const uint32_t out_bits = sign | magnitude;

    float out;
    memcpy(&out, &out_bits, sizeof(float));
    return out;
}

static inline float skainet_fp16_load(const uint8_t* SKAINET_RESTRICT src) {
    /* Read 2 bytes LE. memcpy is strict-aliasing safe and the compiler folds
     * it to a single 16-bit load. */
    uint16_t bits;
    memcpy(&bits, src, sizeof(uint16_t));
    return skainet_fp16_to_float(bits);
}

#if defined(SKAINET_HAVE_WASM_SIMD128)
/* The branch-free scalar conversion on four u32 lanes holding binary16 bits. */
static inline v128_t skainet_fp16x4_to_f32x4(v128_t bits) {
    const v128_t exp_mask = wasm_i32x4_splat((int32_t) SKAINET_FP16_EXP_FIELD);
    const v128_t rebias = wasm_i32x4_splat((int32_t) SKAINET_FP16_REBIAS);
    const v128_t sign = wasm_i32x4_shl(wasm_v128_and(bits, wasm_i32x4_splat(0x8000)), 16);
    const v128_t shifted = wasm_i32x4_shl(wasm_v128_and(bits, wasm_i32x4_splat(0x7FFF)), 13);
    const v128_t exp_field = wasm_v128_and(shifted, exp_mask);
    const v128_t is_inf_nan = wasm_i32x4_eq(exp_field, exp_mask);
    const v128_t is_subnormal = wasm_i32x4_eq(exp_field, wasm_i32x4_splat(0));
    const v128_t biased = wasm_i32x4_add(wasm_i32x4_add(shifted, rebias), wasm_v128_and(rebias, is_inf_nan));
    const v128_t bumped = wasm_i32x4_add(biased, wasm_i32x4_splat((int32_t) SKAINET_FP16_SUBNORMAL_BUMP));
    const v128_t renormalized = wasm_f32x4_sub(bumped, wasm_f32x4_splat(0x1p-14f));
    const v128_t magnitude = wasm_v128_bitselect(renormalized, biased, is_subnormal);
    return wasm_v128_or(sign, magnitude);
}
#endif

/* c[0..count) += a_ip * fp16(src[0..count)) — the m == 1 stream. Eight values per SIMD step, scalar tail. */
static inline void skainet_fp16_stream_axpy(
    float* SKAINET_RESTRICT c, const uint8_t* SKAINET_RESTRICT src, float a_ip, int32_t count
) {
    int32_t j = 0;
#if defined(SKAINET_HAVE_NEON)
    const float32x4_t va = vdupq_n_f32(a_ip);
    for (; j + 8 <= count; j += 8) {
        const uint16x8_t v = vreinterpretq_u16_u8(vld1q_u8(src + (size_t) j * 2));
        const float32x4_t lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v)));
        const float32x4_t hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v)));
        vst1q_f32(c + j, vfmaq_f32(vld1q_f32(c + j), va, lo));
        vst1q_f32(c + j + 4, vfmaq_f32(vld1q_f32(c + j + 4), va, hi));
    }
#elif defined(SKAINET_HAVE_WASM_SIMD128)
    const v128_t va = wasm_f32x4_splat(a_ip);
    for (; j + 8 <= count; j += 8) {
        const v128_t v = wasm_v128_load(src + (size_t) j * 2);
        const v128_t lo = skainet_fp16x4_to_f32x4(wasm_u32x4_extend_low_u16x8(v));
        const v128_t hi = skainet_fp16x4_to_f32x4(wasm_u32x4_extend_high_u16x8(v));
        wasm_v128_store(c + j, wasm_f32x4_add(wasm_v128_load(c + j), wasm_f32x4_mul(va, lo)));
        wasm_v128_store(c + j + 4, wasm_f32x4_add(wasm_v128_load(c + j + 4), wasm_f32x4_mul(va, hi)));
    }
#endif
    for (; j < count; ++j) c[j] += a_ip * skainet_fp16_load(src + (size_t) j * 2);
}

/* dst[0..count) = fp16(src[0..count)) — the tile decode. */
static inline void skainet_fp16_widen(
    float* SKAINET_RESTRICT dst, const uint8_t* SKAINET_RESTRICT src, int32_t count
) {
    int32_t j = 0;
#if defined(SKAINET_HAVE_NEON)
    for (; j + 8 <= count; j += 8) {
        const uint16x8_t v = vreinterpretq_u16_u8(vld1q_u8(src + (size_t) j * 2));
        vst1q_f32(dst + j, vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v))));
        vst1q_f32(dst + j + 4, vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v))));
    }
#elif defined(SKAINET_HAVE_WASM_SIMD128)
    for (; j + 8 <= count; j += 8) {
        const v128_t v = wasm_v128_load(src + (size_t) j * 2);
        wasm_v128_store(dst + j, skainet_fp16x4_to_f32x4(wasm_u32x4_extend_low_u16x8(v)));
        wasm_v128_store(dst + j + 4, skainet_fp16x4_to_f32x4(wasm_u32x4_extend_high_u16x8(v)));
    }
#endif
    for (; j < count; ++j) dst[j] = skainet_fp16_load(src + (size_t) j * 2);
}

/* c[0..count) += a_ip * w[0..count) on the decoded tile. */
static inline void skainet_f32_axpy(
    float* SKAINET_RESTRICT c, const float* SKAINET_RESTRICT w, float a_ip, int32_t count
) {
    int32_t j = 0;
#if defined(SKAINET_HAVE_NEON)
    const float32x4_t va = vdupq_n_f32(a_ip);
    for (; j + 4 <= count; j += 4) vst1q_f32(c + j, vfmaq_f32(vld1q_f32(c + j), va, vld1q_f32(w + j)));
#elif defined(SKAINET_HAVE_WASM_SIMD128)
    const v128_t va = wasm_f32x4_splat(a_ip);
    for (; j + 4 <= count; j += 4) {
        wasm_v128_store(c + j, wasm_f32x4_add(wasm_v128_load(c + j), wasm_f32x4_mul(va, wasm_v128_load(w + j))));
    }
#endif
    for (; j < count; ++j) c[j] += a_ip * w[j];
}

typedef struct {
    const float* a;        /* a + a_offset */
    const uint8_t* b;      /* b + b_byte_offset */
    float* c;              /* c + c_offset */
    int32_t a_stride, b_byte_stride, c_stride;
    int32_t m, n, k;
} skainet_fp16_ctx;

/* Columns [j0, j1) of C, all rows. */
static void skainet_fp16_columns(void* vctx, int32_t j0, int32_t j1) {
    const skainet_fp16_ctx* c = (const skainet_fp16_ctx*) vctx;

    /* Zero the output block. Required by the SPI for k == 0 AND a
     * prerequisite for the accumulators below. */
    for (int32_t i = 0; i < c->m; ++i) {
        float* SKAINET_RESTRICT c_row = c->c + (size_t) i * c->c_stride;
        for (int32_t j = j0; j < j1; ++j) c_row[j] = 0.0f;
    }
    if (c->k <= 0) return;

    if (c->m == 1) {
        const float* SKAINET_RESTRICT a_row = c->a;
        float* SKAINET_RESTRICT c_row = c->c + j0;
        for (int32_t p = 0; p < c->k; ++p) {
            const uint8_t* SKAINET_RESTRICT b_row = c->b + (size_t) p * c->b_byte_stride + (size_t) j0 * 2;
            skainet_fp16_stream_axpy(c_row, b_row, a_row[p], j1 - j0);
        }
        return;
    }

    float decoded[SKAINET_FP16_TILE];

    for (int32_t t0 = j0; t0 < j1; t0 += SKAINET_FP16_TILE) {
        const int32_t tile = (j1 - t0) < SKAINET_FP16_TILE ? (j1 - t0) : SKAINET_FP16_TILE;

        for (int32_t p = 0; p < c->k; ++p) {
            const uint8_t* SKAINET_RESTRICT b_row = c->b + (size_t) p * c->b_byte_stride + (size_t) t0 * 2;

            /* Decode this row's tile once, for all m rows of C below. */
            skainet_fp16_widen(decoded, b_row, tile);

            for (int32_t i = 0; i < c->m; ++i) {
                const float a_ip = c->a[(size_t) i * c->a_stride + p];
                float* SKAINET_RESTRICT c_row = c->c + (size_t) i * c->c_stride + t0;
                skainet_f32_axpy(c_row, decoded, a_ip, tile);
            }
        }
    }
}

SKAINET_API void skainet_fp16_matmul(
    const float* SKAINET_RESTRICT a, int32_t a_offset, int32_t a_stride,
    const uint8_t* SKAINET_RESTRICT b, int32_t b_byte_offset, int32_t b_byte_stride,
    float* SKAINET_RESTRICT c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;
    skainet_fp16_ctx ctx;
    ctx.a = a + a_offset;
    ctx.b = b + b_byte_offset;
    ctx.c = c + c_offset;
    ctx.a_stride = a_stride;
    ctx.b_byte_stride = b_byte_stride;
    ctx.c_stride = c_stride;
    ctx.m = m;
    ctx.n = n;
    ctx.k = k < 0 ? 0 : k;
    skainet_run_rows(skainet_fp16_columns, &ctx, n);
}
