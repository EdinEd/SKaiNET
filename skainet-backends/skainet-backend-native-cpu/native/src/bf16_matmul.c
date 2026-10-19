#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native row-major FP32 × BF16 matmul matching the
 * sk.ainet.backend.api.kernel.Bf16MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 *   A, C : FP32 (FloatArray on the JVM side; float* here).
 *          Strides in floats. `a_stride == k` for a contiguous parent.
 *   B    : packed BF16 (ByteArray on the JVM side; uint8_t* here).
 *          Strides in *bytes*. `b_byte_stride == n * 2` for a contiguous
 *          parent. Each BF16 value is 2 bytes little-endian.
 *
 * BF16 → FP32 conversion is the bit-shift identity
 *   float_bits = ((uint32_t) bf16_bits) << 16
 * (BF16 shares the FP32 sign and exponent layout; only the trailing 16
 *  mantissa bits are discarded).
 *
 * Iteration order depends on m.
 *
 * At m == 1 it is plain i-p-j (outer-product into the single row of C). The
 * inner `c[j] += a_ip * bf16_to_float(b[j])` loop streams two contiguous
 * arrays. Every B element is used exactly once, so there is nothing to reuse
 * and sequential streaming is the right shape.
 *
 * At m > 1, i-p-j walks the whole of B once per row of A. For ffn_up 8B at
 * m=16 that is 16 passes over 90 MiB, and the kernel is bandwidth-bound long
 * before it is dequant-bound — the shift itself is nearly free. So j is tiled,
 * and within a tile each B row is widened once into a small stack buffer and
 * multiplied into all m rows of C. B is then read once in total rather than m
 * times. The tile is sized so the widened row and the m C rows it feeds stay
 * resident together.
 *
 * SIMD bodies (NEON on AArch64, SIMD128 in the Emscripten sidecar) widen eight
 * BF16 values per step — unpack to 32-bit lanes, shift left 16 — and
 * multiply-add four lanes at a time, both on the m == 1 stream and on the
 * tile; the portable body is the scalar form of the same loops.
 *
 * Accumulation order into any given C element is p ascending on every path,
 * so they are bit-identical to each other and to the original i-p-j
 * formulation, not merely close (the NEON path uses a fused multiply-add,
 * as the auto-vectorised portable body does under -ffast-math on AArch64).
 *
 * Threading (#1195 pattern): the columns of C are split into disjoint ranges
 * through skainet_row_threads.h when n is large enough; each worker runs the
 * whole m × k work for its columns, so no accumulation order changes. On
 * Emscripten that is the pthread pool of the mt sidecar.
 *
 * On ARMv8.6-A+ a future pass can swap the shift for a `bfdot`/`bfmmla`
 * intrinsic kernel; that lives behind a runtime feature check and is out of
 * scope here.
 *
 * Caller contract (mirrors skainet_fp32_matmul):
 *  - C is FULLY OVERWRITTEN in the m×n block.
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; defensively treated as no-op.
 */
/*
 * Columns widened per pass. 512 floats is a 2 KiB stack buffer — small enough
 * to leave the C rows it feeds resident alongside it, large enough that the
 * per-tile loop overhead disappears against the k*m inner work.
 */
#define SKAINET_BF16_TILE 512

static inline float skainet_bf16_to_float(const uint8_t* SKAINET_RESTRICT src) {
    /* Read 2 bytes LE. memcpy is strict-aliasing safe and the compiler folds
     * it to a single 16-bit load. */
    uint16_t bits;
    memcpy(&bits, src, sizeof(uint16_t));
    const uint32_t fp32_bits = ((uint32_t) bits) << 16;
    float f;
    memcpy(&f, &fp32_bits, sizeof(float));
    return f;
}

/*
 * c[0..count) += a_ip * bf16(src[0..count)) — the m == 1 stream; and
 * dst[0..count) = bf16(src[0..count)) — the tile widen. Eight values per SIMD
 * step, scalar tail.
 */
static inline void skainet_bf16_stream_axpy(
    float* SKAINET_RESTRICT c, const uint8_t* SKAINET_RESTRICT src, float a_ip, int32_t count
) {
    int32_t j = 0;
#if defined(SKAINET_HAVE_NEON)
    const float32x4_t va = vdupq_n_f32(a_ip);
    for (; j + 8 <= count; j += 8) {
        const uint16x8_t v = vreinterpretq_u16_u8(vld1q_u8(src + (size_t) j * 2));
        const float32x4_t lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16));
        const float32x4_t hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16));
        vst1q_f32(c + j, vfmaq_f32(vld1q_f32(c + j), va, lo));
        vst1q_f32(c + j + 4, vfmaq_f32(vld1q_f32(c + j + 4), va, hi));
    }
#elif defined(SKAINET_HAVE_WASM_SIMD128)
    const v128_t va = wasm_f32x4_splat(a_ip);
    for (; j + 8 <= count; j += 8) {
        const v128_t v = wasm_v128_load(src + (size_t) j * 2);
        const v128_t lo = wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(v), 16);
        const v128_t hi = wasm_i32x4_shl(wasm_u32x4_extend_high_u16x8(v), 16);
        wasm_v128_store(c + j, wasm_f32x4_add(wasm_v128_load(c + j), wasm_f32x4_mul(va, lo)));
        wasm_v128_store(c + j + 4, wasm_f32x4_add(wasm_v128_load(c + j + 4), wasm_f32x4_mul(va, hi)));
    }
#endif
    for (; j < count; ++j) c[j] += a_ip * skainet_bf16_to_float(src + (size_t) j * 2);
}

static inline void skainet_bf16_widen(
    float* SKAINET_RESTRICT dst, const uint8_t* SKAINET_RESTRICT src, int32_t count
) {
    int32_t j = 0;
#if defined(SKAINET_HAVE_NEON)
    for (; j + 8 <= count; j += 8) {
        const uint16x8_t v = vreinterpretq_u16_u8(vld1q_u8(src + (size_t) j * 2));
        vst1q_f32(dst + j, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16)));
        vst1q_f32(dst + j + 4, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16)));
    }
#elif defined(SKAINET_HAVE_WASM_SIMD128)
    for (; j + 8 <= count; j += 8) {
        const v128_t v = wasm_v128_load(src + (size_t) j * 2);
        wasm_v128_store(dst + j, wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(v), 16));
        wasm_v128_store(dst + j + 4, wasm_i32x4_shl(wasm_u32x4_extend_high_u16x8(v), 16));
    }
#endif
    for (; j < count; ++j) dst[j] = skainet_bf16_to_float(src + (size_t) j * 2);
}

/* c[0..count) += a_ip * w[0..count) on the widened tile. */
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
} skainet_bf16_ctx;

/* Columns [j0, j1) of C, all rows. */
static void skainet_bf16_columns(void* vctx, int32_t j0, int32_t j1) {
    const skainet_bf16_ctx* c = (const skainet_bf16_ctx*) vctx;

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
            skainet_bf16_stream_axpy(c_row, b_row, a_row[p], j1 - j0);
        }
        return;
    }

    float widened[SKAINET_BF16_TILE];

    for (int32_t t0 = j0; t0 < j1; t0 += SKAINET_BF16_TILE) {
        const int32_t tile = (j1 - t0) < SKAINET_BF16_TILE ? (j1 - t0) : SKAINET_BF16_TILE;

        for (int32_t p = 0; p < c->k; ++p) {
            const uint8_t* SKAINET_RESTRICT b_row = c->b + (size_t) p * c->b_byte_stride + (size_t) t0 * 2;

            /* Widen this row's tile once, for all m rows of C below. */
            skainet_bf16_widen(widened, b_row, tile);

            for (int32_t i = 0; i < c->m; ++i) {
                const float a_ip = c->a[(size_t) i * c->a_stride + p];
                float* SKAINET_RESTRICT c_row = c->c + (size_t) i * c->c_stride + t0;
                skainet_f32_axpy(c_row, widened, a_ip, tile);
            }
        }
    }
}

SKAINET_API void skainet_bf16_matmul(
    const float* SKAINET_RESTRICT a, int32_t a_offset, int32_t a_stride,
    const uint8_t* SKAINET_RESTRICT b, int32_t b_byte_offset, int32_t b_byte_stride,
    float* SKAINET_RESTRICT c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;
    skainet_bf16_ctx ctx;
    ctx.a = a + a_offset;
    ctx.b = b + b_byte_offset;
    ctx.c = c + c_offset;
    ctx.a_stride = a_stride;
    ctx.b_byte_stride = b_byte_stride;
    ctx.c_stride = c_stride;
    ctx.m = m;
    ctx.n = n;
    ctx.k = k < 0 ? 0 : k;
    skainet_run_rows(skainet_bf16_columns, &ctx, n);
}
