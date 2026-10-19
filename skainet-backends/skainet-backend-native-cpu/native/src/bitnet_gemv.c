#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Native `bitnet_gemv`: int8 activations (one absmax scale per token) against
 * ternary weights, matching sk.ainet.backend.api.kernel.BitNetGemvKernel.
 *
 * A ternary weight is -1, 0 or +1 times a block scale, so the arithmetic is an
 * add, a subtract or nothing — no multiplies in the inner loop. What NEON adds
 * is the unpacking: a whole vector of 2-bit codes becomes a vector of int8
 * signs with a shift, a mask and a subtract, after which `sdot` (ARMv8.2
 * dotprod) accumulates 16 products per instruction.
 *
 * Weight layout — canonical GGML TQ2_0, exactly what TernaryCodec writes and
 * what a GGUF holds:
 *   block = 66 bytes: 64 payload + FP16 scale
 *   byte `c*32 + m` of a block holds four elements *32 apart*:
 *     element c*128 + l*32 + m sits in bit pair l
 * That interleave is why the vector path is natural here: for a fixed chunk c
 * and bit pair l, bytes c*32..c*32+31 unpack to elements
 * c*128 + l*32 .. + 31 — thirty-two *consecutive* elements, which line up with
 * thirty-two consecutive activations.
 *
 * BitNet b1.58's own packing (four consecutive elements per byte, one scale for
 * the whole tensor) takes the scalar path here: de-interleaving it costs more
 * than it saves, and a checkpoint that is going to be run through this kernel
 * ships as TQ2_0. TQ1_0's base-3 packing is likewise scalar — it is a storage
 * format, not a runtime one.
 *
 * The Emscripten sidecar has the same block body in SIMD128: the unpacking is
 * identical (shift, mask) and the sixteen int8 products go through
 * `i32x4.dot_i16x8_s` on widened halves, the stand-in for `sdot` every other
 * int8 kernel in the sidecar uses.
 *
 * Output rows are independent, so they are split over the shared row pool
 * (skainet_row_threads.h) when there are enough of them — the mt sidecar's
 * pthreads in a browser, the native pool elsewhere. Per-row arithmetic never
 * changes with the partition, so the result is bit-identical to a single
 * thread's.
 */

#define SKAINET_TQ2_0_BLOCK_SIZE 256
#define SKAINET_TQ2_0_BYTES_PER_BLOCK 66

static inline float skainet_bitnet_fp16(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &bits, sizeof(r));
    return r;
}

/* Scalar reference for one TQ2_0 block: the shape every vector path must match. */
static inline int32_t skainet_tq2_0_block_scalar(
    const uint8_t* SKAINET_RESTRICT qs,
    const int8_t* SKAINET_RESTRICT act,
    int32_t count) {
    int32_t acc = 0;
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t l = 0; l < 4; ++l) {
            for (int32_t m = 0; m < 32; ++m) {
                int32_t element = c * 128 + l * 32 + m;
                if (element >= count) {
                    continue;
                }
                int32_t code = ((qs[c * 32 + m] >> (2 * l)) & 3) - 1;
                if (code > 0) {
                    acc += act[element];
                } else if (code < 0) {
                    acc -= act[element];
                }
            }
        }
    }
    return acc;
}

#if defined(SKAINET_HAVE_NEON)
/*
 * One whole TQ2_0 block (256 elements) with NEON. Two 32-byte chunks × four bit
 * pairs, each producing 32 consecutive weight codes to pair with 32 consecutive
 * activations.
 *
 * Returns the *biased* sum `Σ code·a` with codes still in `{0,1,2}`. Converting
 * them to `{-1,0,+1}` would cost a subtract per strip; instead the caller
 * subtracts `Σ a` once per block, since `Σ (code-1)·a = Σ code·a − Σ a` and the
 * activation sum is the same for every output row.
 */
SKAINET_DOTPROD_TARGET
static inline int32_t skainet_tq2_0_block_neon(
    const uint8_t* SKAINET_RESTRICT qs,
    const int8_t* SKAINET_RESTRICT act) {
    const uint8x16_t mask = vdupq_n_u8(3);
    int32x4_t acc = vdupq_n_s32(0);

    /*
     * `vshrq_n_u8` takes a *compile-time* shift, so the four bit pairs are
     * unrolled rather than looped — which is what one would write by hand
     * anyway: four independent 32-element strips per 32-byte chunk.
     */
/*
 * `vshrq_n_u8` requires a shift in [1, 8] — clang rejects a literal 0, which is
 * why the strips take an already-shifted vector rather than a shift amount, and
 * why strip 0 passes the bytes through untouched.
 */
#if defined(SKAINET_HAVE_DOTPROD) || defined(SKAINET_DOTPROD_DISPATCH)
#define SKAINET_TQ2_STRIP(chunk, strip, lo_bits, hi_bits)                                     \
    do {                                                                                      \
        const int8x16_t wlo = vreinterpretq_s8_u8(vandq_u8((lo_bits), mask));                 \
        const int8x16_t whi = vreinterpretq_s8_u8(vandq_u8((hi_bits), mask));                 \
        const int8_t* a = act + (chunk) * 128 + (strip) * 32;                                  \
        acc = vdotq_s32(acc, wlo, vld1q_s8(a));                                                \
        acc = vdotq_s32(acc, whi, vld1q_s8(a + 16));                                           \
    } while (0)
#else
#define SKAINET_TQ2_STRIP(chunk, strip, lo_bits, hi_bits)                                     \
    do {                                                                                      \
        const int8x16_t wlo = vreinterpretq_s8_u8(vandq_u8((lo_bits), mask));                 \
        const int8x16_t whi = vreinterpretq_s8_u8(vandq_u8((hi_bits), mask));                 \
        const int8_t* a = act + (chunk) * 128 + (strip) * 32;                                  \
        const int8x16_t alo = vld1q_s8(a);                                                     \
        const int8x16_t ahi = vld1q_s8(a + 16);                                                \
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(wlo), vget_low_s8(alo)));                  \
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wlo), vget_high_s8(alo)));                \
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(whi), vget_low_s8(ahi)));                  \
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(whi), vget_high_s8(ahi)));                \
    } while (0)
#endif

    {
        const uint8x16_t lo = vld1q_u8(qs);
        const uint8x16_t hi = vld1q_u8(qs + 16);
        SKAINET_TQ2_STRIP(0, 0, lo, hi);
        SKAINET_TQ2_STRIP(0, 1, vshrq_n_u8(lo, 2), vshrq_n_u8(hi, 2));
        SKAINET_TQ2_STRIP(0, 2, vshrq_n_u8(lo, 4), vshrq_n_u8(hi, 4));
        SKAINET_TQ2_STRIP(0, 3, vshrq_n_u8(lo, 6), vshrq_n_u8(hi, 6));
    }
    {
        const uint8x16_t lo = vld1q_u8(qs + 32);
        const uint8x16_t hi = vld1q_u8(qs + 48);
        SKAINET_TQ2_STRIP(1, 0, lo, hi);
        SKAINET_TQ2_STRIP(1, 1, vshrq_n_u8(lo, 2), vshrq_n_u8(hi, 2));
        SKAINET_TQ2_STRIP(1, 2, vshrq_n_u8(lo, 4), vshrq_n_u8(hi, 4));
        SKAINET_TQ2_STRIP(1, 3, vshrq_n_u8(lo, 6), vshrq_n_u8(hi, 6));
    }
#undef SKAINET_TQ2_STRIP

    return vaddvq_s32(acc);
}
#define skainet_tq2_0_block_simd skainet_tq2_0_block_neon
#elif defined(SKAINET_HAVE_WASM_SIMD128)
/*
 * The NEON block in SIMD128: the same two chunks × four bit pairs, codes kept in
 * {0,1,2} with the Σ a bias removed by the caller. `wasm_u8x16_shr` takes a
 * runtime shift, so the strips can loop over the bit pair.
 */
static inline int32_t skainet_tq2_0_block_wasm(
    const uint8_t* SKAINET_RESTRICT qs,
    const int8_t* SKAINET_RESTRICT act) {
    const v128_t mask = wasm_u8x16_splat(3);
    v128_t acc = wasm_i32x4_splat(0);
    for (int32_t c = 0; c < 2; ++c) {
        const v128_t lo = wasm_v128_load(qs + c * 32);
        const v128_t hi = wasm_v128_load(qs + c * 32 + 16);
        for (int32_t l = 0; l < 4; ++l) {
            const v128_t wlo = wasm_v128_and(wasm_u8x16_shr(lo, (uint32_t) (2 * l)), mask);
            const v128_t whi = wasm_v128_and(wasm_u8x16_shr(hi, (uint32_t) (2 * l)), mask);
            const int8_t* a = act + c * 128 + l * 32;
            acc = wasm_i32x4_add(acc, skainet_wasm_dot_u8x16_i8x16(wlo, wasm_v128_load(a)));
            acc = wasm_i32x4_add(acc, skainet_wasm_dot_u8x16_i8x16(whi, wasm_v128_load(a + 16)));
        }
    }
    return skainet_wasm_hadd_i32(acc);
}
#define skainet_tq2_0_block_simd skainet_tq2_0_block_wasm
#endif /* SKAINET_HAVE_NEON / SKAINET_HAVE_WASM_SIMD128 */

typedef struct {
    const int8_t* act;
    const uint8_t* w;
    const int32_t* activation_sums;
    float* out;
    float activation_scale;
    int32_t input_dim;
    int32_t blocks_per_row;
} skainet_bitnet_gemv_job;

/* Output rows [o_start, o_end): every block of the row, scaled and summed. */
static void skainet_bitnet_gemv_rows(void* ctx, int32_t o_start, int32_t o_end) {
    const skainet_bitnet_gemv_job* j = (const skainet_bitnet_gemv_job*) ctx;
    const int8_t* act = j->act;
    const int32_t input_dim = j->input_dim;
    const int32_t blocks_per_row = j->blocks_per_row;
    for (int32_t o = o_start; o < o_end; ++o) {
        float sum = 0.0f;
        for (int32_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block = j->w + (size_t)(o * blocks_per_row + b) * SKAINET_TQ2_0_BYTES_PER_BLOCK;
            const int32_t first = b * SKAINET_TQ2_0_BLOCK_SIZE;
            const int32_t count = (input_dim - first) < SKAINET_TQ2_0_BLOCK_SIZE
                ? (input_dim - first)
                : SKAINET_TQ2_0_BLOCK_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block + 64, sizeof(scale_bits));
            const float d = skainet_bitnet_fp16(scale_bits);
            int32_t partial;
#ifdef skainet_tq2_0_block_simd
            if (count == SKAINET_TQ2_0_BLOCK_SIZE) {
                partial = skainet_tq2_0_block_simd(block, act + first) - j->activation_sums[b];
            } else {
                partial = skainet_tq2_0_block_scalar(block, act + first, count);
            }
#else
            partial = skainet_tq2_0_block_scalar(block, act + first, count);
#endif
            sum += (float)partial * d;
        }
        j->out[o] = sum * j->activation_scale;
    }
}

SKAINET_API void skainet_bitnet_gemv_tq2_0(
    const int8_t* SKAINET_RESTRICT activation, int32_t activation_offset,
    float activation_scale,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset) {
    if (input_dim <= 0 || output_dim <= 0) {
        return;
    }
    const int8_t* act = activation + activation_offset;
    const uint8_t* w = weight + weight_byte_offset;
    const int32_t blocks_per_row = (input_dim + SKAINET_TQ2_0_BLOCK_SIZE - 1) / SKAINET_TQ2_0_BLOCK_SIZE;

    /*
     * Σ activation per block, computed once for the whole call: the vector path
     * accumulates codes in {0,1,2} and removes the bias here rather than paying a
     * subtract per 16 weights, per output row.
     */
    enum { SKAINET_MAX_STACK_BLOCKS = 128 };
    int32_t stack_sums[SKAINET_MAX_STACK_BLOCKS];
    int32_t* activation_sums = stack_sums;
    int32_t* heap_sums = NULL;
    if (blocks_per_row > SKAINET_MAX_STACK_BLOCKS) {
        heap_sums = (int32_t*)malloc((size_t)blocks_per_row * sizeof(int32_t));
        if (heap_sums == NULL) {
            return;
        }
        activation_sums = heap_sums;
    }
    for (int32_t b = 0; b < blocks_per_row; ++b) {
        const int32_t first = b * SKAINET_TQ2_0_BLOCK_SIZE;
        const int32_t count = (input_dim - first) < SKAINET_TQ2_0_BLOCK_SIZE
            ? (input_dim - first)
            : SKAINET_TQ2_0_BLOCK_SIZE;
        int32_t s = 0;
        for (int32_t i = 0; i < count; ++i) {
            s += act[first + i];
        }
        activation_sums[b] = s;
    }

    skainet_bitnet_gemv_job job;
    job.act = act;
    job.w = w;
    job.activation_sums = activation_sums;
    job.out = output + output_offset;
    job.activation_scale = activation_scale;
    job.input_dim = input_dim;
    job.blocks_per_row = blocks_per_row;
    skainet_run_rows(skainet_bitnet_gemv_rows, &job, output_dim);
    free(heap_sums);
}
