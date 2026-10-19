/*
 * Adapter over the vendored NeoGPU ternary LUT kernel
 * (src/vendor/neogpu/hs_ml_ternary_neon.c, MIT, vendored verbatim — see the
 * vendor README). Exposes the two useful symbols under the skainet_kernels
 * ABI, threads the projection on the shared row pool, and closes the vendored file's non-atomic LUT-init guard with a
 * pthread_once warm-up, so first use is race-free under any threading.
 *
 * The vendored file needs pthreads and GNU/Clang builtins, so it is only in
 * the build where SKAINET_HAVE_NEOGPU_TERNARY is defined (non-MSVC native
 * builds). The other branch below is a portable mirror with identical
 * semantics, including the byte-code-3 → +2.0 decode (loaders reject code 3
 * at import; the kernel contract is "garbage in, garbage out", never a
 * crash). It is what MSVC and the Emscripten sidecar compile: the same
 * 256-entry LUT as a compile-time constant (no init guard to race on), a
 * SIMD128 body in the sidecar, and the shared row pool for threading.
 */

#include "skainet_kernels.h"

#include <string.h>

#ifdef SKAINET_HAVE_NEOGPU_TERNARY

#include <pthread.h>

#include "skainet_row_threads.h"

/* The vendored file ships no header; these mirror its public definitions. */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                   const uint8_t *W, uint32_t N, uint32_t K);
extern void hs_ml_lmhead_stage1(float *out, const float *in,
                                const uint8_t *P0, const uint8_t *P1,
                                const uint8_t *P2, const uint8_t *P3,
                                const uint16_t *row_scale,
                                uint32_t N, uint32_t K);

static pthread_once_t skainet_ternary_lut_once = PTHREAD_ONCE_INIT;

static void skainet_ternary_lut_warmup(void) {
    /* Any call builds the 256-entry LUT; N=1 stays on the calling thread
     * (far below the vendored THREAD_THRESHOLD of 512). */
    static const float in[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    static const uint8_t w[1] = { 0 };
    float out;
    hs_ml_ternary_f32_proj(&out, in, w, 1u, 4u);
}

/* Row-pool threading for the projection (SKaiNET#1357). Called as is, the vendored
 * proj spawns and joins 4 pthreads per call once N >= 512 — ~210 calls per BitNet
 * 2B4T decode step, each paying thread-creation and wake-up latency (the exact
 * create/join pattern skainet_row_threads.c measured at 994 ms/step vs pooled, #1195).
 * Instead the adapter drives the vendored kernel from the persistent row pool and
 * hands it row slices below its THREAD_THRESHOLD, so every slice runs inline on the
 * pool thread. Per-row results do not depend on the slicing — bit-identical. */
#define SKAINET_NEOGPU_INLINE_ROWS 511

typedef struct {
    float* out;
    const float* in;
    const uint8_t* w;
    uint32_t row_bytes;
    uint32_t k;
} skainet_ternary_proj_job;

static void skainet_ternary_proj_rows(void* ctx, int32_t o_start, int32_t o_end) {
    const skainet_ternary_proj_job* j = (const skainet_ternary_proj_job*)ctx;
    for (int32_t o = o_start; o < o_end; o += SKAINET_NEOGPU_INLINE_ROWS) {
        int32_t n = o_end - o < SKAINET_NEOGPU_INLINE_ROWS ? o_end - o : SKAINET_NEOGPU_INLINE_ROWS;
        hs_ml_ternary_f32_proj(j->out + o, j->in, j->w + (size_t)o * j->row_bytes, (uint32_t)n, j->k);
    }
}

void skainet_ternary_f32_gemv(
    const float* input, int32_t input_offset,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    pthread_once(&skainet_ternary_lut_once, skainet_ternary_lut_warmup);
    /* input_dim == 0 falls through: the vendored kernel writes 0.0f per row. */
    skainet_ternary_proj_job job = {
        output + output_offset,
        input + input_offset,
        weight + weight_byte_offset,
        (uint32_t)(input_dim / 4),
        (uint32_t)input_dim,
    };
    skainet_run_rows(skainet_ternary_proj_rows, &job, output_dim);
}

void skainet_ternary_lmhead_stage1(
    const float* input, int32_t input_offset,
    const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
    const uint16_t* row_scale, int32_t row_scale_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    pthread_once(&skainet_ternary_lut_once, skainet_ternary_lut_warmup);
    const uint8_t* base = planes + planes_byte_offset;
    hs_ml_lmhead_stage1(
        output + output_offset,
        input + input_offset,
        base,
        base + (size_t)plane_stride_bytes,
        base + (size_t)plane_stride_bytes * 2,
        base + (size_t)plane_stride_bytes * 3,
        row_scale + row_scale_offset,
        (uint32_t)output_dim,
        (uint32_t)input_dim);
}

#else /* !SKAINET_HAVE_NEOGPU_TERNARY — portable mirror (MSVC, Emscripten) */

#include "skainet_simd.h"
#include "skainet_row_threads.h"

/*
 * decode(byte, lane) = ((byte >> (lane*2)) & 3) - 1 — exactly the vendored
 * LUT (so byte code 3 yields +2, matching TernaryCodec.decodeBitNet), as a
 * compile-time constant: one 16-byte row per weight byte, the four decoded
 * codes of its four elements. The vendored file builds the same table on
 * first use behind a non-atomic flag; a constant needs no guard, which is
 * what makes the row pool below safe without pthread_once.
 */
#define SKAINET_TERNARY_D(b, lane) ((float) ((int) (((b) >> ((lane) * 2)) & 3) - 1))
#define SKAINET_TERNARY_E(b) \
    { SKAINET_TERNARY_D(b, 0), SKAINET_TERNARY_D(b, 1), SKAINET_TERNARY_D(b, 2), SKAINET_TERNARY_D(b, 3) }
#define SKAINET_TERNARY_E16(b) \
    SKAINET_TERNARY_E(b + 0),  SKAINET_TERNARY_E(b + 1),  SKAINET_TERNARY_E(b + 2),  SKAINET_TERNARY_E(b + 3), \
    SKAINET_TERNARY_E(b + 4),  SKAINET_TERNARY_E(b + 5),  SKAINET_TERNARY_E(b + 6),  SKAINET_TERNARY_E(b + 7), \
    SKAINET_TERNARY_E(b + 8),  SKAINET_TERNARY_E(b + 9),  SKAINET_TERNARY_E(b + 10), SKAINET_TERNARY_E(b + 11), \
    SKAINET_TERNARY_E(b + 12), SKAINET_TERNARY_E(b + 13), SKAINET_TERNARY_E(b + 14), SKAINET_TERNARY_E(b + 15)

static const float skainet_ternary_lut[256][4] = {
    SKAINET_TERNARY_E16(0),   SKAINET_TERNARY_E16(16),  SKAINET_TERNARY_E16(32),  SKAINET_TERNARY_E16(48),
    SKAINET_TERNARY_E16(64),  SKAINET_TERNARY_E16(80),  SKAINET_TERNARY_E16(96),  SKAINET_TERNARY_E16(112),
    SKAINET_TERNARY_E16(128), SKAINET_TERNARY_E16(144), SKAINET_TERNARY_E16(160), SKAINET_TERNARY_E16(176),
    SKAINET_TERNARY_E16(192), SKAINET_TERNARY_E16(208), SKAINET_TERNARY_E16(224), SKAINET_TERNARY_E16(240),
};

#undef SKAINET_TERNARY_E16
#undef SKAINET_TERNARY_E
#undef SKAINET_TERNARY_D

/* FP16 row scale decode, same bit arithmetic as the vendored file (sign
 * ignored — encoders store max|row| >= 0). */
static inline float skainet_ternary_row_scale(uint16_t h16) {
    const uint32_t eu = (h16 >> 10) & 0x1Fu;
    const uint32_t mu = h16 & 0x03FFu;
    const uint32_t fu = ((eu + 112u) << 23) | (mu << 13);
    float rsc;
    memcpy(&rsc, &fu, 4);
    return rsc;
}

typedef struct {
    float* out;
    const float* in;
    const uint8_t* w;
    int32_t row_bytes;
} skainet_ternary_gemv_job;

/* out[o] = Σ_bi lut[w[o][bi]] · in[bi*4 .. bi*4+3] for rows [o_start, o_end). */
static void skainet_ternary_gemv_rows(void* ctx, int32_t o_start, int32_t o_end) {
    const skainet_ternary_gemv_job* j = (const skainet_ternary_gemv_job*) ctx;
    const float* in = j->in;
    const int32_t row_bytes = j->row_bytes;
    for (int32_t o = o_start; o < o_end; ++o) {
        const uint8_t* wrow = j->w + (size_t) o * row_bytes;
#ifdef SKAINET_HAVE_WASM_SIMD128
        /* One LUT row is one f32x4 of decoded codes against four activations:
         * a load, a multiply and an add per weight byte. Four accumulators
         * keep the adds independent; the vendored NEON body does the same. */
        v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        int32_t bi = 0;
        for (; bi + 4 <= row_bytes; bi += 4) {
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[wrow[bi]]), wasm_v128_load(in + (size_t) bi * 4)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[wrow[bi + 1]]), wasm_v128_load(in + (size_t) bi * 4 + 4)));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[wrow[bi + 2]]), wasm_v128_load(in + (size_t) bi * 4 + 8)));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[wrow[bi + 3]]), wasm_v128_load(in + (size_t) bi * 4 + 12)));
        }
        for (; bi < row_bytes; ++bi) {
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[wrow[bi]]), wasm_v128_load(in + (size_t) bi * 4)));
        }
        j->out[o] = skainet_wasm_hadd_f32(wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3)));
#else
        float acc = 0.0f;
        for (int32_t bi = 0; bi < row_bytes; ++bi) {
            const float* d = skainet_ternary_lut[wrow[bi]];
            acc += d[0] * in[bi * 4    ];
            acc += d[1] * in[bi * 4 + 1];
            acc += d[2] * in[bi * 4 + 2];
            acc += d[3] * in[bi * 4 + 3];
        }
        j->out[o] = acc;
#endif
    }
}

void skainet_ternary_f32_gemv(
    const float* input, int32_t input_offset,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    skainet_ternary_gemv_job job = {
        output + output_offset,
        input + input_offset,
        weight + weight_byte_offset,
        input_dim / 4,
    };
    /* input_dim == 0: row_bytes 0, every row is written as 0.0f. */
    skainet_run_rows(skainet_ternary_gemv_rows, &job, output_dim);
}

typedef struct {
    float* out;
    const float* in;
    const uint8_t* p0;
    const uint8_t* p1;
    const uint8_t* p2;
    const uint8_t* p3;
    const uint16_t* row_scale;
    int32_t row_bytes;
} skainet_ternary_lmhead_job;

/* out[o] = rs[o] · Σ_bi (lut[p0] + lut[p1]/3 + lut[p2]/9 + lut[p3]/27) · in[bi*4 ..]. */
static void skainet_ternary_lmhead_rows(void* ctx, int32_t o_start, int32_t o_end) {
    const skainet_ternary_lmhead_job* j = (const skainet_ternary_lmhead_job*) ctx;
    const float* in = j->in;
    const int32_t row_bytes = j->row_bytes;
    const float w1 = 1.0f / 3.0f, w2 = 1.0f / 9.0f, w3 = 1.0f / 27.0f;
    for (int32_t o = o_start; o < o_end; ++o) {
        const uint8_t* r0 = j->p0 + (size_t) o * row_bytes;
        const uint8_t* r1 = j->p1 + (size_t) o * row_bytes;
        const uint8_t* r2 = j->p2 + (size_t) o * row_bytes;
        const uint8_t* r3 = j->p3 + (size_t) o * row_bytes;
#ifdef SKAINET_HAVE_WASM_SIMD128
        const v128_t vw1 = wasm_f32x4_splat(w1), vw2 = wasm_f32x4_splat(w2), vw3 = wasm_f32x4_splat(w3);
        v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = acc0;
        int32_t bi = 0;
        for (; bi + 2 <= row_bytes; bi += 2) {
            v128_t u0 = wasm_v128_load(skainet_ternary_lut[r0[bi]]);
            u0 = wasm_f32x4_add(u0, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r1[bi]]), vw1));
            u0 = wasm_f32x4_add(u0, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r2[bi]]), vw2));
            u0 = wasm_f32x4_add(u0, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r3[bi]]), vw3));
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(u0, wasm_v128_load(in + (size_t) bi * 4)));
            v128_t u1 = wasm_v128_load(skainet_ternary_lut[r0[bi + 1]]);
            u1 = wasm_f32x4_add(u1, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r1[bi + 1]]), vw1));
            u1 = wasm_f32x4_add(u1, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r2[bi + 1]]), vw2));
            u1 = wasm_f32x4_add(u1, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r3[bi + 1]]), vw3));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(u1, wasm_v128_load(in + (size_t) bi * 4 + 4)));
        }
        for (; bi < row_bytes; ++bi) {
            v128_t u = wasm_v128_load(skainet_ternary_lut[r0[bi]]);
            u = wasm_f32x4_add(u, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r1[bi]]), vw1));
            u = wasm_f32x4_add(u, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r2[bi]]), vw2));
            u = wasm_f32x4_add(u, wasm_f32x4_mul(wasm_v128_load(skainet_ternary_lut[r3[bi]]), vw3));
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(u, wasm_v128_load(in + (size_t) bi * 4)));
        }
        const float s = skainet_wasm_hadd_f32(wasm_f32x4_add(acc0, acc1));
#else
        float s = 0.0f;
        for (int32_t bi = 0; bi < row_bytes; ++bi) {
            const float* d0 = skainet_ternary_lut[r0[bi]];
            const float* d1 = skainet_ternary_lut[r1[bi]];
            const float* d2 = skainet_ternary_lut[r2[bi]];
            const float* d3 = skainet_ternary_lut[r3[bi]];
            for (int lane = 0; lane < 4; ++lane) {
                const float u = d0[lane] + d1[lane] * w1 + d2[lane] * w2 + d3[lane] * w3;
                s += u * in[bi * 4 + lane];
            }
        }
#endif
        j->out[o] = s * skainet_ternary_row_scale(j->row_scale[o]);
    }
}

void skainet_ternary_lmhead_stage1(
    const float* input, int32_t input_offset,
    const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
    const uint16_t* row_scale, int32_t row_scale_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    const uint8_t* base = planes + planes_byte_offset;
    skainet_ternary_lmhead_job job = {
        output + output_offset,
        input + input_offset,
        base,
        base + (size_t) plane_stride_bytes,
        base + (size_t) plane_stride_bytes * 2,
        base + (size_t) plane_stride_bytes * 3,
        row_scale + row_scale_offset,
        input_dim / 4,
    };
    skainet_run_rows(skainet_ternary_lmhead_rows, &job, output_dim);
}

#endif /* SKAINET_HAVE_NEOGPU_TERNARY */
