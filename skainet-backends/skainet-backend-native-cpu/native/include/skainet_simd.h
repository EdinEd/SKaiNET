#ifndef SKAINET_SIMD_H
#define SKAINET_SIMD_H

/*
 * Compile-time SIMD capability detection for the native CPU kernels.
 *
 * The kernels keep their portable scalar bodies as the `#else` fallback,
 * so x86_64 (which auto-vectorizes well under -O3 -ffast-math) and any
 * pre-ARMv8.2 target keep compiling unchanged. The NEON paths are only
 * taken when the compiler advertises `__ARM_NEON` (AArch64 always does
 * with the right -march). `__ARM_FEATURE_DOTPROD` / `__ARM_FEATURE_MATMUL_INT8`
 * are gated on the build flags (`-march=armv8.2-a+dotprod`, etc.).
 *
 * AARCH64-VERIFIED (2026-07-02): the NEON paths (fp32 vfmaq_f32, q4k
 * vdotq_s32 dotprod, q5k, q6k, q8_0) were cross-built with
 * `-march=armv8.2-a+fp16+dotprod` (aarch64 gcc 8.3, the K/N-bundled
 * toolchain) and parity-checked against the commonMain scalar references:
 *   - under qemu-aarch64 via
 *       ./gradlew :skainet-backends:skainet-backend-native-cpu:linuxArm64Test -PcrossArm64=true
 *   - AND on the physical SL2610 board (Cortex-A55, aarch64): the same
 *     test.kexe run natively on-device — 23/23 tests green, no SIGILL.
 *     `/proc/cpuinfo` confirmed asimddp + fphp/asimdhp present and i8mm
 *     absent, matching the chosen -march (no +i8mm).
 * The linked archive was confirmed to contain udot/sdot + fmla, i.e. the
 * SIMD paths — not the scalar fallback — executed. bf16 and fp16 have no
 * NEON path (scalar only).
 *
 * q4_0 gained a plain-NEON body (nibble unpack + widen + vfmaq_f32, no
 * dotprod requirement) later — parity-checked under qemu-aarch64 via the
 * same linuxArm64Test lane (#920).
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define SKAINET_HAVE_NEON 1
#endif
/* WebAssembly SIMD128 (SKEEP-007 Tier 1): the browser sidecar is built with
 * `emcc -msimd128`, which defines __wasm_simd128__. Kernels that have an explicit
 * SIMD128 body select it with `#elif defined(__wasm_simd128__)` between their NEON
 * and portable branches. Every matmul kernel in the sidecar build has one
 * (q8_0, q4_0, q4k, q5k, q6k, q5_0, q5_1, fp32, fp32_wt, bf16, fp16); the helpers
 * below are the SIMD128 counterparts of the NEON ones above. */
#if defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#  define SKAINET_HAVE_WASM_SIMD128 1
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#  define SKAINET_HAVE_DOTPROD 1
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
#  define SKAINET_HAVE_I8MM 1
#endif

/*
 * Apple arm64 runtime dotprod dispatch (#920). Apple archives are compiled at
 * the SDK-default baseline: a Kotlin/Native klib embeds exactly ONE static
 * archive, and Apple A12 (iPhone XS/XR, still iOS-supported) lacks
 * FEAT_DotProd while A13+/M-series have it — so `-march=...+dotprod` at the
 * TU level is not shippable there. Instead the dotprod hot bodies in
 * q4k/q6k are compiled twice: a baseline copy and a copy carrying
 * `__attribute__((target("dotprod")))` (which is what gates vdotq_s32
 * codegen in clang; attributed functions are never inlined into baseline
 * callers, keeping sdot out of the baseline path). Selection happens once
 * per matmul call via the cached sysctl probe in skainet_cpu_features.h.
 *
 * Non-Apple builds keep the compile-time guard as the ONLY mechanism:
 * SKAINET_DOTPROD_DISPATCH stays undefined and the TU-level -march carries
 * the feature, so Linux codegen is unchanged.
 */
#if defined(__APPLE__) && defined(SKAINET_HAVE_NEON) && !defined(SKAINET_HAVE_DOTPROD)
#  define SKAINET_DOTPROD_DISPATCH 1
#  define SKAINET_DOTPROD_TARGET __attribute__((target("dotprod")))
#else
#  define SKAINET_DOTPROD_TARGET /* TU-level -march carries the feature */
#endif

#ifdef SKAINET_HAVE_NEON
/* Horizontal sum of a float32x4 lane vector. AArch64 has vaddvq_f32
 * natively; this wrapper keeps call sites readable. */
static inline float skainet_neon_hadd_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

/* Widen 16 unsigned bytes to four float32x4 lanes (out[0]=lanes 0..3, …). */
static inline void skainet_neon_u8x16_to_f32x4x4(uint8x16_t v, float32x4_t out[4]) {
    const uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
    const uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
    out[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
    out[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
}
#endif

#ifdef SKAINET_HAVE_WASM_SIMD128
/* Horizontal sum of an f32x4 lane vector. */
static inline float skainet_wasm_hadd_f32(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)
         + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/* Horizontal sum of an i32x4 lane vector. */
static inline int32_t skainet_wasm_hadd_i32(v128_t v) {
    return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1)
         + wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

/*
 * 16 × (int8 · int8) → i32x4 partial sums: the SIMD128 stand-in for NEON's
 * vdotq_s32. Both operands widen to i16x8 halves and `i32x4.dot_i16x8_s`
 * multiplies lane pairs and adds them, so each i32 lane holds four products.
 * Exact for any int8 inputs (|product| ≤ 2^14, four of them fit an i32).
 */
static inline v128_t skainet_wasm_dot_i8x16(v128_t a, v128_t b) {
    const v128_t lo = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a),
                                           wasm_i16x8_extend_low_i8x16(b));
    const v128_t hi = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a),
                                           wasm_i16x8_extend_high_i8x16(b));
    return wasm_i32x4_add(lo, hi);
}

/* Same for an UNSIGNED 8-bit left operand (quant codes 0..31) against int8. */
static inline v128_t skainet_wasm_dot_u8x16_i8x16(v128_t codes, v128_t a) {
    const v128_t lo = wasm_i32x4_dot_i16x8(wasm_u16x8_extend_low_u8x16(codes),
                                           wasm_i16x8_extend_low_i8x16(a));
    const v128_t hi = wasm_i32x4_dot_i16x8(wasm_u16x8_extend_high_u8x16(codes),
                                           wasm_i16x8_extend_high_i8x16(a));
    return wasm_i32x4_add(lo, hi);
}

/* Σ of 16 int8 lanes as i32x4 partial sums (pairwise widening adds). */
static inline v128_t skainet_wasm_sum_i8x16(v128_t a) {
    return wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extadd_pairwise_i8x16(a));
}

/* Widen 16 unsigned bytes to four f32x4 lanes (out[0] = lanes 0..3, …). */
static inline void skainet_wasm_u8x16_to_f32x4x4(v128_t v, v128_t out[4]) {
    const v128_t lo16 = wasm_u16x8_extend_low_u8x16(v);
    const v128_t hi16 = wasm_u16x8_extend_high_u8x16(v);
    out[0] = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(lo16));
    out[1] = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(lo16));
    out[2] = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(hi16));
    out[3] = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(hi16));
}

/*
 * Expand the 32-bit high-bit plane of a Q5_0/Q5_1 block into two u8x16 masks
 * holding 0x10 where element j (lo: 0..15, hi: 16..31) has its fifth bit set:
 * broadcast each `qh` byte to its eight lanes with a swizzle, test the per-lane
 * bit and turn the compare mask into the code value 16.
 */
static inline void skainet_wasm_q5_fifth_bits(uint32_t qh32, v128_t* fifth_lo, v128_t* fifth_hi) {
    const v128_t bytes = wasm_i32x4_splat((int32_t) qh32);
    const v128_t idx_lo = wasm_i8x16_const(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const v128_t idx_hi = wasm_i8x16_const(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const v128_t bitmask = wasm_i8x16_const(1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80,
                                            1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80);
    const v128_t sixteen = wasm_i8x16_splat(0x10);
    const v128_t zero = wasm_i8x16_splat(0);
    const v128_t lo = wasm_v128_and(wasm_i8x16_swizzle(bytes, idx_lo), bitmask);
    const v128_t hi = wasm_v128_and(wasm_i8x16_swizzle(bytes, idx_hi), bitmask);
    *fifth_lo = wasm_v128_and(wasm_i8x16_ne(lo, zero), sixteen);
    *fifth_hi = wasm_v128_and(wasm_i8x16_ne(hi, zero), sixteen);
}
#endif

/*
 * Transcendentals called from element loops go through these opaque wrappers. GCC with
 * -ffast-math and glibc's math-vector.h otherwise vectorises such a loop into libmvec's
 * _ZGVbN4v_expf / _ZGVbN4v_tanhf / _ZGVbN2v_erf, which the shared library is not linked
 * against (and which older glibc and aarch64 builds lack): the symbol fails to resolve at
 * the first call and the loader terminates the process. Under -ffast-math GCC also narrows
 * (float) erf((double) x) to erff. A non-inlined scalar call per element costs nothing next
 * to the function itself; clang and MSVC never emit the vector variants, so there the
 * wrappers inline away.
 */
#include <math.h>
#if defined(__GNUC__) && !defined(__clang__)
#  define SKAINET_LIBM_FN static __attribute__((noinline, unused))
#else
#  define SKAINET_LIBM_FN static inline
#endif
SKAINET_LIBM_FN float skainet_expf(float x) { return expf(x); }
SKAINET_LIBM_FN float skainet_tanhf(float x) { return tanhf(x); }
SKAINET_LIBM_FN double skainet_erf(double x) { return erf(x); }

#endif /* SKAINET_SIMD_H */
