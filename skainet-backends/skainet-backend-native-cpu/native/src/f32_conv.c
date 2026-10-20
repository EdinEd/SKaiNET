#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Dense FP32 convolution as im2col + GEMM (SKEEP-007 resident activations, conv step).
 *
 * One kernel serves conv1d, conv2d and conv3d: the geometry is three spatial dims (depth, height,
 * width) and a lower-rank convolution sets the leading ones to 1 (kernel 1, stride 1, padding 0,
 * dilation 1, output 1). Layouts are the ops layer's: input `[n, c_in, d, h, w]`, weight
 * `[c_out, c_in / groups, k_d, k_h, k_w]`, bias `[c_out]` or NULL, output `[n, c_out, out_d, out_h, out_w]`.
 *
 *   skainet_f32_conv(in, w, bias, out, col, geo)
 *     geo is SKAINET_CONV_GEO_INTS int32 values in the order
 *       n, c_in, c_out, groups,
 *       in_d, in_h, in_w,  k_d, k_h, k_w,  s_d, s_h, s_w,  p_d, p_h, p_w,  d_d, d_h, d_w,
 *       out_d, out_h, out_w
 *     with out_x = (in_x + 2 p_x - d_x (k_x - 1) - 1) / s_x + 1 as the caller computed it.
 *     `col` is caller-provided scratch of patch_count × col_size floats, where
 *     patch_count = out_d · out_h · out_w and col_size = (c_in / groups) · k_d · k_h · k_w; it is
 *     reused for every (batch, group) pair, so the caller allocates it once per call.
 *
 * For each batch and group the receptive fields are gathered patch-major into `col`
 * (`col[p, :]` is patch p's c_in_g × k_d × k_h × k_w window, zero where it falls in the padding —
 * the layout the JVM's `JvmVectorKernels` im2col produces), and the output block
 * `out[oc, p] = bias[oc] + Σ_k w[oc, k] · col[p, k]` is one `skainet_fp32_matmul_wt` call with the
 * weight rows as `in` and the patches as `w`: every output is a contiguous dot product over two
 * rows, which is the SIMD128 / NEON body that kernel already has, accumulated in ascending k.
 *
 * Threading: the gather runs over patches and the GEMM over patches through the shared row pool,
 * so a 28 × 28 map (784 patches) threads on the mt sidecar and the native pool, while a tiny map
 * stays on the caller. Workers write disjoint rows of `col` and disjoint columns of `out`.
 */

enum {
    G_N = 0, G_C_IN, G_C_OUT, G_GROUPS,
    G_IN_D, G_IN_H, G_IN_W,
    G_K_D, G_K_H, G_K_W,
    G_S_D, G_S_H, G_S_W,
    G_P_D, G_P_H, G_P_W,
    G_D_D, G_D_H, G_D_W,
    G_OUT_D, G_OUT_H, G_OUT_W
};

typedef struct {
    const float* in;        /* this batch, this group's first channel */
    float* col;             /* patch_count rows of col_size */
    int32_t c_in_g;
    int32_t in_d, in_h, in_w;
    int32_t k_d, k_h, k_w;
    int32_t s_d, s_h, s_w;
    int32_t p_d, p_h, p_w;
    int32_t d_d, d_h, d_w;
    int32_t out_h, out_w;
    int32_t col_size;
} skainet_im2col_ctx;

/* Patches [p0, p1): one row of `col` each. */
static void skainet_im2col_rows(void* vctx, int32_t p0, int32_t p1) {
    const skainet_im2col_ctx* c = (const skainet_im2col_ctx*) vctx;
    const int32_t out_hw = c->out_h * c->out_w;
    const size_t in_hw = (size_t) c->in_h * c->in_w;
    const size_t in_dhw = in_hw * (size_t) c->in_d;
    const int32_t k_w = c->k_w;
    for (int32_t p = p0; p < p1; ++p) {
        const int32_t od = p / out_hw;
        const int32_t rem = p - od * out_hw;
        const int32_t oh = rem / c->out_w;
        const int32_t ow = rem - oh * c->out_w;
        const int32_t d0 = od * c->s_d - c->p_d;
        const int32_t h0 = oh * c->s_h - c->p_h;
        const int32_t w0 = ow * c->s_w - c->p_w;
        float* dst = c->col + (size_t) p * c->col_size;
        for (int32_t ci = 0; ci < c->c_in_g; ++ci) {
            const float* chan = c->in + (size_t) ci * in_dhw;
            for (int32_t kd = 0; kd < c->k_d; ++kd) {
                const int32_t id = d0 + kd * c->d_d;
                const int d_ok = id >= 0 && id < c->in_d;
                for (int32_t kh = 0; kh < c->k_h; ++kh) {
                    const int32_t ih = h0 + kh * c->d_h;
                    if (!d_ok || ih < 0 || ih >= c->in_h) {
                        memset(dst, 0, (size_t) k_w * sizeof(float));
                        dst += k_w;
                        continue;
                    }
                    const float* row = chan + (size_t) id * in_hw + (size_t) ih * c->in_w;
                    if (c->d_w == 1) {
                        /* kw with w0 + kw inside [0, in_w): one contiguous copy, zeros either side. */
                        int32_t kw0 = w0 < 0 ? -w0 : 0;
                        int32_t kw1 = c->in_w - w0;
                        if (kw1 > k_w) kw1 = k_w;
                        if (kw1 < kw0) kw1 = kw0;
                        if (kw0 > 0) memset(dst, 0, (size_t) kw0 * sizeof(float));
                        if (kw1 > kw0) memcpy(dst + kw0, row + w0 + kw0, (size_t) (kw1 - kw0) * sizeof(float));
                        if (kw1 < k_w) memset(dst + kw1, 0, (size_t) (k_w - kw1) * sizeof(float));
                        dst += k_w;
                    } else {
                        for (int32_t kw = 0; kw < k_w; ++kw) {
                            const int32_t iw = w0 + kw * c->d_w;
                            *dst++ = (iw >= 0 && iw < c->in_w) ? row[iw] : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

SKAINET_API void skainet_f32_conv(
    const float* in, const float* w, const float* bias, float* out, float* col, const int32_t* geo
) {
    const int32_t n = geo[G_N], c_in = geo[G_C_IN], c_out = geo[G_C_OUT], groups = geo[G_GROUPS];
    if (n <= 0 || c_out <= 0 || groups <= 0) return;
    const int32_t out_d = geo[G_OUT_D], out_h = geo[G_OUT_H], out_w = geo[G_OUT_W];
    const int32_t patch_count = out_d * out_h * out_w;
    if (patch_count <= 0) return;
    const int32_t c_in_g = c_in / groups;
    const int32_t c_out_g = c_out / groups;
    const int32_t col_size = c_in_g * geo[G_K_D] * geo[G_K_H] * geo[G_K_W];
    const size_t in_dhw = (size_t) geo[G_IN_D] * geo[G_IN_H] * geo[G_IN_W];
    const size_t in_batch = (size_t) c_in * in_dhw;
    const size_t out_batch = (size_t) c_out * patch_count;

    skainet_im2col_ctx ctx;
    ctx.col = col;
    ctx.c_in_g = c_in_g;
    ctx.in_d = geo[G_IN_D]; ctx.in_h = geo[G_IN_H]; ctx.in_w = geo[G_IN_W];
    ctx.k_d = geo[G_K_D]; ctx.k_h = geo[G_K_H]; ctx.k_w = geo[G_K_W];
    ctx.s_d = geo[G_S_D]; ctx.s_h = geo[G_S_H]; ctx.s_w = geo[G_S_W];
    ctx.p_d = geo[G_P_D]; ctx.p_h = geo[G_P_H]; ctx.p_w = geo[G_P_W];
    ctx.d_d = geo[G_D_D]; ctx.d_h = geo[G_D_H]; ctx.d_w = geo[G_D_W];
    ctx.out_h = out_h; ctx.out_w = out_w;
    ctx.col_size = col_size;

    for (int32_t b = 0; b < n; ++b) {
        for (int32_t g = 0; g < groups; ++g) {
            const float* w_g = w + (size_t) g * c_out_g * col_size;
            float* out_bg = out + (size_t) b * out_batch + (size_t) g * c_out_g * patch_count;
            if (col_size > 0 && c_out_g > 0) {
                ctx.in = in + (size_t) b * in_batch + (size_t) g * c_in_g * in_dhw;
                skainet_run_rows(skainet_im2col_rows, &ctx, patch_count);
            }
            /* k <= 0 zeroes the block inside the kernel (an empty receptive field). */
            skainet_fp32_matmul_wt(w_g, 0, col, 0, out_bg, 0, c_out_g, patch_count, col_size);
            if (bias) {
                for (int32_t oc = 0; oc < c_out_g; ++oc) {
                    const float bv = bias[(size_t) g * c_out_g + oc];
                    float* row = out_bg + (size_t) oc * patch_count;
                    for (int32_t p = 0; p < patch_count; ++p) row[p] += bv;
                }
            }
        }
    }
}
