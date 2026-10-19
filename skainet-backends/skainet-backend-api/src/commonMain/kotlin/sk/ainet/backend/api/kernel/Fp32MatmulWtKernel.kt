package sk.ainet.backend.api.kernel

/**
 * Dense `x · Wᵀ` for a row-major `[out, in]` FP32 weight — the shape a `Linear` layer stores its
 * weight in, multiplied in place:
 *
 *   `out[i, o] = Σ_p x[i, p] · w[o, p]`
 *
 * Every output is one contiguous dot product over the weight's own row, so no transposed copy of
 * the weight is ever made. This is the SPI face of `skainet_fp32_matmul_wt` (SKEEP-007), which the
 * wasm sidecar has served since 0.61.0 and the JVM FFM provider binds as well: a tier whose
 * [Fp32MatmulKernel] wants the weight input-major (`[k, n]`) otherwise pays a scattered transpose
 * copy per weight and caches it — about 0.5 GB for a 22-layer encoder.
 *
 * Operands are contiguous: `x` is `m` rows of `k`, `w` is `n` rows of `k`, `out` is `m` rows of
 * `n`, each from its offset. Implementations must not mutate `x` or `w`, must fully overwrite the
 * `m × n` block of `out`, and may assume the arrays do not alias. `k == 0` zeroes the block;
 * `m == 0` or `n == 0` is a no-op. Each output accumulates its products in ascending `p`, so a
 * conforming kernel is bit-identical to transpose-then-GEMM with the same accumulation order.
 */
public interface Fp32MatmulWtKernel {
    public fun matmulWt(
        x: FloatArray, xOffset: Int,
        w: FloatArray, wOffset: Int,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    )
}
