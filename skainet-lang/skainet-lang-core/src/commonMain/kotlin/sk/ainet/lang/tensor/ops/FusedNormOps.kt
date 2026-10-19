package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Optional fused normalisations a [TensorOps] implementation may offer beside the interface —
 * the hook `LayerNormalization` (and an RMS-norm module) consults before running the decomposed
 * `mean` / `subtract` / `multiply` / `sqrt` / `divide` chain, so a backend with a one-pass kernel
 * (the SIMD128 sidecar on wasmJs, the native FFM provider on the JVM) serves the layer in one
 * call and the others keep the chain unchanged. Not part of [TensorOps]: an op set that does not
 * implement this interface — a recording or differentiating one — never sees the fused call, and
 * the recorded graph and the gradients keep their decomposed form.
 *
 * Both functions return `null` when the operands are not something the fused path takes (a dtype
 * other than FP32, parameters that are not dense, a kernel that is not loaded), and the caller
 * continues with the decomposition. The normalised dimensions are the trailing [normalizedRank]
 * dimensions of [input], flattened into one row; `gamma` and `beta` have exactly that shape.
 */
public interface FusedNormOps {
    /**
     * `(input − mean) / sqrt(var + eps) · gamma + beta` over each row of the trailing
     * [normalizedRank] dimensions, with the biased variance; [gamma] / [beta] `null` skip the affine.
     */
    public fun <T : DType, V> layerNorm(
        input: Tensor<T, V>,
        gamma: Tensor<T, V>?,
        beta: Tensor<T, V>?,
        eps: Float,
        normalizedRank: Int,
    ): Tensor<T, V>?

    /** `input / sqrt(mean(input²) + eps) · gamma` over each row of the trailing [normalizedRank] dimensions. */
    public fun <T : DType, V> rmsNorm(
        input: Tensor<T, V>,
        gamma: Tensor<T, V>?,
        eps: Float,
        normalizedRank: Int,
    ): Tensor<T, V>?
}
