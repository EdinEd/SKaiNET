package sk.ainet.backend.api.kernel

/**
 * Row-wise softmax over dense FP32: `out[r, :] = softmax(input[r, :])` for `rows` contiguous rows of
 * `cols`, or the log-softmax (`input - (log Σ exp(input - max) + max)`) when [log] is `true`. The
 * SPI face of `skainet_f32_softmax_rows` (SKEEP-007 fused ops), which the wasm sidecar and the JVM
 * FFM provider both serve; a provider without it returns `null` from [KernelProvider.softmaxFp32]
 * and the ops layer keeps its common loop.
 *
 * Operands are contiguous from their offsets: `rows × cols` floats each. Implementations must not
 * mutate [input] and must fully overwrite the output block; `rows == 0` or `cols == 0` is a no-op.
 * `exp` and `log` are the platform's libm, so results agree with the common Kotlin loop to the
 * last-ulp differences between the two.
 */
public interface Fp32SoftmaxKernel {
    public fun softmaxRows(
        input: FloatArray, inputOffset: Int,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int,
        log: Boolean,
    )
}

/**
 * Fused normalisations over the last dimension of dense FP32, one pass per row — the SPI face of
 * `skainet_f32_layer_norm` and `skainet_f32_rms_norm` (SKEEP-007 fused ops):
 *
 * - [layerNorm]: `out[r, c] = (x[r, c] − mean_r) / sqrt(var_r + eps) · gamma[c] + beta[c]` with the
 *   mean and the biased variance over the row;
 * - [rmsNorm]: `out[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps) · gamma[c]`.
 *
 * [gamma] and [beta] are `cols` floats or `null` (the affine step is skipped). Operands are
 * contiguous from their offsets; implementations must not mutate the inputs and must fully overwrite
 * the `rows × cols` output block. The row sums accumulate in lanes where the kernel is vectorised,
 * so results differ from the decomposed `mean`/`subtract`/`sqrt`/`divide` chain in the last bits.
 */
public interface Fp32NormKernel {
    public fun layerNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?, beta: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int,
        eps: Float,
    )

    public fun rmsNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int,
        eps: Float,
    )
}

/**
 * Fused scaled dot-product attention over dense FP32 — the SPI face of `skainet_f32_sdpa` (SKEEP-007
 * fused ops), with the full contract of `TensorOps.scaledDotProductAttention`:
 *
 * `out = softmax(query · keyᵀ · scale + rules + mask) · value`
 *
 * [query] is `[batch, heads, seqQ, headDim]`, [key] and [value] are `[batch, kvHeads, seqKV, headDim]`
 * with query head `h` reading K/V head `h / (heads / kvHeads)` (grouped-query attention), [out] is
 * `[batch, heads, seqQ, headDim]`, all contiguous whole arrays. [scale] is applied as given — the
 * caller resolves the `0 → 1/sqrt(headDim)` default. The structural rules are those of
 * `AttentionMask`: [causal], a sliding [window] with [rightContext], and per-batch [keyLengths]
 * (`[batch]` or `null`); a key outside a row's attended set scores `-1e30` and its product is
 * never computed, so a fully masked row is uniform and finite. [mask] is a dense additive mask
 * (or `null`) broadcast trailing-aligned over [maskDims] `= [batch | 1, heads | kvHeads | 1,
 * seqQ | 1, seqKV | 1]`, added to every score.
 *
 * The `q · k` dot products accumulate in lanes where the kernel is vectorised; the
 * probability-weighted sum of `value` accumulates per output element in ascending key order.
 * Implementations must not mutate the inputs and must fully overwrite [out].
 */
public interface Fp32AttentionKernel {
    public fun sdpa(
        query: FloatArray, key: FloatArray, value: FloatArray, out: FloatArray,
        batch: Int, heads: Int, kvHeads: Int,
        seqQ: Int, seqKV: Int, headDim: Int,
        scale: Float, causal: Boolean, window: Int, rightContext: Int,
        keyLengths: IntArray?,
        mask: FloatArray?, maskDims: IntArray?,
    )
}
