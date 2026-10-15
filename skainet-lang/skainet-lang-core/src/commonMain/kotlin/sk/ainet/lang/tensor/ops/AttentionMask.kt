package sk.ainet.lang.tensor.ops

/**
 * The one definition of which keys a query row of [TensorOps.scaledDotProductAttention] attends to (#1343).
 *
 * Every mask the op knows is a *structural* rule over positions, not a tensor:
 *
 * - **causal**: key `ki` is attended when `ki <= absQ`;
 * - **sliding window** of size `w` with `rightContext = r`: `absQ - w + 1 <= ki <= absQ + r`
 *   (`r = 0` is the classic causal left window, `r > 0` a local bidirectional band);
 * - **key padding** with per-row `length`: `ki < length`.
 *
 * `absQ = seqKV - seqQ + qi` aligns the last query with the last key, which is what a KV-cache
 * decode step needs (`seqQ = 1`, `seqKV = pos + 1`) and reduces to `qi` for `seqQ == seqKV`.
 * When several rules apply their key sets intersect. A masked score is `score + MASKED`
 * ([MASKED] is a large finite negative, never `-inf`), so a fully masked row stays finite.
 *
 * The CPU kernel, the StableHLO decomposition and the dense fallback mask ([additive]) all read
 * this object, so the three cannot drift; the kind of rule in force travels on the graph node as
 * `parameters["maskKind"]` ([kindOf]) and `keyLengths` as its own operand, so a compiled target
 * can regenerate the cheapest legal form (lengths for a fused kernel, iota/compare/select for a
 * portable decomposition) instead of inheriting a `[batch, 1, seqQ, seqKV]` constant.
 */
public object AttentionMask {

    /** Additive value of a masked score: `-1e30`, the value the attention paths agreed on before this builder existed. */
    public const val MASKED: Float = -1.0e30f

    public const val KIND_NONE: String = "none"
    public const val KIND_CAUSAL: String = "causal"
    public const val KIND_SLIDING_WINDOW: String = "slidingWindow"
    public const val KIND_KEY_PADDING: String = "keyPadding"
    public const val KIND_COMBINED: String = "combined"

    /** The recorded `maskKind` for a call: padding alone, a positional rule alone, both (`combined`), or `none`. */
    public fun kindOf(causal: Boolean, window: Int, hasLengths: Boolean): String {
        val positional = causal || window > 0
        return when {
            hasLengths && positional -> KIND_COMBINED
            hasLengths -> KIND_KEY_PADDING
            window > 0 -> KIND_SLIDING_WINDOW
            causal -> KIND_CAUSAL
            else -> KIND_NONE
        }
    }

    /** Whether a node with this `maskKind` carries a `keyLengths` operand. */
    public fun hasLengths(maskKind: String?): Boolean = maskKind == KIND_KEY_PADDING || maskKind == KIND_COMBINED

    /** Argument contract shared by every implementor of the op. */
    public fun validate(window: Int, rightContext: Int) {
        require(window >= 0) { "SDPA: window must be >= 0, got $window" }
        require(rightContext >= 0) { "SDPA: rightContext must be >= 0, got $rightContext" }
        require(rightContext == 0 || window > 0) { "SDPA: rightContext ($rightContext) is only meaningful with window > 0" }
    }

    /** Absolute position of query row [qi]: the last query aligns with the last key. */
    public fun queryPosition(qi: Int, seqQ: Int, seqKV: Int): Int = seqKV - seqQ + qi

    /** First attended key (inclusive) of query row [qi]. */
    public fun firstKey(qi: Int, seqQ: Int, seqKV: Int, window: Int): Int =
        if (window > 0) maxOf(0, queryPosition(qi, seqQ, seqKV) - window + 1) else 0

    /**
     * End of the attended keys (exclusive) of query row [qi]; [length] is the row's key-padding
     * length (`seqKV` when there is none). The result may be `<=` [firstKey]: an empty range.
     */
    public fun endKey(qi: Int, seqQ: Int, seqKV: Int, causal: Boolean, window: Int, rightContext: Int, length: Int): Int {
        var end = minOf(seqKV, length)
        val absQ = queryPosition(qi, seqQ, seqKV)
        if (causal) end = minOf(end, absQ + 1)
        if (window > 0) end = minOf(end, absQ + rightContext + 1)
        return maxOf(0, end)
    }

    /** The attended keys of query row [qi] as a half-open range `[first, end)`, possibly empty. */
    public fun keyRange(qi: Int, seqQ: Int, seqKV: Int, causal: Boolean, window: Int, rightContext: Int, length: Int): IntRange =
        firstKey(qi, seqQ, seqKV, window) until endKey(qi, seqQ, seqKV, causal, window, rightContext, length)

    /** Whether query row [qi] attends to key [ki] under the given rules. */
    public fun attends(qi: Int, ki: Int, seqQ: Int, seqKV: Int, causal: Boolean, window: Int, rightContext: Int, length: Int): Boolean =
        ki in keyRange(qi, seqQ, seqKV, causal, window, rightContext, length)

    /**
     * The dense additive form of the rules: `[batch, 1, seqQ, seqKV]` row-major, `0` where attended
     * and [MASKED] elsewhere — the tensor a caller built by hand before the op took the rules as
     * metadata, and the reference the structured paths are tested against bit for bit.
     *
     * @param lengths per-row key-padding lengths (`size == batch`), or `null` for no padding
     */
    public fun additive(batch: Int, seqQ: Int, seqKV: Int, causal: Boolean, window: Int, rightContext: Int, lengths: IntArray?): FloatArray {
        validate(window, rightContext)
        require(lengths == null || lengths.size == batch) { "SDPA: keyLengths has ${lengths?.size} rows, batch is $batch" }
        val out = FloatArray(batch * seqQ * seqKV) { MASKED }
        for (b in 0 until batch) {
            val length = lengths?.get(b) ?: seqKV
            for (qi in 0 until seqQ) {
                val row = (b * seqQ + qi) * seqKV
                for (ki in keyRange(qi, seqQ, seqKV, causal, window, rightContext, length)) out[row + ki] = 0f
            }
        }
        return out
    }
}
