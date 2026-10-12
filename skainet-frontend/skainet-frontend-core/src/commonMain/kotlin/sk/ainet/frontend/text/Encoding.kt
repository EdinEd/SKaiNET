package sk.ainet.frontend.text

/**
 * Side-output form of an encoding (FE-C-08): what the BERT module's `encodeWithMetadata` becomes,
 * generalised. Attention masks and token type ids belong here, never on the base [Tokenizer].
 *
 * @property ids token ids including any added special tokens
 * @property attentionMask `1` for every real token, `0` for padding
 * @property tokenTypeIds segment ids (`0` first sequence, `1` second); `null` when the algorithm has no notion of segments
 * @property specialTokensMask `1` where the id is a special token added by the encoding
 * @property offsets character offsets into the input per token; `null` when the algorithm cannot produce them
 */
public class Encoding(
    public val ids: IntArray,
    public val attentionMask: IntArray,
    public val tokenTypeIds: IntArray?,
    public val specialTokensMask: IntArray,
    public val offsets: Array<IntRange>? = null,
) {
    public val length: Int get() = ids.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Encoding) return false
        return ids.contentEquals(other.ids) &&
            attentionMask.contentEquals(other.attentionMask) &&
            (tokenTypeIds?.contentEquals(other.tokenTypeIds ?: IntArray(0)) ?: (other.tokenTypeIds == null)) &&
            specialTokensMask.contentEquals(other.specialTokensMask)
    }

    override fun hashCode(): Int {
        var r = ids.contentHashCode()
        r = 31 * r + attentionMask.contentHashCode()
        r = 31 * r + (tokenTypeIds?.contentHashCode() ?: 0)
        r = 31 * r + specialTokensMask.contentHashCode()
        return r
    }

    override fun toString(): String = "Encoding(ids=${ids.toList()}, attentionMask=${attentionMask.toList()}, tokenTypeIds=${tokenTypeIds?.toList()})"
}

/** How [BatchTokenizer.encodeWithMetadata] cuts an over-long input. */
public sealed interface Truncation {
    public object None : Truncation

    /** Keep the first [maxLength] ids (specials included). */
    public data class Longest(val maxLength: Int) : Truncation
}

/** How [BatchTokenizer.encodeBatch] pads a batch. */
public sealed interface Padding {
    /** Pad every row to the longest row. */
    public object Longest : Padding

    /** Pad (and truncate) every row to exactly [length]. */
    public data class Fixed(val length: Int) : Padding
}

/**
 * Row-major padded batch. Arrays rather than tensors so no `ExecutionContext` is needed here; a
 * consumer turns `ids`/`attentionMask` into `[batchSize, seqLen]` tensors in one call.
 */
public class BatchEncoding(
    public val batchSize: Int,
    public val seqLen: Int,
    public val ids: IntArray,
    public val attentionMask: IntArray,
    public val tokenTypeIds: IntArray?,
) {
    init {
        require(ids.size == batchSize * seqLen) { "ids has ${ids.size} entries, expected $batchSize*$seqLen" }
    }

    public fun row(i: Int): IntArray = ids.copyOfRange(i * seqLen, (i + 1) * seqLen)
}

/** Tokenizers that produce [Encoding]s with masks and segment ids (WordPiece / BERT-class encoders). */
public interface BatchTokenizer : Tokenizer {
    public fun encodeWithMetadata(text: String, pair: String? = null, truncation: Truncation = Truncation.None): Encoding

    public fun encodeBatch(
        texts: List<String>,
        padding: Padding = Padding.Longest,
        truncation: Truncation = Truncation.None,
    ): BatchEncoding {
        val encs = texts.map { encodeWithMetadata(it, null, truncation) }
        val seqLen = when (padding) {
            is Padding.Longest -> encs.maxOfOrNull { it.length } ?: 0
            is Padding.Fixed -> padding.length
        }
        val padId = specialTokens.pad ?: 0
        val n = encs.size
        val ids = IntArray(n * seqLen) { padId }
        val mask = IntArray(n * seqLen)
        val hasTypes = encs.any { it.tokenTypeIds != null }
        val types = if (hasTypes) IntArray(n * seqLen) else null
        for ((r, e) in encs.withIndex()) {
            val len = minOf(e.length, seqLen)
            for (c in 0 until len) {
                ids[r * seqLen + c] = e.ids[c]
                mask[r * seqLen + c] = e.attentionMask[c]
                if (types != null) types[r * seqLen + c] = e.tokenTypeIds?.get(c) ?: 0
            }
        }
        return BatchEncoding(n, seqLen, ids, mask, types)
    }
}
