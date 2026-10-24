package sk.ainet.frontend.tokenizer

/**
 * Normaliser + pre-tokenizer of one tokenizer, plus the decoder facts the inverse needs.
 *
 * @property stripLeadingSpaces how many leading spaces the reference decoder removes from a whole
 *   sequence (`Strip` decoder step, or a `Metaspace` decoder that undoes the prepended prefix)
 */
public class TextPipeline(
    public val normalizer: Normalizer = Normalizer.None,
    public val preTokenizer: PreTokenizer = PreTokenizer.None,
    public val stripLeadingSpaces: Int = 0,
) {
    /** Normalise then cut [segment] into model chunks. */
    public fun chunks(segment: String): List<String> = preTokenizer.split(normalizer.apply(segment))

    public companion object {
        public val NONE: TextPipeline = TextPipeline()
    }
}

/**
 * Implemented by algorithms whose whole-sequence [sk.ainet.frontend.text.Tokenizer.decode] strips
 * leading spaces (the SentencePiece prefix, a `Strip` decoder step). A wrapper that decodes a
 * sequence in runs — [SpecialTokenSplitter] between added tokens — decodes each run with
 * [decodeRun] and applies [leadingSpacesToStrip] once, to the whole result, as the reference does.
 */
public interface RunDecoder {
    public val leadingSpacesToStrip: Int
    public fun decodeRun(ids: IntArray): String
}

/**
 * Shared atomic matching of added tokens over a text: walks left to right, longest registered
 * token first, honouring `single_word` (only at word boundaries) and `lstrip`/`rstrip`
 * (absorb adjacent whitespace). The non-added stretches are handed to [encodeSegment].
 */
internal class AddedTokenMatcher(tokens: Collection<AddedToken>) {
    private val byContent: Map<String, AddedToken> = tokens.filter { it.content.isNotEmpty() }.associateBy { it.content }
    /**
     * The distinct lengths of the added tokens per first character, longest first. Only a token starting with
     * `text[from]` can match at `from`, and at most one per length, so a position costs one map lookup per
     * length. Gemma's GGUF declares ~6,400 added tokens; comparing each of them at every position cost ~450k
     * region compares for a 70-character chat turn.
     */
    private val lengthsByFirstChar: Map<Char, IntArray> =
        byContent.keys.groupBy { it[0] }.mapValues { (_, c) -> c.map { it.length }.distinct().sortedDescending().toIntArray() }

    val isEmpty: Boolean get() = byContent.isEmpty()
    val ids: Set<Int> = byContent.values.map { it.id }.toSet()
    val specialIds: Set<Int> = byContent.values.filter { it.special }.map { it.id }.toSet()
    val idToContent: Map<Int, String> = byContent.values.associate { it.id to it.content }
    val contentToId: Map<String, Int> = byContent.mapValues { it.value.id }

    inline fun encode(text: String, out: MutableList<Int>, encodeSegment: (String) -> Unit) {
        if (isEmpty) { encodeSegment(text); return }
        var i = 0
        var segmentStart = 0
        while (i < text.length) {
            val tok = matchAt(text, i)
            if (tok == null) { i++; continue }
            var segEnd = i
            if (tok.lstrip) while (segEnd > segmentStart && text[segEnd - 1].isWhitespace()) segEnd--
            if (segEnd > segmentStart) encodeSegment(text.substring(segmentStart, segEnd))
            out.add(tok.id)
            i += tok.content.length
            if (tok.rstrip) while (i < text.length && text[i].isWhitespace()) i++
            segmentStart = i
        }
        if (segmentStart < text.length) encodeSegment(text.substring(segmentStart))
    }

    private fun matchAt(text: String, from: Int): AddedToken? {
        for (length in lengthsByFirstChar[text[from]] ?: return null) {
            if (from + length > text.length) continue
            val tok = byContent[text.substring(from, from + length)] ?: continue
            if (tok.singleWord) {
                val before = if (from > 0) text[from - 1] else ' '
                val afterIdx = from + length
                val after = if (afterIdx < text.length) text[afterIdx] else ' '
                if (before.isLetterOrDigit() || after.isLetterOrDigit()) continue
            }
            return tok
        }
        return null
    }
}
