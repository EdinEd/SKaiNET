package sk.ainet.frontend.tokenizer

import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * Decorator that adds atomic added-token matching on top of a base [Tokenizer] that does not
 * already implement it.
 *
 * Chat-template models embed control markers — `<bos>`, `<|im_start|>`, `<|turn>`, `<tool_call>` —
 * that the model is trained to see as a single id rather than as the byte/SP-encoded fragments
 * their literal string would otherwise produce. The reference library matches *every*
 * `added_tokens` entry atomically before pre-tokenization, special or not (SKaiNET #1324); the
 * `special` flag only decides what `decode(skipSpecialTokens = true)` drops.
 *
 * Algorithm: **encode** walks left to right; at each position the longest registered token
 * wins, otherwise the non-added segment up to the next match is handed to `base.encode`.
 * **decode** collects contiguous base ids and flushes them through `base.decode`, emitting the
 * added token's string in between. Added-token boundaries always sit on UTF-8 boundaries, so the
 * byte-level spanning some bases do inside `decode(IntArray)` is preserved within each run.
 *
 * @param base the algorithm tokenizer
 * @param tokens the added tokens with their `lstrip` / `rstrip` / `single_word` flags; matching
 *   honours them exactly as [AddedTokenMatcher] does for the algorithms that match inline
 * @param bosTokenId / eosTokenId 0.57 overrides, kept for source compatibility
 */
public class SpecialTokenSplitter private constructor(
    private val base: Tokenizer,
    tokens: List<AddedToken>,
    bosTokenId: Int?,
    eosTokenId: Int?,
) : Tokenizer {

    /**
     * 0.57-compatible form: added tokens keyed by surface form, no whitespace flags.
     *
     * @param addedTokens added tokens keyed by surface form (the 0.57 `specialTokens` parameter)
     * @param specialIds the subset of ids flagged `special` (defaults to all of them)
     */
    public constructor(
        base: Tokenizer,
        addedTokens: Map<String, Int>,
        bosTokenId: Int? = base.specialTokens.bos,
        eosTokenId: Int? = base.specialTokens.eos,
        specialIds: Set<Int> = addedTokens.values.toSet(),
    ) : this(base, addedTokens.map { (content, id) -> AddedToken(content = content, id = id, special = id in specialIds) }, bosTokenId, eosTokenId)

    /** Added tokens as declared in `tokenizer.json`, flags included (`<mask>` is typically `lstrip`). */
    public constructor(
        base: Tokenizer,
        tokens: Collection<AddedToken>,
        bosTokenId: Int? = base.specialTokens.bos,
        eosTokenId: Int? = base.specialTokens.eos,
    ) : this(base, tokens.toList(), bosTokenId, eosTokenId)

    private val matcher = AddedTokenMatcher(tokens)
    private val addedTokens: Map<String, Int> = matcher.contentToId
    private val specialIdToString: Map<Int, String> = matcher.idToContent
    private val specialIds: Set<Int> = matcher.specialIds

    override val vocabulary: Vocabulary = ExtendedVocabulary(base.vocabulary, addedTokens)

    override val specialTokens: SpecialTokens = base.specialTokens.copy(
        bos = bosTokenId,
        eos = eosTokenId,
        declared = base.specialTokens.declared + specialIds,
    )

    override val spec: FrontEndSpec get() = base.spec

    override fun encode(text: String): IntArray {
        if (matcher.isEmpty) return base.encode(text)
        val out = ArrayList<Int>(text.length)
        matcher.encode(text, out) { segment -> for (id in base.encode(segment)) out.add(id) }
        return IntArray(out.size) { out[it] }
    }

    /**
     * Single-id decode must delegate to the base's own [Tokenizer.decodeToken], not to [decode]:
     * the batch path would re-enable the leading-space stripping bases like
     * [SentencePieceTokenizer] deliberately disable for per-token streaming ("the process" would
     * stream as "theprocess").
     */
    override fun decodeToken(id: Int): String = specialIdToString[id] ?: base.decodeToken(id)

    override fun decode(ids: IntArray): String {
        if (ids.isEmpty()) return ""
        if (specialIdToString.isEmpty()) return base.decode(ids)
        val runDecoder = base as? RunDecoder
        val sb = StringBuilder()
        val buffer = ArrayList<Int>()
        fun flush() {
            if (buffer.isEmpty()) return
            val run = buffer.toIntArray()
            sb.append(runDecoder?.decodeRun(run) ?: base.decode(run))
            buffer.clear()
        }
        for (id in ids) {
            val special = specialIdToString[id]
            if (special != null) {
                flush()
                sb.append(special)
            } else {
                buffer.add(id)
            }
        }
        flush()
        var result = sb.toString()
        // The reference strips the prefix once, from the whole sequence — never per run.
        var strip = runDecoder?.leadingSpacesToStrip ?: 0
        while (strip > 0 && result.startsWith(' ')) { result = result.substring(1); strip-- }
        return result
    }
}

/** A base vocabulary plus added tokens whose ids may lie above the base range. */
internal class ExtendedVocabulary(private val base: Vocabulary, private val added: Map<String, Int>) : Vocabulary {
    private val addedById: Map<Int, String> = added.entries.associate { (k, v) -> v to k }
    override val size: Int = maxOf(base.size, (added.values.maxOrNull() ?: -1) + 1)
    override fun tokenToId(piece: String): Int? = added[piece] ?: base.tokenToId(piece)
    override fun idToToken(id: Int): String? = addedById[id] ?: base.idToToken(id)
}
