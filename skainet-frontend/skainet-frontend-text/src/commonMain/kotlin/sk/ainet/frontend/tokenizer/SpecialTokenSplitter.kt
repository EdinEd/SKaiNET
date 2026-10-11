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
 * @param addedTokens added tokens keyed by surface form (the 0.57 `specialTokens` parameter)
 * @param bosTokenId / eosTokenId 0.57 overrides, kept for source compatibility
 * @param specialIds the subset of [specialTokens] ids flagged `special` (defaults to all of them)
 */
public class SpecialTokenSplitter(
    private val base: Tokenizer,
    private val addedTokens: Map<String, Int>,
    bosTokenId: Int? = base.specialTokens.bos,
    eosTokenId: Int? = base.specialTokens.eos,
    specialIds: Set<Int> = addedTokens.values.toSet(),
) : Tokenizer {

    private val specialIdToString: Map<Int, String> = addedTokens.entries.associate { (k, v) -> v to k }

    /** Longest-first ordering so e.g. `<|im_start|>` wins over `<|im`. */
    private val specialsByLengthDesc: List<String> = addedTokens.keys.filter { it.isNotEmpty() }.sortedByDescending { it.length }

    override val vocabulary: Vocabulary = ExtendedVocabulary(base.vocabulary, addedTokens)

    override val specialTokens: SpecialTokens = base.specialTokens.copy(
        bos = bosTokenId,
        eos = eosTokenId,
        declared = base.specialTokens.declared + specialIds,
    )

    override val spec: FrontEndSpec get() = base.spec

    override fun encode(text: String): IntArray {
        if (specialsByLengthDesc.isEmpty()) return base.encode(text)
        val out = ArrayList<Int>(text.length)
        var i = 0
        while (i < text.length) {
            val matched = matchSpecialAt(text, i)
            if (matched != null) {
                out.add(addedTokens.getValue(matched))
                i += matched.length
                continue
            }
            val nextSpecial = nextSpecialStart(text, i)
            val segment = text.substring(i, nextSpecial)
            for (id in base.encode(segment)) out.add(id)
            i = nextSpecial
        }
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

    private fun matchSpecialAt(text: String, from: Int): String? {
        for (tok in specialsByLengthDesc) {
            if (text.regionMatches(from, tok, 0, tok.length)) return tok
        }
        return null
    }

    private fun nextSpecialStart(text: String, from: Int): Int {
        var earliest = text.length
        for (tok in specialsByLengthDesc) {
            val idx = text.indexOf(tok, startIndex = from + 1)
            if (idx in 0 until earliest) earliest = idx
        }
        return earliest
    }
}

/** A base vocabulary plus added tokens whose ids may lie above the base range. */
internal class ExtendedVocabulary(private val base: Vocabulary, private val added: Map<String, Int>) : Vocabulary {
    private val addedById: Map<Int, String> = added.entries.associate { (k, v) -> v to k }
    override val size: Int = maxOf(base.size, (added.values.maxOrNull() ?: -1) + 1)
    override fun tokenToId(piece: String): Int? = added[piece] ?: base.tokenToId(piece)
    override fun idToToken(id: Int): String? = addedById[id] ?: base.idToToken(id)
}
