package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * SentencePiece tokenizer for LLaMA, Gemma, TinyLlama, Mistral-v0.1 and other models whose GGUF
 * `tokenizer.ggml.model` is `"llama"` (or `"gemma4"`).
 *
 * This matches the algorithm used by `llm_tokenizer_spm` in llama.cpp:
 *
 * 1. **Whitespace escape**: every space is replaced with `▁` (U+2581), and — when
 *    [addSpacePrefix] is true — a leading `▁` is prepended so the first word can still match
 *    merged vocab entries like `▁Hello`.
 * 2. **Symbol split**: the escaped input is broken into code-point-sized symbols.
 * 3. **Score-priority BPE**: at each step the adjacent pair whose concatenation is in the vocab
 *    with the highest score is merged, until no pair is in the vocab. This is the *score-wins*
 *    rule, the opposite of the merge-rank rule of [QwenByteLevelBpeTokenizer].
 * 4. **Byte fallback**: a symbol not in the vocab is re-emitted one UTF-8 byte at a time as
 *    `<0x00>`..`<0xFF>` (GGUF `token_type == 6`), else [unknownTokenId].
 *
 * Decode is the inverse; a leading space is stripped when [addSpacePrefix] is set.
 *
 * For `tokenizer.json` files with `model.type == "Unigram"` the reference algorithm is the
 * Viterbi lattice of [UnigramTokenizer]; [fromTokenizerJson] is kept for source compatibility.
 */
public class SentencePieceTokenizer internal constructor(
    tokens: List<String>,
    private val idToScore: FloatArray,
    public val unknownTokenId: Int?,
    bosTokenId: Int?,
    eosTokenId: Int?,
    public val addSpacePrefix: Boolean,
    policy: SpecialTokens?,
    source: String?,
) : Tokenizer, RunDecoder {

    public constructor(
        tokens: List<String>,
        scores: List<Float>,
        unknownTokenId: Int? = null,
        bosTokenId: Int? = null,
        eosTokenId: Int? = null,
        addSpacePrefix: Boolean = true,
        policy: SpecialTokens? = null,
        source: String? = null,
    ) : this(tokens, scores.toFloatArray(), unknownTokenId, bosTokenId, eosTokenId, addSpacePrefix, policy, source)

    private val pieces: ArrayVocabulary

    /** `byteTokenIds[b]` = vocab id of `<0xBB>`, or `-1` if absent. */
    private val byteTokenIds: IntArray

    init {
        require(tokens.size == idToScore.size) {
            "tokens (${tokens.size}) and scores (${idToScore.size}) must have the same length"
        }
        pieces = ArrayVocabulary(tokens)
        byteTokenIds = IntArray(256) { b -> pieces.tokenToId(byteTokenString(b)) ?: -1 }
    }

    override val vocabulary: Vocabulary = pieces

    override val specialTokens: SpecialTokens = policy ?: SpecialTokens(bos = bosTokenId, eos = eosTokenId, unk = unknownTokenId)

    override val spec: FrontEndSpec = FrontEndSpec.Text(algorithm = ALGORITHM, artifact = source)

    override fun encode(text: String): IntArray {
        val input = preprocess(text)
        if (input.isEmpty()) return IntArray(0)

        val symbols = splitIntoSymbols(input)
        mergeByScore(symbols)

        val out = ArrayList<Int>(symbols.size)
        var idx = 0
        while (idx >= 0) {
            val s = symbols[idx]
            if (s.size > 0) emitSymbol(s.text, out)
            idx = s.next
        }
        return IntArray(out.size) { out[it] }
    }

    override fun decode(ids: IntArray): String = decodeRaw(ids, stripLeadingSpace = addSpacePrefix)

    /**
     * Decode a single token to its surface piece without stripping the leading word-boundary
     * space — see [Tokenizer.decodeToken].
     */
    override fun decodeToken(id: Int): String = decodeRaw(intArrayOf(id), stripLeadingSpace = false)

    override val leadingSpacesToStrip: Int get() = if (addSpacePrefix) 1 else 0
    override fun decodeRun(ids: IntArray): String = decodeRaw(ids, stripLeadingSpace = false)

    /**
     * Decode [ids] to text. When [stripLeadingSpace] is true a single leading space (the artefact
     * of `addSpacePrefix` at encode time) is removed — correct for a whole sequence, but NOT for an
     * individual streaming token. (Named `decodeRaw` since SKEEP-006 because the contract's
     * `decode(ids, skipSpecialTokens)` has the same JVM signature.)
     */
    public fun decodeRaw(ids: IntArray, stripLeadingSpace: Boolean): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val token = pieces.idToToken(id)
                ?: error("decode: unknown token id $id")
            val byte = parseByteToken(token)
            if (byte != null) {
                byteBuf.add(byte)
                continue
            }
            if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
            sb.append(token)
        }
        if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))

        val result = sb.toString().replace(WHITESPACE_ESCAPE, ' ')
        return if (stripLeadingSpace && result.startsWith(' ')) result.substring(1) else result
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    private fun preprocess(text: String): String {
        val escaped = text.replace(' ', WHITESPACE_ESCAPE)
        return if (addSpacePrefix && !escaped.startsWith(WHITESPACE_ESCAPE)) WHITESPACE_ESCAPE + escaped else escaped
    }

    private fun splitIntoSymbols(input: String): MutableList<Symbol> {
        val symbols = ArrayList<Symbol>(input.length)
        var i = 0
        var prev = -1
        while (i < input.length) {
            val c = input[i]
            val charCount =
                if (c.isHighSurrogate() && i + 1 < input.length && input[i + 1].isLowSurrogate()) 2
                else 1
            symbols.add(Symbol(text = input.substring(i, i + charCount), size = charCount, prev = prev, next = -1))
            if (prev >= 0) symbols[prev].next = symbols.size - 1
            prev = symbols.size - 1
            i += charCount
        }
        return symbols
    }

    private fun mergeByScore(symbols: MutableList<Symbol>) {
        while (true) {
            var bestLeft = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var i = 0
            while (i >= 0) {
                val left = symbols[i]
                val rightIdx = left.next
                if (rightIdx < 0) break
                val right = symbols[rightIdx]
                val merged = left.text + right.text
                val id = pieces.tokenToId(merged)
                if (id != null) {
                    val score = idToScore[id]
                    if (score > bestScore) {
                        bestScore = score
                        bestLeft = i
                    }
                }
                i = rightIdx
            }
            if (bestLeft < 0) return
            val left = symbols[bestLeft]
            val rightIdx = left.next
            val right = symbols[rightIdx]
            left.text = left.text + right.text
            left.size += right.size
            left.next = right.next
            if (right.next >= 0) symbols[right.next].prev = bestLeft
            right.size = 0
        }
    }

    private fun emitSymbol(text: String, out: ArrayList<Int>) {
        val id = pieces.tokenToId(text)
        if (id != null) {
            out.add(id)
            return
        }
        val bytes = text.encodeToByteArray()
        for (b in bytes) {
            val unsigned = b.toInt() and 0xFF
            val byteId = byteTokenIds[unsigned]
            if (byteId >= 0) {
                out.add(byteId)
            } else if (unknownTokenId != null) {
                out.add(unknownTokenId)
            } else {
                error("SentencePieceTokenizer: cannot encode byte 0x" + unsigned.toString(16) + ": no byte-fallback token and no UNK id")
            }
        }
    }

    private fun flushBytes(buf: ArrayList<Byte>): String {
        val arr = ByteArray(buf.size) { buf[it] }
        buf.clear()
        return arr.decodeToString()
    }

    private class Symbol(var text: String, var size: Int, var prev: Int, var next: Int)

    public companion object {
        public const val ALGORITHM: String = "sentencepiece"

        /** SentencePiece whitespace-escape character: `▁` (U+2581). */
        public const val WHITESPACE_ESCAPE: Char = '▁'

        private const val HEX = "0123456789ABCDEF"

        /** `<0xNN>` for byte [b]. */
        public fun byteTokenString(b: Int): String = "<0x" + HEX[(b ushr 4) and 0xF] + HEX[b and 0xF] + ">"

        /** The raw byte of a `<0xNN>` byte-fallback token, or `null` for a normal vocab entry. */
        public fun parseByteToken(token: String): Byte? {
            if (token.length != 6) return null
            if (token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') return null
            val hi = hexDigit(token[3]) ?: return null
            val lo = hexDigit(token[4]) ?: return null
            return ((hi shl 4) or lo).toByte()
        }

        private fun hexDigit(c: Char): Int? = when (c) {
            in '0'..'9' -> c.code - '0'.code
            in 'a'..'f' -> 10 + (c.code - 'a'.code)
            in 'A'..'F' -> 10 + (c.code - 'A'.code)
            else -> null
        }

        /**
         * Build from GGUF metadata fields: `tokenizer.ggml.tokens` and `.scores` (required),
         * `.token_type` (type 2 flags the unknown token), `.bos_token_id` / `.eos_token_id` /
         * `.unknown_token_id`, `.add_space_prefix` (default `true`).
         */
        public fun fromGgufFields(fields: Map<String, Any?>): SentencePieceTokenizer {
            val tokens = GgufTokenizerFields.tokens(fields)
            val types = GgufTokenizerFields.tokenTypeIds(fields)
            return fromGgufFields(fields, tokens, types, GgufTokenizerFields.specialTokenMap(tokens, types))
        }

        /** As [fromGgufFields] with the tokens, token types and special tokens already taken from [fields]. */
        internal fun fromGgufFields(
            fields: Map<String, Any?>,
            tokens: List<String>,
            types: IntArray?,
            specials: Map<String, Int>,
        ): SentencePieceTokenizer {
            val scores = GgufTokenizerFields.scores(fields)
                ?: error("${GgufTokenizerFields.SCORES} missing — required for SentencePiece")
            require(tokens.size == scores.size) { "GGUF tokens (${tokens.size}) and scores (${scores.size}) disagree" }

            var unknownId = fields[GgufTokenizerFields.UNK]?.toIntFlexible()
            if (unknownId == null) {
                val idx = types?.indexOf(GgufTokenizerFields.TOKEN_TYPE_UNKNOWN) ?: -1
                if (idx >= 0) unknownId = idx
            }
            val addSpacePrefix = fields[GgufTokenizerFields.ADD_SPACE_PREFIX].toBooleanFlexible(default = true)
            val policy = GgufTokenizerFields.specialTokens(fields, specials.values.toSet()).let { if (it.unk == null) it.copy(unk = unknownId) else it }

            return SentencePieceTokenizer(
                tokens = tokens,
                idToScore = scores,
                unknownTokenId = unknownId,
                bosTokenId = policy.bos,
                eosTokenId = policy.eos,
                addSpacePrefix = addSpacePrefix,
                policy = policy,
                source = "gguf",
            )
        }

        /**
         * Build from a parsed `tokenizer.json` root with `model.type == "Unigram"` using the
         * score-greedy merge above. Kept for source compatibility with 0.57; the resolver routes
         * Unigram files to [UnigramTokenizer], the reference lattice algorithm.
         */
        @Deprecated(
            message = "Unigram tokenizer.json files are decoded with the Viterbi lattice (UnigramTokenizer); use TextFrontEnds.fromTokenizerJson.",
            level = DeprecationLevel.WARNING,
        )
        public fun fromTokenizerJson(root: JsonObject): SentencePieceTokenizer {
            val model = root["model"]?.jsonObject ?: error("tokenizer.json missing 'model'")
            val vocabArr = model["vocab"]?.jsonArray ?: error("tokenizer.json missing 'model.vocab'")
            val tokens = ArrayList<String>(vocabArr.size)
            val scores = ArrayList<Float>(vocabArr.size)
            for (entry in vocabArr) {
                val pair = entry.jsonArray
                tokens.add(pair[0].jsonPrimitive.content)
                val raw = pair[1].jsonPrimitive
                scores.add(raw.doubleOrNull?.toFloat() ?: raw.float)
            }
            val json = TokenizerJson(root)
            val policy = json.specialTokens(null, ArrayVocabulary(tokens))
            val metaspace = PreTokenizer.metaspace(json.preTokenizer)
            val addSpacePrefix = (metaspace?.prependScheme ?: "never") != "never" || json.normalizerPrepends
            return SentencePieceTokenizer(
                tokens = tokens,
                scores = scores,
                unknownTokenId = model["unk_id"]?.jsonPrimitive?.int,
                bosTokenId = policy.bos,
                eosTokenId = policy.eos,
                addSpacePrefix = addSpacePrefix,
                policy = policy,
                source = TokenizerJson.FILE_NAME,
            )
        }
    }
}
