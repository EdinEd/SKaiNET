package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * Unigram language-model tokenizer (`tokenizer.json` `model.type: Unigram`: T5, ALBERT, XLNet,
 * Gemma exports): the segmentation of each pre-tokenized chunk that maximises the summed piece
 * log-probabilities, found with the Viterbi lattice the reference library uses.
 *
 * A character no piece covers costs `minScore - UNK_PENALTY` and becomes [unkId] (consecutive
 * unknowns fused), or its `<0xNN>` byte tokens when [byteFallback] is set and they exist.
 */
public class UnigramTokenizer(
    pieces: List<String>,
    scores: List<Float>,
    public val unkId: Int?,
    public val byteFallback: Boolean,
    private val pipeline: TextPipeline,
    policy: SpecialTokens = SpecialTokens.NONE,
    source: String? = null,
) : Tokenizer, RunDecoder {

    private val vocab: ArrayVocabulary = ArrayVocabulary(pieces)
    private val idToScore: FloatArray = FloatArray(scores.size) { scores[it] }
    private val minScore: Float = scores.minOrNull() ?: 0f
    private val maxPieceLength: Int = pieces.maxOfOrNull { it.length } ?: 1
    private val byteTokenIds: IntArray = IntArray(256) { b -> vocab.tokenToId(SentencePieceTokenizer.byteTokenString(b)) ?: -1 }

    init {
        require(pieces.size == scores.size) { "pieces (${pieces.size}) and scores (${scores.size}) must have the same length" }
    }

    override val vocabulary: Vocabulary = vocab
    override val specialTokens: SpecialTokens = if (policy.unk == null && unkId != null) policy.copy(unk = unkId) else policy
    override val spec: FrontEndSpec = FrontEndSpec.Text(
        algorithm = ALGORITHM,
        artifact = source,
        params = JsonObject(mapOf("byte_fallback" to JsonPrimitive(byteFallback), "pre_tokenizer" to JsonPrimitive(pipeline.preTokenizer.name))),
    )

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val out = ArrayList<Int>(text.length)
        for (chunk in pipeline.chunks(text)) if (chunk.isNotEmpty()) viterbi(chunk, out)
        return IntArray(out.size) { out[it] }
    }

    private fun viterbi(chunk: String, out: ArrayList<Int>) {
        val n = chunk.length
        val best = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val backPiece = IntArray(n + 1) { -1 }     // id of the piece ending at i, or -1 for unk
        val backStart = IntArray(n + 1) { -1 }
        best[0] = 0f
        val unkScore = minScore - UNK_PENALTY
        for (start in 0 until n) {
            if (best[start] == Float.NEGATIVE_INFINITY) continue
            // never start inside a surrogate pair
            if (chunk[start].isLowSurrogate() && start > 0 && chunk[start - 1].isHighSurrogate()) continue
            val maxEnd = minOf(n, start + maxPieceLength)
            var covered = false
            for (end in start + 1..maxEnd) {
                if (end < n && chunk[end].isLowSurrogate() && chunk[end - 1].isHighSurrogate()) continue
                val id = vocab.tokenToId(chunk.substring(start, end)) ?: continue
                covered = covered || end == start + charWidth(chunk, start)
                val s = best[start] + idToScore[id]
                if (s > best[end]) { best[end] = s; backPiece[end] = id; backStart[end] = start }
            }
            if (!covered) {
                val end = start + charWidth(chunk, start)
                val s = best[start] + unkScore
                if (s > best[end]) { best[end] = s; backPiece[end] = -1; backStart[end] = start }
            }
        }
        // walk back
        val rev = ArrayList<Int>()
        var pos = n
        var lastWasUnk = false
        while (pos > 0) {
            val start = backStart[pos]
            val id = backPiece[pos]
            if (id >= 0) {
                rev.add(id); lastWasUnk = false
            } else {
                val sym = chunk.substring(start, pos)
                val bytes = sym.encodeToByteArray()
                if (byteFallback && bytes.all { byteTokenIds[it.toInt() and 0xFF] >= 0 }) {
                    for (i in bytes.indices.reversed()) rev.add(byteTokenIds[bytes[i].toInt() and 0xFF])
                    lastWasUnk = false
                } else {
                    val unk = unkId ?: error("Unigram: no piece covers '$sym' and the vocab has no unk id")
                    if (!lastWasUnk) rev.add(unk)   // fuse_unk
                    lastWasUnk = true
                }
            }
            pos = start
        }
        for (i in rev.indices.reversed()) out.add(rev[i])
    }

    private fun charWidth(s: String, i: Int): Int =
        if (s[i].isHighSurrogate() && i + 1 < s.length && s[i + 1].isLowSurrogate()) 2 else 1

    override fun decode(ids: IntArray): String = decode(ids, pipeline.stripLeadingSpaces)

    override fun decodeToken(id: Int): String = decode(intArrayOf(id), 0)

    override val leadingSpacesToStrip: Int get() = pipeline.stripLeadingSpaces
    override fun decodeRun(ids: IntArray): String = decode(ids, 0)

    private fun decode(ids: IntArray, stripLeading: Int): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val token = vocab.idToToken(id) ?: error("decode: unknown token id $id")
            val byte = SentencePieceTokenizer.parseByteToken(token)
            if (byte != null) { byteBuf.add(byte); continue }
            if (byteBuf.isNotEmpty()) { sb.append(ByteArray(byteBuf.size) { byteBuf[it] }.decodeToString()); byteBuf.clear() }
            sb.append(token)
        }
        if (byteBuf.isNotEmpty()) sb.append(ByteArray(byteBuf.size) { byteBuf[it] }.decodeToString())
        var result = sb.toString().replace(SentencePieceTokenizer.WHITESPACE_ESCAPE, ' ')
        var strip = stripLeading
        while (strip > 0 && result.startsWith(' ')) { result = result.substring(1); strip-- }
        return result
    }

    public companion object {
        public const val ALGORITHM: String = "unigram"
        public const val UNK_PENALTY: Float = 10f

        internal fun fromTokenizerJson(json: TokenizerJson, config: TokenizerConfig?): UnigramTokenizer {
            val (pieces, scores) = json.unigramVocab()
            val vocab = ArrayVocabulary(pieces)
            return UnigramTokenizer(
                pieces = pieces,
                scores = scores,
                unkId = json.unkId(vocab),
                byteFallback = json.byteFallback,
                pipeline = json.pipeline(),
                policy = json.specialTokens(config, vocab),
                source = TokenizerJson.FILE_NAME,
            )
        }
    }
}
