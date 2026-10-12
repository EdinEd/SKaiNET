package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * SentencePiece-style BPE as exported to `tokenizer.json` (`model.type: BPE` with a `Metaspace`
 * pre-tokenizer or a `Prepend ▁` / `Replace " " → "▁"` normalizer, and usually
 * `byte_fallback: true`): Llama 2 / TinyLlama, Mistral v0.1, Gemma, mmBERT (SKaiNET #1326, the
 * encode side of #912).
 *
 * Unlike [QwenByteLevelBpeTokenizer] the alphabet is plain code points with `▁` for space; a
 * character missing from the vocab falls back to its UTF-8 bytes as `<0xNN>` tokens when
 * [byteFallback] is set, otherwise to [unkId]. Merges are applied by rank ([BpeCore]).
 *
 * Added tokens are matched by the provider's [SpecialTokenSplitter]; this class is the algorithm.
 */
public class MetaspaceBpeTokenizer(
    tokens: List<String>,
    merges: List<Pair<String, String>>,
    public val unkId: Int?,
    public val byteFallback: Boolean,
    private val pipeline: TextPipeline,
    policy: SpecialTokens = SpecialTokens.NONE,
    private val ignoreMerges: Boolean = false,
    source: String? = null,
) : Tokenizer, RunDecoder {

    private val pieces: ArrayVocabulary = ArrayVocabulary(tokens)
    private val bpe = BpeCore(merges)
    private val byteTokenIds: IntArray = IntArray(256) { b -> pieces.tokenToId(SentencePieceTokenizer.byteTokenString(b)) ?: -1 }

    override val vocabulary: Vocabulary = pieces
    override val specialTokens: SpecialTokens = if (policy.unk == null && unkId != null) policy.copy(unk = unkId) else policy

    override val spec: FrontEndSpec = FrontEndSpec.Text(
        algorithm = ALGORITHM,
        artifact = source,
        params = JsonObject(
            mapOf(
                "byte_fallback" to JsonPrimitive(byteFallback),
                "normalizer" to JsonPrimitive(pipeline.normalizer.name),
                "pre_tokenizer" to JsonPrimitive(pipeline.preTokenizer.name),
            ),
        ),
    )

    public val mergeCount: Int get() = bpe.size

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val out = ArrayList<Int>(text.length)
        for (chunk in pipeline.chunks(text)) {
            if (chunk.isEmpty()) continue
            if (ignoreMerges) {
                val whole = pieces.tokenToId(chunk)
                if (whole != null) { out.add(whole); continue }
            }
            val symbols = initialSymbols(chunk)
            for (piece in bpe.merge(symbols)) {
                val id = pieces.tokenToId(piece)
                if (id != null) { out.add(id); continue }
                if (!emitFallback(piece, out)) error("BPE produced symbol not in vocab and no fallback: '$piece' (from chunk '$chunk')")
            }
        }
        return IntArray(out.size) { out[it] }
    }

    /** Code-point symbols; a character absent from the vocab becomes its `<0xNN>` byte tokens up front, as the reference does. */
    private fun initialSymbols(chunk: String): MutableList<String> {
        val symbols = ArrayList<String>(chunk.length)
        var i = 0
        while (i < chunk.length) {
            val c = chunk[i]
            val w = if (c.isHighSurrogate() && i + 1 < chunk.length && chunk[i + 1].isLowSurrogate()) 2 else 1
            val sym = chunk.substring(i, i + w)
            if (pieces.tokenToId(sym) == null && byteFallback) {
                for (b in sym.encodeToByteArray()) {
                    val tok = SentencePieceTokenizer.byteTokenString(b.toInt() and 0xFF)
                    if (pieces.tokenToId(tok) != null) symbols.add(tok) else symbols.add(sym)
                }
            } else {
                symbols.add(sym)
            }
            i += w
        }
        return symbols
    }

    private fun emitFallback(piece: String, out: ArrayList<Int>): Boolean {
        if (byteFallback) {
            val bytes = piece.encodeToByteArray()
            if (bytes.all { byteTokenIds[it.toInt() and 0xFF] >= 0 }) {
                for (b in bytes) out.add(byteTokenIds[b.toInt() and 0xFF])
                return true
            }
        }
        val unk = unkId ?: return false
        if (out.isEmpty() || out.last() != unk) out.add(unk)   // fuse_unk
        return true
    }

    override fun decode(ids: IntArray): String = decode(ids, stripLeading = pipeline.stripLeadingSpaces)

    override fun decodeToken(id: Int): String = decode(intArrayOf(id), stripLeading = 0)

    override val leadingSpacesToStrip: Int get() = pipeline.stripLeadingSpaces
    override fun decodeRun(ids: IntArray): String = decode(ids, stripLeading = 0)

    private fun decode(ids: IntArray, stripLeading: Int): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val token = pieces.idToToken(id) ?: error("decode: unknown token id $id")
            val byte = SentencePieceTokenizer.parseByteToken(token)
            if (byte != null) { byteBuf.add(byte); continue }
            if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
            sb.append(token)
        }
        if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
        var result = sb.toString().replace(SentencePieceTokenizer.WHITESPACE_ESCAPE, ' ')
        var strip = stripLeading
        while (strip > 0 && result.startsWith(' ')) { result = result.substring(1); strip-- }
        return result
    }

    private fun flushBytes(buf: ArrayList<Byte>): String {
        val arr = ByteArray(buf.size) { buf[it] }
        buf.clear()
        return arr.decodeToString()
    }

    public companion object {
        public const val ALGORITHM: String = "metaspace-bpe"

        internal fun fromTokenizerJson(json: TokenizerJson, config: TokenizerConfig?): MetaspaceBpeTokenizer {
            val tokens = json.vocabArray()
            val policy = json.specialTokens(config, ArrayVocabulary(tokens))
            return MetaspaceBpeTokenizer(
                tokens = tokens,
                merges = json.merges(),
                unkId = json.unkId(ArrayVocabulary(tokens)),
                byteFallback = json.byteFallback,
                pipeline = json.pipeline(),
                policy = policy,
                ignoreMerges = json.ignoreMerges,
                source = TokenizerJson.FILE_NAME,
            )
        }
    }
}
