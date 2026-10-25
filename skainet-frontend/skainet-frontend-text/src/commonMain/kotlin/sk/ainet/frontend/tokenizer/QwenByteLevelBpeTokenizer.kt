package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * GPT-2-style byte-level BPE tokenizer (Qwen, GPT-2, Mistral-Nemo, ModernBERT, …).
 *
 * Implements the encoding pipeline of the Hugging Face `tokenizers` library and llama.cpp for
 * byte-level BPE:
 *
 * 1. Match added tokens atomically (`<|im_start|>`, `<|endoftext|>`, …).
 * 2. For each remaining segment apply the file's normalizer (e.g. NFC) and pre-tokenizer
 *    (the GPT-2 regex, or a `Split` regex followed by `ByteLevel`).
 * 3. Map every chunk's UTF-8 bytes to the byte-level alphabet via [ByteToUnicode] (so "Hello"
 *    becomes `Hello`, " is" becomes `Ġis`, "\n" becomes `Ċ`).
 * 4. Apply the merges by **lowest merge rank first** ([BpeCore]).
 * 5. Look up each symbol in the vocab.
 *
 * Decode is the reverse: concatenate token strings, reverse the byte map, UTF-8-decode.
 *
 * @param tokens vocab indexed by id (the 256 base glyphs, every merged symbol, and any added tokens)
 * @param merges merge list in priority order; rank 0 is the highest-priority merge
 * @param specialTokens added tokens by surface form → id; matched atomically, longest first
 * @param bosTokenId optional BOS id (0.57 signature; never emitted by the one-argument [encode])
 * @param eosTokenId optional EOS id
 * @param pipeline normalizer and pre-tokenizer (defaults to the GPT-2 regex)
 * @param policy the full [SpecialTokens]; derived from the other arguments when `null`
 * @param ignoreMerges `model.ignore_merges`: a chunk that is itself a vocab entry is one token
 * @param source the side file this tokenizer was read from, for [spec]
 */
public class QwenByteLevelBpeTokenizer(
    tokens: List<String>,
    merges: List<Pair<String, String>>,
    specialTokens: Map<String, Int>,
    bosTokenId: Int? = null,
    eosTokenId: Int? = null,
    pipeline: TextPipeline = DEFAULT_PIPELINE,
    policy: SpecialTokens? = null,
    private val ignoreMerges: Boolean = false,
    source: String? = null,
) : Tokenizer {

    private val pieces: ArrayVocabulary = ArrayVocabulary(tokens)
    private val added = AddedTokenMatcher(specialTokens.map { (content, id) -> AddedToken(content, id, special = policy?.declared?.contains(id) ?: true) })
    private val bpe = BpeCore(merges)
    private val pipeline: TextPipeline = pipeline
    private val byteLevelPre: Boolean = PreTokenizer.isByteLevel(pipeline.preTokenizer)

    override val vocabulary: Vocabulary = ExtendedVocabulary(pieces, added.contentToId)

    override val specialTokens: SpecialTokens = policy ?: SpecialTokens(
        bos = bosTokenId,
        eos = eosTokenId,
        declared = added.ids,
    )

    override val spec: FrontEndSpec = FrontEndSpec.Text(
        algorithm = ALGORITHM,
        artifact = source,
        params = JsonObject(mapOf("normalizer" to JsonPrimitive(pipeline.normalizer.name), "pre_tokenizer" to JsonPrimitive(pipeline.preTokenizer.name))),
    )

    /** Number of merges actually registered — the parity check of SKaiNET #1323. */
    public val mergeCount: Int get() = bpe.size

    override fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length)
        added.encode(text, out) { segment -> encodeSegment(segment, out) }
        return IntArray(out.size) { out[it] }
    }

    override fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val special = added.idToContent[id]
            if (special != null) {
                if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
                sb.append(special)
                continue
            }
            val token = pieces.idToToken(id)
                ?: error("decode: unknown token id $id")
            for (c in token) {
                val b = ByteToUnicode.unicodeToByte[c]
                    ?: error("decode: token '$token' contains non-byte-level char U+${c.code.toString(16)}")
                byteBuf.add(b)
            }
        }
        if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
        return sb.toString()
    }

    private fun flushBytes(buf: ArrayList<Byte>): String {
        val arr = ByteArray(buf.size) { buf[it] }
        buf.clear()
        return arr.decodeToString()
    }

    private fun encodeSegment(segment: String, out: ArrayList<Int>) {
        if (segment.isEmpty()) return
        for (chunk in pipeline.chunks(segment)) {
            if (chunk.isEmpty()) continue
            val glyphs = if (byteLevelPre) chunk else ByteToUnicode.encode(chunk)
            if (ignoreMerges) {
                val whole = pieces.tokenToId(glyphs)
                if (whole != null) { out.add(whole); continue }
            }
            for (piece in bpe.mergeWord(glyphs)) {
                val id = pieces.tokenToId(piece)
                    ?: error("BPE produced symbol not in vocab: '$piece' (from chunk '$chunk')")
                out.add(id)
            }
        }
    }

    public companion object {
        public const val ALGORITHM: String = "bpe"

        /** The GPT-2 pipeline: no normalizer, GPT-2 regex, byte-level alphabet. */
        public val DEFAULT_PIPELINE: TextPipeline = TextPipeline(Normalizer.None, PreTokenizer.ByteLevel())

        /**
         * Build from GGUF metadata fields (`tokenizer.ggml.*`). Every token whose token_type is
         * CONTROL (3) or USER_DEFINED (4) is an atomic added token. Merges arrive as
         * space-separated `"first second"` strings.
         */
        public fun fromGgufFields(fields: Map<String, Any?>): QwenByteLevelBpeTokenizer {
            val tokens = GgufTokenizerFields.tokens(fields)
            val mergesRaw = (fields[GgufTokenizerFields.MERGES] as? List<*>)
                ?.filterIsInstance<String>()
                ?: error("${GgufTokenizerFields.MERGES} missing — required for byte-level BPE")
            val merges = mergesRaw.map { BpeCore.parseMergeLine(it) }
            val specials = GgufTokenizerFields.specialTokenMap(tokens, GgufTokenizerFields.tokenTypeIds(fields))
            val policy = GgufTokenizerFields.specialTokens(fields, specials.values.toSet())
            return QwenByteLevelBpeTokenizer(
                tokens = tokens,
                merges = merges,
                specialTokens = specials,
                bosTokenId = policy.bos,
                eosTokenId = policy.eos,
                policy = policy,
                source = "gguf",
            )
        }

        /**
         * Build from a parsed Hugging Face `tokenizer.json` root object with `model.type == "BPE"`
         * and a byte-level pre-tokenizer. Accepts both merge forms (#1323), registers every added
         * token (#1324) and applies the normalizer (#1325). Prefer `TextFrontEnds.fromTokenizerJson`,
         * which also reads `tokenizer_config.json`.
         */
        public fun fromTokenizerJson(root: JsonObject): QwenByteLevelBpeTokenizer =
            fromTokenizerJson(TokenizerJson(root), config = null)

        internal fun fromTokenizerJson(json: TokenizerJson, config: TokenizerConfig?): QwenByteLevelBpeTokenizer {
            val tokens = json.vocabArray()
            val pipeline = json.pipeline()
            val pre = if (PreTokenizer.isByteLevel(pipeline.preTokenizer) || pipeline.preTokenizer !is PreTokenizer.None) pipeline.preTokenizer else PreTokenizer.ByteLevel()
            val effective = TextPipeline(pipeline.normalizer, pre, pipeline.stripLeadingSpaces)
            val added = json.addedTokens.associate { it.content to it.id }
            val policy = json.specialTokens(config, ExtendedVocabulary(ArrayVocabulary(tokens), added))
            return QwenByteLevelBpeTokenizer(
                tokens = tokens,
                merges = json.merges(),
                specialTokens = added,
                bosTokenId = policy.bos,
                eosTokenId = policy.eos,
                pipeline = effective,
                policy = policy,
                ignoreMerges = json.ignoreMerges,
                source = TokenizerJson.FILE_NAME,
            )
        }
    }
}
