package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.BatchTokenizer
import sk.ainet.frontend.text.Encoding
import sk.ainet.frontend.text.MapVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Truncation
import sk.ainet.frontend.text.Vocabulary

/**
 * WordPiece tokenizer for BERT-class encoders, matching the reference library: the
 * `BertNormalizer` (clean-up, CJK spacing, lower-casing, accent stripping), the
 * `BertPreTokenizer` (whitespace and punctuation), then greedy longest-match-first sub-word
 * lookup with the `##` continuation prefix; a word longer than [maxInputCharsPerWord] or with an
 * uncovered character becomes `[UNK]`.
 *
 * The one-argument [encode] returns the piece ids only; [encodeWithMetadata] produces the
 * `[CLS] … [SEP]` encoding with attention mask and segment ids BERT models consume (FE-C-08),
 * and `encode(text, addSpecialTokens = true)` wraps with `[CLS]`/`[SEP]` through [specialTokens].
 * Ported from the SKaiNET-transformers bert module with a real JSON parser.
 */
public class WordPieceTokenizer(
    vocab: Map<String, Int>,
    public val unkToken: String = "[UNK]",
    public val continuingSubwordPrefix: String = "##",
    public val maxInputCharsPerWord: Int = 100,
    private val pipeline: TextPipeline = DEFAULT_PIPELINE,
    policy: SpecialTokens? = null,
    source: String? = null,
    addedTokens: List<AddedToken> = emptyList(),
    /** Reference `WordPiece` decoder `cleanup`: no space before `.`, `?`, `!`, `,` and around contractions. */
    public val cleanup: Boolean = true,
) : BatchTokenizer {

    private val vocab: MapVocabulary = MapVocabulary(vocab)
    private val unkId: Int = vocab[unkToken] ?: error("WordPiece vocab is missing the unknown token '$unkToken'")
    private val added = AddedTokenMatcher(
        addedTokens.ifEmpty { listOf(CLS, SEP, PAD, MASK, unkToken).mapNotNull { name -> vocab[name]?.let { AddedToken(name, it, special = true) } } },
    )

    override val vocabulary: Vocabulary = this.vocab

    override val specialTokens: SpecialTokens = policy ?: SpecialTokens(
        bos = vocab[CLS], eos = vocab[SEP], pad = vocab[PAD], unk = unkId,
        addBos = vocab[CLS] != null, addEos = vocab[SEP] != null,
        declared = setOfNotNull(vocab[CLS], vocab[SEP], vocab[PAD], vocab[MASK], unkId),
    )

    override val spec: FrontEndSpec = FrontEndSpec.Text(
        algorithm = ALGORITHM,
        artifact = source,
        params = JsonObject(mapOf("normalizer" to JsonPrimitive(pipeline.normalizer.name), "continuing_subword_prefix" to JsonPrimitive(continuingSubwordPrefix))),
    )

    /** Pieces of [text] after added-token matching, normalisation, pre-tokenization and WordPiece lookup. */
    public fun tokenize(text: String): List<String> = tokenizeOrdered(text)

    private fun tokenizeOrdered(text: String): List<String> {
        val result = ArrayList<String>()
        val sink = object : AbstractMutableList<Int>() {
            override val size: Int get() = 0
            override fun get(index: Int): Int = throw IndexOutOfBoundsException()
            override fun set(index: Int, element: Int): Int = throw IndexOutOfBoundsException()
            override fun removeAt(index: Int): Int = throw IndexOutOfBoundsException()
            override fun add(index: Int, element: Int) { result.add(added.idToContent[element] ?: unkToken) }
        }
        added.encode(text, sink) { segment -> for (word in pipeline.chunks(segment)) wordPiece(word, result) }
        return result
    }

    override fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length)
        added.encode(text, out) { segment ->
            val pieces = ArrayList<String>()
            for (word in pipeline.chunks(segment)) wordPiece(word, pieces)
            for (piece in pieces) out.add(vocab.tokenToId(piece) ?: unkId)
        }
        return IntArray(out.size) { out[it] }
    }

    override fun encodeWithMetadata(text: String, pair: String?, truncation: Truncation): Encoding {
        val cls = specialTokens.bos
        val sep = specialTokens.eos
        val a = encode(text)
        val b = pair?.let { encode(it) }
        val ids = ArrayList<Int>(a.size + (b?.size ?: 0) + 3)
        val types = ArrayList<Int>()
        val specialMask = ArrayList<Int>()
        fun add(id: Int, type: Int, special: Boolean) { ids.add(id); types.add(type); specialMask.add(if (special) 1 else 0) }
        if (cls != null) add(cls, 0, true)
        for (id in a) add(id, 0, false)
        if (sep != null) add(sep, 0, true)
        if (b != null) {
            for (id in b) add(id, 1, false)
            if (sep != null) add(sep, 1, true)
        }
        var n = ids.size
        if (truncation is Truncation.Longest && n > truncation.maxLength) {
            n = truncation.maxLength
            if (sep != null && n > 0) { ids[n - 1] = sep; specialMask[n - 1] = 1 }
        }
        return Encoding(
            ids = IntArray(n) { ids[it] },
            attentionMask = IntArray(n) { 1 },
            tokenTypeIds = IntArray(n) { types[it] },
            specialTokensMask = IntArray(n) { specialMask[it] },
        )
    }

    /** Reference WordPiece decoder: pieces joined with spaces, `##` continuations glued on, then [cleanup]. */
    override fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        for (id in ids) {
            val piece = vocab.idToToken(id) ?: unkToken
            if (piece.startsWith(continuingSubwordPrefix)) {
                sb.append(piece, continuingSubwordPrefix.length, piece.length)
            } else {
                if (sb.isNotEmpty()) sb.append(' ')
                sb.append(piece)
            }
        }
        return if (cleanup) cleanUp(sb.toString()) else sb.toString()
    }

    override fun decodeToken(id: Int): String = vocab.idToToken(id) ?: unkToken

    private fun wordPiece(word: String, out: MutableList<String>) {
        if (word.isEmpty()) return
        if (word.length > maxInputCharsPerWord) { out.add(unkToken); return }
        var start = 0
        val pieces = ArrayList<String>()
        while (start < word.length) {
            var end = word.length
            var found: String? = null
            while (start < end) {
                val sub = if (start == 0) word.substring(start, end) else continuingSubwordPrefix + word.substring(start, end)
                if (vocab.tokenToId(sub) != null) { found = sub; break }
                end--
            }
            if (found == null) { out.add(unkToken); return }
            pieces.add(found)
            start = end
        }
        out.addAll(pieces)
    }

    public companion object {
        /** The reference decoder's `cleanup` table (`tokenizers`, `decoders/wordpiece.rs`). */
        internal fun cleanUp(s: String): String = s
            .replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
            .replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't")
            .replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")

        public const val ALGORITHM: String = "wordpiece"
        public const val CLS: String = "[CLS]"
        public const val SEP: String = "[SEP]"
        public const val PAD: String = "[PAD]"
        public const val MASK: String = "[MASK]"

        /** The `bert-base-uncased` pipeline: BertNormalizer (lowercase, strip accents), BertPreTokenizer. */
        public val DEFAULT_PIPELINE: TextPipeline = TextPipeline(Normalizer.Bert(stripAccents = null, lowercase = true), PreTokenizer.Bert)

        /** Pipeline for cased vocabularies. */
        public val CASED_PIPELINE: TextPipeline = TextPipeline(Normalizer.Bert(stripAccents = false, lowercase = false), PreTokenizer.Bert)

        /** Parse a `vocab.txt` (one token per line, line number = id). */
        public fun fromVocabTxt(content: String, doLowerCase: Boolean = true): WordPieceTokenizer {
            val vocab = LinkedHashMap<String, Int>()
            var idx = 0
            for (line in content.lineSequence()) {
                val token = line.trimEnd('\r')
                if (token.isNotEmpty() || idx > 0) {       // keep ids aligned with line numbers
                    if (token.isNotEmpty()) if (token !in vocab) vocab[token] = idx
                    idx++
                }
            }
            require(vocab.isNotEmpty()) { "vocab.txt is empty" }
            return WordPieceTokenizer(vocab, pipeline = if (doLowerCase) DEFAULT_PIPELINE else CASED_PIPELINE, source = "vocab.txt")
        }

        internal fun fromTokenizerJson(json: TokenizerJson, config: TokenizerConfig?): WordPieceTokenizer {
            val vocab = json.vocabMap()
            val unk = json.model["unk_token"]?.let { (it as? JsonPrimitive)?.content } ?: "[UNK]"
            val prefix = json.model["continuing_subword_prefix"]?.let { (it as? JsonPrimitive)?.content } ?: "##"
            val max = json.model["max_input_chars_per_word"]?.let { (it as? JsonPrimitive)?.content?.toIntOrNull() } ?: 100
            val pipeline = json.pipeline().let { p ->
                // Files predating the normalizer block: fall back to the config's do_lower_case.
                if (p.normalizer is Normalizer.None && p.preTokenizer is PreTokenizer.None) {
                    if (config?.doLowerCase != false) DEFAULT_PIPELINE else CASED_PIPELINE
                } else p
            }
            val policy = json.specialTokens(config, MapVocabulary(vocab)).let { pol ->
                val cls = pol.bos ?: vocab[CLS]; val sep = pol.eos ?: vocab[SEP]
                pol.copy(
                    bos = cls, eos = sep,
                    pad = pol.pad ?: vocab[PAD], unk = pol.unk ?: vocab[unk],
                    addBos = cls != null && (pol.addBos || json.postProcessor == null),
                    addEos = sep != null && (pol.addEos || json.postProcessor == null),
                    declared = pol.declared + setOfNotNull(cls, sep, vocab[PAD], vocab[MASK], vocab[unk]),
                )
            }
            return WordPieceTokenizer(vocab, unk, prefix, max, pipeline, policy, TokenizerJson.FILE_NAME, addedTokens = json.addedTokens)
        }
    }
}
