package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import sk.ainet.context.ExecutionContext
import sk.ainet.frontend.FrontEnd
import sk.ainet.frontend.FrontEndProvider
import sk.ainet.frontend.FrontEndResolver
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.Modality
import sk.ainet.frontend.UnsupportedFrontEndException
import sk.ainet.frontend.text.TextFrontEnd
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFamilyRegistry
import sk.ainet.model.ModelFormat
import sk.ainet.model.ModelIdentity
import kotlin.jvm.JvmOverloads
import kotlin.jvm.JvmStatic

/**
 * The text modality's [FrontEndProvider] and the static entry points for building a
 * [Tokenizer] from one source (SKEEP-006 §Resolution, dispatch table FE-R-02).
 *
 * Evidence order for a [CheckpointView]: a `frontend.json` text spec, `tekken.json`,
 * `tokenizer.json` (+ `tokenizer_config.json`), the GGUF `tokenizer.ggml.*` fields, `vocab.txt`.
 *
 * ```kotlin
 * val resolver = TextFrontEnds.resolver()                 // builtIn registry, no ExecutionContext needed
 * val text = resolver.resolveText(GgufCheckpointView(reader))
 * val ids = text.tokenizer.encode("Hello", addSpecialTokens = true)
 * ```
 */
public object TextFrontEnds : FrontEndProvider {

    override val modality: Modality get() = Modality.TEXT

    /** A [FrontEndResolver] with this provider only. */
    @JvmStatic
    @JvmOverloads
    public fun resolver(registry: ModelFamilyRegistry = ModelFamilyRegistry.builtIn, ctx: ExecutionContext? = null): FrontEndResolver =
        FrontEndResolver(listOf(this), registry, ctx)

    override fun describe(view: CheckpointView, identity: ModelIdentity): FrontEndSpec? {
        explicitSpec(view)?.let { return it }
        if (view.sideFile(CheckpointView.TEKKEN_JSON) != null) return FrontEndSpec.Text(TekkenTokenizer.ALGORITHM, CheckpointView.TEKKEN_JSON)
        view.sideFileText(CheckpointView.TOKENIZER_JSON)?.let { json ->
            val parsed = TokenizerJson.parse(json)
            return FrontEndSpec.Text(algorithmFor(parsed), CheckpointView.TOKENIZER_JSON, JsonObject(mapOf("model_type" to JsonPrimitive(parsed.modelType))))
        }
        (view.metadata[GgufTokenizerFields.MODEL] as? String)?.let { model ->
            return FrontEndSpec.Text(ggufAlgorithm(model), "gguf", JsonObject(mapOf("tokenizer.ggml.model" to JsonPrimitive(model))))
        }
        if (view.sideFile(CheckpointView.VOCAB_TXT) != null) return FrontEndSpec.Text(WordPieceTokenizer.ALGORITHM, CheckpointView.VOCAB_TXT)
        return null
    }

    override fun build(view: CheckpointView, identity: ModelIdentity, ctx: ExecutionContext?): FrontEnd<*, *>? {
        val configText = view.sideFileText(CheckpointView.TOKENIZER_CONFIG_JSON)
        val config = configText?.let { TokenizerConfig.parse(it) }
        val tokenizer: Tokenizer = when {
            explicitSpec(view) != null -> fromExplicitSpec(view, explicitSpec(view)!!, config)
            view.sideFile(CheckpointView.TEKKEN_JSON) != null -> fromTekkenJson(view.sideFileText(CheckpointView.TEKKEN_JSON)!!)
            view.sideFile(CheckpointView.TOKENIZER_JSON) != null -> fromTokenizerJson(view.sideFileText(CheckpointView.TOKENIZER_JSON)!!, configText)
            view.metadata[GgufTokenizerFields.MODEL] != null -> fromGguf(view.metadata)
            view.sideFile(CheckpointView.VOCAB_TXT) != null -> fromVocabTxt(view.sideFileText(CheckpointView.VOCAB_TXT)!!, config?.doLowerCase ?: true)
            else -> return null
        }
        val chatTemplate = (view.metadata[GgufTokenizerFields.CHAT_TEMPLATE] as? String) ?: config?.chatTemplate
        return TextFrontEnd(tokenizer, chatTemplate, null)
    }

    private fun explicitSpec(view: CheckpointView): FrontEndSpec.Text? {
        val json = view.sideFileText(CheckpointView.FRONTEND_JSON) ?: return null
        return FrontEndSpec.listFromJson(json).filterIsInstance<FrontEndSpec.Text>().firstOrNull()
    }

    private fun fromExplicitSpec(view: CheckpointView, spec: FrontEndSpec.Text, config: TokenizerConfig?): Tokenizer {
        val artifact = spec.artifact ?: throw UnsupportedFrontEndException("frontend.json text spec needs an 'artifact' naming the side file")
        return when {
            artifact == "gguf" -> fromGguf(view.metadata)
            artifact.endsWith("tekken.json") -> fromTekkenJson(view.sideFileText(artifact) ?: missing(view, artifact))
            artifact.endsWith("vocab.txt") -> fromVocabTxt(view.sideFileText(artifact) ?: missing(view, artifact), config?.doLowerCase ?: true)
            else -> fromTokenizerJson(view.sideFileText(artifact) ?: missing(view, artifact), view.sideFileText(CheckpointView.TOKENIZER_CONFIG_JSON))
        }
    }

    private fun missing(view: CheckpointView, name: String): Nothing =
        throw UnsupportedFrontEndException("frontend.json names '$name' but the checkpoint has only ${view.listSideFiles().sorted()}")

    // ------------------------------------------------------------------
    // Static entry points (one source each)
    // ------------------------------------------------------------------

    /**
     * Build a tokenizer from GGUF metadata fields (`tokenizer.ggml.*`). Dispatches on
     * `tokenizer.ggml.model`: `gpt2`/`bpe` → byte-level BPE; `llama`/`sentencepiece`/`gemma4` →
     * SentencePiece wrapped in a [SpecialTokenSplitter] when CONTROL/USER_DEFINED tokens exist;
     * `bert`/`wordpiece` → WordPiece over the token list.
     */
    @JvmStatic
    public fun fromGguf(fields: Map<String, Any?>): Tokenizer {
        val model = (fields[GgufTokenizerFields.MODEL] as? String)?.lowercase()
            ?: throw UnsupportedFrontEndException("GGUF metadata has no '${GgufTokenizerFields.MODEL}' field")
        return when (model) {
            "gpt2", "bpe" -> QwenByteLevelBpeTokenizer.fromGgufFields(fields)
            "llama", "sentencepiece", "gemma4" -> {
                val base = SentencePieceTokenizer.fromGgufFields(fields)
                val specials = GgufTokenizerFields.specialTokenMap(fields)
                if (specials.isEmpty()) base else SpecialTokenSplitter(base, specials, specialIds = specials.values.toSet())
            }
            "bert", "wordpiece" -> {
                val tokens = GgufTokenizerFields.tokens(fields)
                val vocab = LinkedHashMap<String, Int>(tokens.size * 2)
                for (i in tokens.indices) if (tokens[i] !in vocab) vocab[tokens[i]] = i
                // uncased vocab ⇔ no upper-case letter outside the bracketed / angled special tokens
                val lower = tokens.none { t -> !(t.startsWith("[") || t.startsWith("<")) && t.any { it.isUpperCase() } }
                WordPieceTokenizer(vocab, pipeline = if (lower) WordPieceTokenizer.DEFAULT_PIPELINE else WordPieceTokenizer.CASED_PIPELINE, source = "gguf")
            }
            else -> throw UnsupportedFrontEndException("Unknown GGUF ${GgufTokenizerFields.MODEL}: '$model'")
        }
    }

    /**
     * Build a tokenizer from a `tokenizer.json` string and, optionally, its sibling
     * `tokenizer_config.json` (FE-R-03). Dispatch: `BPE` with a byte-level pre-tokenizer → byte-level
     * BPE; `BPE` with Metaspace / byte fallback → [MetaspaceBpeTokenizer]; `Unigram` →
     * [UnigramTokenizer]; `WordPiece` → [WordPieceTokenizer]. Added tokens are matched atomically
     * for every algorithm (#1324).
     */
    @JvmStatic
    @JvmOverloads
    public fun fromTokenizerJson(json: String, configJson: String? = null): Tokenizer {
        val parsed = TokenizerJson.parse(json)
        val config = configJson?.let { TokenizerConfig.parse(it) }
        return fromTokenizerJson(parsed, config)
    }

    /** As [fromTokenizerJson] over an already parsed root object. */
    @JvmStatic
    public fun fromTokenizerJson(root: JsonObject): Tokenizer = fromTokenizerJson(TokenizerJson(root), null)

    internal fun fromTokenizerJson(parsed: TokenizerJson, config: TokenizerConfig?): Tokenizer {
        val algorithm = algorithmFor(parsed)
        val base: Tokenizer = when (algorithm) {
            QwenByteLevelBpeTokenizer.ALGORITHM -> return QwenByteLevelBpeTokenizer.fromTokenizerJson(parsed, config)  // handles its own added tokens
            MetaspaceBpeTokenizer.ALGORITHM -> MetaspaceBpeTokenizer.fromTokenizerJson(parsed, config)
            UnigramTokenizer.ALGORITHM -> UnigramTokenizer.fromTokenizerJson(parsed, config)
            WordPieceTokenizer.ALGORITHM -> WordPieceTokenizer.fromTokenizerJson(parsed, config)
            else -> throw UnsupportedFrontEndException("Unknown tokenizer.json model.type: '${parsed.modelType}'")
        }
        val added = parsed.addedTokens
        if (added.isEmpty() || base is WordPieceTokenizer) return base   // WordPiece matches its added tokens itself
        // Flags travel with the tokens: `<mask>` is `lstrip` in the ModernBERT / mmBERT files, so the
        // whitespace before it belongs to the special token, not to the preceding segment.
        return SpecialTokenSplitter(base = base, tokens = added, bosTokenId = base.specialTokens.bos, eosTokenId = base.specialTokens.eos)
    }

    /** Build the Mistral Tekken tokenizer from a `tekken.json` string. */
    @JvmStatic
    public fun fromTekkenJson(json: String): TekkenTokenizer = TekkenTokenizer.fromJson(json)

    /** Build a WordPiece tokenizer from a BERT `vocab.txt`. */
    @JvmStatic
    @JvmOverloads
    public fun fromVocabTxt(content: String, doLowerCase: Boolean = true): WordPieceTokenizer = WordPieceTokenizer.fromVocabTxt(content, doLowerCase)

    /** The algorithm the dispatch table selects for a parsed `tokenizer.json`. */
    public fun algorithmFor(parsed: TokenizerJson): String = when (parsed.modelType) {
        "BPE" -> {
            val pre = parsed.preTokenizer
            val metaspaceSignals = PreTokenizer.metaspace(pre) != null || parsed.normalizerPrepends || parsed.byteFallback || replacesSpaceWithMetaspace(parsed)
            if (PreTokenizer.isByteLevel(pre) || !metaspaceSignals) QwenByteLevelBpeTokenizer.ALGORITHM else MetaspaceBpeTokenizer.ALGORITHM
        }
        "Unigram" -> UnigramTokenizer.ALGORITHM
        "WordPiece" -> WordPieceTokenizer.ALGORITHM
        else -> parsed.modelType.lowercase()
    }

    private fun replacesSpaceWithMetaspace(parsed: TokenizerJson): Boolean {
        val n = parsed.root["normalizer"] as? JsonObject ?: return false
        fun visit(o: JsonObject): Boolean = when ((o["type"] as? JsonPrimitive)?.content) {
            "Replace" -> (o["pattern"] as? JsonObject)?.get("String")?.let { (it as? JsonPrimitive)?.content } == " " && (o["content"] as? JsonPrimitive)?.content == "▁"
            "Sequence" -> (o["normalizers"] as? kotlinx.serialization.json.JsonArray)?.any { (it as? JsonObject)?.let(::visit) ?: false } ?: false
            else -> false
        }
        return visit(n)
    }

    private fun ggufAlgorithm(model: String): String = when (model.lowercase()) {
        "gpt2", "bpe" -> QwenByteLevelBpeTokenizer.ALGORITHM
        "llama", "sentencepiece", "gemma4" -> SentencePieceTokenizer.ALGORITHM
        "bert", "wordpiece" -> WordPieceTokenizer.ALGORITHM
        else -> model.lowercase()
    }

    /** The format a view's text evidence belongs to, for messages. */
    internal fun formatOf(view: CheckpointView): ModelFormat = view.format
}
