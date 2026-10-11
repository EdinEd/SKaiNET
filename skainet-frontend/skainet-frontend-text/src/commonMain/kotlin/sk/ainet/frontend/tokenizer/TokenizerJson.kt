package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.UnsupportedFrontEndException
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Vocabulary

/**
 * A parsed Hugging Face `tokenizer.json`: the facts every algorithm needs (model type and
 * tables, added tokens, normalizer, pre-tokenizer, post-processor specials, decoder), read
 * once with a real JSON parser.
 */
public class TokenizerJson(public val root: JsonObject) {

    public val model: JsonObject = root["model"]?.jsonObject
        ?: throw UnsupportedFrontEndException("$FILE_NAME has no 'model'")

    /**
     * `model.type`, or the type inferred from the model's shape for files that predate the field:
     * a `merges` list is unique to BPE, a `vocab` object with a continuation prefix or `unk_token`
     * is WordPiece, a `vocab` array of `[piece, score]` pairs is Unigram.
     */
    public val modelType: String = (model["type"] as? JsonPrimitive)?.content ?: when {
        model["merges"] != null -> "BPE"
        model["vocab"] is JsonArray -> "Unigram"
        model["vocab"] is JsonObject && (model["continuing_subword_prefix"] != null || model["unk_token"] != null || model["max_input_chars_per_word"] != null) -> "WordPiece"
        else -> throw UnsupportedFrontEndException("$FILE_NAME has no model.type and it could not be inferred from the model shape")
    }

    public val byteFallback: Boolean = model["byte_fallback"]?.jsonPrimitive?.booleanOrNull ?: false
    public val ignoreMerges: Boolean = model["ignore_merges"]?.jsonPrimitive?.booleanOrNull ?: false

    public val addedTokens: List<AddedToken> = (root["added_tokens"] as? JsonArray).orEmpty().mapNotNull { el ->
        val obj = el as? JsonObject ?: return@mapNotNull null
        val content = obj["content"]?.jsonPrimitive?.content ?: return@mapNotNull null
        val id = obj["id"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null
        AddedToken(
            content = content,
            id = id,
            special = obj["special"]?.jsonPrimitive?.booleanOrNull ?: true,
            lstrip = obj["lstrip"]?.jsonPrimitive?.booleanOrNull ?: false,
            rstrip = obj["rstrip"]?.jsonPrimitive?.booleanOrNull ?: false,
            singleWord = obj["single_word"]?.jsonPrimitive?.booleanOrNull ?: false,
            normalized = obj["normalized"]?.jsonPrimitive?.booleanOrNull ?: false,
        )
    }

    public val normalizer: Normalizer = Normalizer.fromJson(root["normalizer"].takeUnless { it is JsonNull })
    public val preTokenizer: PreTokenizer = PreTokenizer.fromJson(root["pre_tokenizer"].takeUnless { it is JsonNull })
    public val postProcessor: JsonObject? = root["post_processor"] as? JsonObject
    public val decoder: JsonObject? = root["decoder"] as? JsonObject

    /** Does the normalizer chain prepend the SentencePiece prefix (`Prepend "▁"`)? */
    public val normalizerPrepends: Boolean get() = hasPrepend(root["normalizer"])

    private fun hasPrepend(el: JsonElement?): Boolean {
        val obj = el as? JsonObject ?: return false
        return when (obj["type"]?.jsonPrimitive?.content) {
            "Prepend" -> obj["prepend"]?.jsonPrimitive?.content == "▁"
            "Sequence" -> obj["normalizers"]?.jsonArray.orEmpty().any { hasPrepend(it) }
            else -> false
        }
    }

    /** Normalizer + pre-tokenizer + how many leading spaces the decoder strips from a sequence. */
    public fun pipeline(): TextPipeline = TextPipeline(normalizer, preTokenizer, stripLeadingSpaces())

    private fun stripLeadingSpaces(): Int {
        var strip = 0
        fun visit(el: JsonElement?) {
            val obj = el as? JsonObject ?: return
            when (obj["type"]?.jsonPrimitive?.content) {
                "Sequence" -> obj["decoders"]?.jsonArray.orEmpty().forEach { visit(it) }
                "Strip" -> if (obj["content"]?.jsonPrimitive?.content == " ") strip += obj["start"]?.jsonPrimitive?.intOrNull ?: 0
                "Metaspace" -> {
                    val scheme = obj["prepend_scheme"]?.jsonPrimitive?.content
                        ?: if (obj["add_prefix_space"]?.jsonPrimitive?.booleanOrNull == false) "never" else "always"
                    if (scheme != "never") strip += 1
                }
            }
        }
        visit(decoder)
        return strip
    }

    /** `model.vocab` as an id-indexed array (BPE / WordPiece object form); gaps are empty strings; added tokens above the range extend it (#1324). */
    public fun vocabArray(): List<String> {
        val vocab = model["vocab"]?.jsonObject ?: throw UnsupportedFrontEndException("$FILE_NAME has no 'model.vocab'")
        val maxVocabId = vocab.values.maxOfOrNull { it.jsonPrimitive.intOrNull ?: -1 } ?: -1
        val maxAddedId = addedTokens.maxOfOrNull { it.id } ?: -1
        val tokens = Array(maxOf(maxVocabId, maxAddedId) + 1) { "" }
        for ((tok, idEl) in vocab) {
            val id = idEl.jsonPrimitive.intOrNull ?: continue
            tokens[id] = tok
        }
        for (a in addedTokens) if (tokens[a.id].isEmpty()) tokens[a.id] = a.content
        return tokens.asList()
    }

    /** `model.vocab` as a piece → id map (WordPiece), with added tokens merged in. */
    public fun vocabMap(): Map<String, Int> {
        val vocab = model["vocab"]?.jsonObject ?: throw UnsupportedFrontEndException("$FILE_NAME has no 'model.vocab'")
        val out = LinkedHashMap<String, Int>(vocab.size * 2)
        for ((tok, idEl) in vocab) idEl.jsonPrimitive.intOrNull?.let { out[tok] = it }
        for (a in addedTokens) if (a.content !in out) out[a.content] = a.id
        return out
    }

    /** `model.vocab` as `[piece, score]` pairs (Unigram). */
    public fun unigramVocab(): Pair<List<String>, List<Float>> {
        val arr = model["vocab"]?.jsonArray ?: throw UnsupportedFrontEndException("$FILE_NAME has no 'model.vocab' array")
        val pieces = ArrayList<String>(arr.size)
        val scores = ArrayList<Float>(arr.size)
        for (entry in arr) {
            val pair = entry.jsonArray
            pieces.add(pair[0].jsonPrimitive.content)
            val raw = pair[1].jsonPrimitive
            scores.add(raw.doubleOrNull?.toFloat() ?: raw.floatOrNull ?: 0f)
        }
        return pieces to scores
    }

    /** `model.merges` in either form: `"a b"` strings or `["a", "b"]` arrays (#1323). */
    public fun merges(): List<Pair<String, String>> {
        val arr = model["merges"]?.jsonArray ?: throw UnsupportedFrontEndException("$FILE_NAME has no 'model.merges'")
        return arr.map { el ->
            when (el) {
                is JsonArray -> {
                    if (el.size != 2) throw IllegalArgumentException("$FILE_NAME merges: expected a two-element array, got ${el.size} elements: $el")
                    el[0].jsonPrimitive.content to el[1].jsonPrimitive.content
                }
                is JsonPrimitive -> BpeCore.parseMergeLine(el.content)
                else -> throw UnsupportedFrontEndException("$FILE_NAME merges: unsupported entry $el")
            }
        }
    }

    /** `model.unk_id` (Unigram) or the id of `model.unk_token` / a conventional unknown token. */
    public fun unkId(vocabulary: Vocabulary): Int? {
        model["unk_id"]?.jsonPrimitive?.intOrNull?.let { return it }
        (model["unk_token"] as? JsonPrimitive)?.content?.let { vocabulary.tokenToId(it)?.let { id -> return id } }
        for (name in UNK_NAMES) vocabulary.tokenToId(name)?.let { return it }
        return null
    }

    /**
     * The [SpecialTokens] policy: bos/eos/pad/unk from [config] names first, then the
     * post-processor template, then conventional names; `addBos`/`addEos` from the post-processor's
     * single-sequence template (what `encode(add_special_tokens=True)` really adds), falling back to
     * the config flags.
     */
    public fun specialTokens(config: TokenizerConfig?, vocabulary: Vocabulary): SpecialTokens {
        fun idOf(name: String?): Int? = name?.let { n -> addedTokens.firstOrNull { it.content == n }?.id ?: vocabulary.tokenToId(n) }
        val (prefix, suffix) = templateSpecials()
        val bos = idOf(config?.bosToken) ?: prefix.firstOrNull() ?: BOS_NAMES.firstNotNullOfOrNull { idOf(it) }
        val eos = idOf(config?.eosToken) ?: suffix.lastOrNull() ?: EOS_NAMES.firstNotNullOfOrNull { idOf(it) }
        val pad = idOf(config?.padToken) ?: PAD_NAMES.firstNotNullOfOrNull { idOf(it) }
        val unk = idOf(config?.unkToken) ?: unkId(vocabulary)
        val addBos = when {
            postProcessor != null -> bos != null && bos in prefix
            config?.addBosToken != null -> config.addBosToken
            else -> false
        }
        val addEos = when {
            postProcessor != null -> eos != null && eos in suffix
            config?.addEosToken != null -> config.addEosToken
            else -> false
        }
        return SpecialTokens(
            bos = bos, eos = eos, pad = pad, unk = unk,
            addBos = addBos, addEos = addEos,
            declared = addedTokens.filter { it.special }.map { it.id }.toSet(),
        )
    }

    /** Special ids the post-processor puts before / after a single sequence. */
    public fun templateSpecials(): Pair<List<Int>, List<Int>> {
        val pp = postProcessor ?: return emptyList<Int>() to emptyList()
        val prefix = ArrayList<Int>()
        val suffix = ArrayList<Int>()
        fun visit(obj: JsonObject) {
            when (obj["type"]?.jsonPrimitive?.content) {
                "Sequence" -> obj["processors"]?.jsonArray.orEmpty().forEach { (it as? JsonObject)?.let(::visit) }
                "TemplateProcessing" -> {
                    val single = obj["single"]?.jsonArray ?: return
                    val specialIds = obj["special_tokens"]?.jsonObject.orEmpty().mapValues { (_, v) -> (v as? JsonObject)?.get("ids")?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull }.orEmpty() }
                    var seenSequence = false
                    for (item in single) {
                        val o = item as? JsonObject ?: continue
                        when {
                            o["Sequence"] != null -> seenSequence = true
                            o["SpecialToken"] != null -> {
                                val name = o["SpecialToken"]!!.jsonObject["id"]?.jsonPrimitive?.content ?: continue
                                val ids = specialIds[name] ?: emptyList()
                                if (seenSequence) suffix.addAll(ids) else prefix.addAll(ids)
                            }
                        }
                    }
                }
                "BertProcessing", "RobertaProcessing" -> {
                    (obj["cls"] as? JsonArray)?.getOrNull(1)?.jsonPrimitive?.intOrNull?.let { prefix.add(it) }
                    (obj["sep"] as? JsonArray)?.getOrNull(1)?.jsonPrimitive?.intOrNull?.let { suffix.add(it) }
                }
            }
        }
        visit(pp)
        return prefix to suffix
    }

    public companion object {
        public const val FILE_NAME: String = "tokenizer.json"

        internal val JSON: Json = Json { ignoreUnknownKeys = true; isLenient = true }

        public fun parse(json: String): TokenizerJson = TokenizerJson(JSON.parseToJsonElement(json).jsonObject)

        private val BOS_NAMES = listOf("<bos>", "<s>", "<|begin_of_text|>", "[CLS]", "<|startoftext|>")
        private val EOS_NAMES = listOf("<eos>", "</s>", "<|end_of_text|>", "<|endoftext|>", "[SEP]", "<|im_end|>")
        private val PAD_NAMES = listOf("<pad>", "[PAD]", "<|pad|>", "<|endoftext|>")
        private val UNK_NAMES = listOf("<unk>", "[UNK]", "<|unk|>")
    }
}

/**
 * The sibling `tokenizer_config.json` (FE-R-03): special-token names, the `add_bos_token` /
 * `add_eos_token` flags, lower-casing and the chat template.
 */
public class TokenizerConfig(public val root: JsonObject) {
    public val bosToken: String? = tokenName(root["bos_token"])
    public val eosToken: String? = tokenName(root["eos_token"])
    public val padToken: String? = tokenName(root["pad_token"])
    public val unkToken: String? = tokenName(root["unk_token"])
    public val addBosToken: Boolean? = root["add_bos_token"]?.jsonPrimitive?.booleanOrNull
    public val addEosToken: Boolean? = root["add_eos_token"]?.jsonPrimitive?.booleanOrNull
    public val doLowerCase: Boolean? = root["do_lower_case"]?.jsonPrimitive?.booleanOrNull
    public val chatTemplate: String? = (root["chat_template"] as? JsonPrimitive)?.content
    public val tokenizerClass: String? = (root["tokenizer_class"] as? JsonPrimitive)?.content
    public val modelMaxLength: Long? = (root["model_max_length"] as? JsonPrimitive)?.content?.toDoubleOrNull()?.toLong()

    private fun tokenName(el: JsonElement?): String? = when (el) {
        null, is JsonNull -> null
        is JsonPrimitive -> el.content
        is JsonObject -> (el["content"] as? JsonPrimitive)?.content
        else -> null
    }

    public companion object {
        public const val FILE_NAME: String = "tokenizer_config.json"

        public fun parse(json: String): TokenizerConfig = TokenizerConfig(TokenizerJson.JSON.parseToJsonElement(json).jsonObject)
    }
}
