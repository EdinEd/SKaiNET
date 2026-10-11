package sk.ainet.frontend.tokenizer

import sk.ainet.frontend.text.SpecialTokens

/**
 * GGUF UINT32 fields come back from the GGUF reader as `kotlin.UInt`, which is a value class —
 * not a subclass of `kotlin.Number`. A plain `as? Number` cast silently returns `null` for them,
 * which is how `tokenizer.ggml.bos_token_id` etc. were getting lost. This helper accepts every
 * numeric and unsigned numeric type GGUF can produce.
 */
internal fun Any?.toIntFlexible(): Int? = when (this) {
    is Number -> toInt()
    is UByte -> toInt()
    is UShort -> toInt()
    is UInt -> toInt()
    is ULong -> toInt()
    else -> null
}

internal fun Any?.toBooleanFlexible(default: Boolean): Boolean = when (this) {
    is Boolean -> this
    is Number -> toInt() != 0
    is UInt -> toInt() != 0
    is UByte -> toInt() != 0
    null -> default
    else -> default
}

/** The `tokenizer.ggml.*` keys and token-type codes (ggml convention) shared by the GGUF readers. */
internal object GgufTokenizerFields {
    const val MODEL = "tokenizer.ggml.model"
    const val TOKENS = "tokenizer.ggml.tokens"
    const val SCORES = "tokenizer.ggml.scores"
    const val MERGES = "tokenizer.ggml.merges"
    const val TOKEN_TYPE = "tokenizer.ggml.token_type"
    const val BOS = "tokenizer.ggml.bos_token_id"
    const val EOS = "tokenizer.ggml.eos_token_id"
    const val UNK = "tokenizer.ggml.unknown_token_id"
    const val PAD = "tokenizer.ggml.padding_token_id"
    const val ADD_BOS = "tokenizer.ggml.add_bos_token"
    const val ADD_EOS = "tokenizer.ggml.add_eos_token"
    const val ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix"
    const val CHAT_TEMPLATE = "tokenizer.chat_template"

    const val TOKEN_TYPE_NORMAL = 1
    const val TOKEN_TYPE_UNKNOWN = 2
    const val TOKEN_TYPE_CONTROL = 3
    const val TOKEN_TYPE_USER_DEFINED = 4
    const val TOKEN_TYPE_BYTE = 6

    fun tokens(fields: Map<String, Any?>): List<String> =
        (fields[TOKENS] as? List<*>)?.filterIsInstance<String>()
            ?: error("$TOKENS missing or malformed")

    fun tokenTypes(fields: Map<String, Any?>): List<Int>? =
        (fields[TOKEN_TYPE] as? List<*>)?.mapNotNull { it.toIntFlexible() }

    /** CONTROL and USER_DEFINED tokens: the atomic chat-template markers, keyed by surface form. */
    fun specialTokenMap(fields: Map<String, Any?>): Map<String, Int> {
        val tokens = (fields[TOKENS] as? List<*>)?.filterIsInstance<String>().orEmpty()
        val types = tokenTypes(fields).orEmpty()
        if (tokens.isEmpty() || types.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, Int>()
        val limit = minOf(tokens.size, types.size)
        for (i in 0 until limit) {
            val t = types[i]
            if ((t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED) && tokens[i].isNotEmpty()) if (tokens[i] !in out) out[tokens[i]] = i
        }
        return out
    }

    /** `SpecialTokens` from the GGUF fields; [declared] are the CONTROL / USER_DEFINED ids. */
    fun specialTokens(fields: Map<String, Any?>, declared: Set<Int>): SpecialTokens = SpecialTokens(
        bos = fields[BOS]?.toIntFlexible(),
        eos = fields[EOS]?.toIntFlexible(),
        pad = fields[PAD]?.toIntFlexible(),
        unk = fields[UNK]?.toIntFlexible(),
        addBos = fields[ADD_BOS].toBooleanFlexible(default = false),
        addEos = fields[ADD_EOS].toBooleanFlexible(default = false),
        declared = declared,
    )
}
