package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.UnsupportedFrontEndException

/**
 * The `tokenizer.json#normalizer` step, applied to every non-added-token segment before
 * pre-tokenization (SKaiNET #1325). Built by [fromJson]; unknown normalizer types fail loudly
 * instead of being skipped — that silent skip was the bug.
 */
public sealed interface Normalizer {
    public fun apply(text: String): String

    /** Short description for specs and messages. */
    public val name: String

    public object None : Normalizer {
        override fun apply(text: String): String = text
        override val name: String get() = "none"
    }

    public class Sequence(public val steps: List<Normalizer>) : Normalizer {
        override fun apply(text: String): String = steps.fold(text) { acc, n -> n.apply(acc) }
        override val name: String get() = steps.joinToString("+", prefix = "[", postfix = "]") { it.name }
    }

    /** Literal-string replace (`pattern.String`) or regex replace (`pattern.Regex`). */
    public class Replace(private val pattern: String, private val content: String, private val isRegex: Boolean) : Normalizer {
        private val regex: Regex? = if (isRegex) Regex(pattern) else null
        override fun apply(text: String): String =
            if (regex != null) regex.replace(text, Regex.escapeReplacement(content)) else text.replace(pattern, content)
        override val name: String get() = "replace"
    }

    public class Prepend(public val prepend: String) : Normalizer {
        override fun apply(text: String): String = if (text.isEmpty()) text else prepend + text
        override val name: String get() = "prepend"
    }

    public class Unicode(public val form: NormalizationForm) : Normalizer {
        override fun apply(text: String): String = UnicodeNormalization.normalize(text, form)
        override val name: String get() = form.name.lowercase()
    }

    public object Lowercase : Normalizer {
        override fun apply(text: String): String = text.lowercase()
        override val name: String get() = "lowercase"
    }

    public object Strip : Normalizer {
        override fun apply(text: String): String = text.trim()
        override val name: String get() = "strip"
    }

    /** NFD, drop combining marks (`Mn`), as the reference `StripAccents`. */
    public object StripAccents : Normalizer {
        override fun apply(text: String): String = stripAccents(text)
        override val name: String get() = "strip-accents"
    }

    /**
     * The reference `BertNormalizer`: optional control-character cleanup and whitespace folding,
     * spaces around CJK ideographs, lower-casing and accent stripping (defaults to stripping when
     * lower-casing, as the reference does for `strip_accents = null`).
     */
    public class Bert(
        public val cleanText: Boolean = true,
        public val handleChineseChars: Boolean = true,
        public val stripAccents: Boolean?,
        public val lowercase: Boolean = true,
    ) : Normalizer {
        override fun apply(text: String): String {
            var s = text
            if (cleanText) s = cleanText(s)
            if (handleChineseChars) s = spaceChineseChars(s)
            val strip = stripAccents ?: lowercase
            if (lowercase) s = s.lowercase()
            if (strip) s = stripAccents(s)
            return s
        }
        override val name: String get() = "bert"
    }

    /**
     * The SentencePiece `Precompiled` charsmap (NMT NFKC). The exact table is binary and model
     * specific; this approximates it with NFKC, which is what the shipped `nmt_nfkc` maps encode
     * for the overwhelming majority of inputs. Reported in [name] so a golden can tell.
     */
    public object Precompiled : Normalizer {
        override fun apply(text: String): String = UnicodeNormalization.normalize(text, NormalizationForm.NFKC)
        override val name: String get() = "precompiled~nfkc"
    }

    public companion object {
        /** Build from the `normalizer` element of a tokenizer.json root (`null` → [None]). */
        public fun fromJson(element: JsonElement?): Normalizer {
            if (element == null) return None
            val obj = element as? JsonObject ?: return None
            val type = obj["type"]?.jsonPrimitive?.content
                ?: throw UnsupportedFrontEndException("tokenizer.json normalizer without a type: $obj")
            return when (type) {
                "Sequence" -> Sequence(obj["normalizers"]?.jsonArray.orEmpty().map { fromJson(it) })
                "Replace" -> {
                    val pattern = obj["pattern"]?.jsonObject
                        ?: throw UnsupportedFrontEndException("tokenizer.json Replace normalizer without a pattern")
                    val content = obj["content"]?.jsonPrimitive?.content ?: ""
                    when {
                        pattern["String"] != null -> Replace(pattern["String"]!!.jsonPrimitive.content, content, isRegex = false)
                        pattern["Regex"] != null -> Replace(pattern["Regex"]!!.jsonPrimitive.content, content, isRegex = true)
                        else -> throw UnsupportedFrontEndException("tokenizer.json Replace normalizer: unknown pattern kind $pattern")
                    }
                }
                "Prepend" -> Prepend(obj["prepend"]?.jsonPrimitive?.content ?: "")
                "NFC" -> Unicode(NormalizationForm.NFC)
                "NFD" -> Unicode(NormalizationForm.NFD)
                "NFKC" -> Unicode(NormalizationForm.NFKC)
                "NFKD" -> Unicode(NormalizationForm.NFKD)
                "Lowercase" -> Lowercase
                "Strip" -> Strip
                "StripAccents" -> StripAccents
                "BertNormalizer" -> Bert(
                    cleanText = obj["clean_text"]?.jsonPrimitive?.booleanOrNull ?: true,
                    handleChineseChars = obj["handle_chinese_chars"]?.jsonPrimitive?.booleanOrNull ?: true,
                    stripAccents = obj["strip_accents"]?.jsonPrimitive?.booleanOrNull,
                    lowercase = obj["lowercase"]?.jsonPrimitive?.booleanOrNull ?: true,
                )
                "Precompiled" -> Precompiled
                else -> throw UnsupportedFrontEndException("tokenizer.json normalizer type '$type' is not supported")
            }
        }

        internal fun stripAccents(text: String): String {
            val decomposed = UnicodeNormalization.normalize(text, NormalizationForm.NFD)
            val sb = StringBuilder(decomposed.length)
            for (ch in decomposed) if (ch.category != CharCategory.NON_SPACING_MARK) sb.append(ch)
            return sb.toString()
        }

        internal fun cleanText(text: String): String {
            val sb = StringBuilder(text.length)
            for (ch in text) {
                val code = ch.code
                if (code == 0 || code == 0xFFFD) continue
                if (ch.category == CharCategory.CONTROL && ch != '\t' && ch != '\n' && ch != '\r') continue
                if (ch.isWhitespace()) sb.append(' ') else sb.append(ch)
            }
            return sb.toString()
        }

        internal fun spaceChineseChars(text: String): String {
            val sb = StringBuilder(text.length + 8)
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAtCompat(i)
                val width = if (cp > 0xFFFF) 2 else 1
                if (isChineseChar(cp)) {
                    sb.append(' ')
                    sb.append(text, i, i + width)
                    sb.append(' ')
                } else {
                    sb.append(text, i, i + width)
                }
                i += width
            }
            return sb.toString()
        }

        private fun isChineseChar(cp: Int): Boolean =
            (cp in 0x4E00..0x9FFF) || (cp in 0x3400..0x4DBF) || (cp in 0x20000..0x2A6DF) || (cp in 0x2A700..0x2B73F) ||
                (cp in 0x2B740..0x2B81F) || (cp in 0x2B820..0x2CEAF) || (cp in 0xF900..0xFAFF) || (cp in 0x2F800..0x2FA1F)
    }
}

/** Code point at [index], joining a surrogate pair when present. */
internal fun String.codePointAtCompat(index: Int): Int {
    val c = this[index]
    if (c.isHighSurrogate() && index + 1 < length) {
        val low = this[index + 1]
        if (low.isLowSurrogate()) return ((c.code - 0xD800) shl 10) + (low.code - 0xDC00) + 0x10000
    }
    return c.code
}
