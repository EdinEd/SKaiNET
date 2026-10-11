package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * Mistral Tekken tokenizer.
 *
 * Tekken is a tiktoken-based BPE tokenizer used by Mistral models (Mistral, Mixtral, Codestral,
 * Voxtral, …). Unlike `tokenizer.json`, `tekken.json` uses base64-encoded byte sequences for vocab
 * tokens, implicit merge ordering from vocab rank (lower rank = higher priority), a separate
 * special-token list with reserved id space `[0, numSpecialTokens)`, and a tiktoken-style
 * pre-tokenization regex.
 *
 * Token id layout: ids `[0, numSpecialTokens)` are special tokens (`<unk>`, `<s>`, `</s>`,
 * `[INST]`, …); ids from [numSpecialTokens] on are vocab tokens (rank + offset).
 *
 * [vocabSize] is the total id space (specials + vocab), as for every other [Tokenizer]; the
 * vocab-only count is [vocabTokenCount].
 */
public class TekkenTokenizer(
    private val vocabTokenBytes: List<ByteArray>,
    private val vocabTokenStrings: List<String?>,
    private val specialTokenIds: Map<String, Int>,
    private val specialTokensById: Map<Int, String>,
    public val numSpecialTokens: Int = 1000,
    private val pattern: Regex,
    source: String? = null,
) : Tokenizer {
    /** BPE rank lookup: byte sequence → rank (merge priority). */
    private val bytesToRank: HashMap<ByteArrayKey, Int> = HashMap(vocabTokenBytes.size * 2)
    private val specialsByLengthDesc: List<String> = specialTokenIds.keys.filter { it.isNotEmpty() }.sortedByDescending { it.length }

    init {
        for (i in vocabTokenBytes.indices) bytesToRank[ByteArrayKey(vocabTokenBytes[i])] = i
    }

    /** Number of vocab tokens (excluding special tokens). */
    public val vocabTokenCount: Int get() = vocabTokenBytes.size

    /** Total token count (vocab + special tokens). */
    public val totalTokens: Int get() = vocabTokenBytes.size + numSpecialTokens

    override val vocabulary: Vocabulary = object : Vocabulary {
        override val size: Int get() = totalTokens
        override fun tokenToId(piece: String): Int? =
            specialTokenIds[piece] ?: bytesToRank[ByteArrayKey(piece.encodeToByteArray())]?.let { it + numSpecialTokens }
        override fun idToToken(id: Int): String? = when {
            id < 0 -> null
            id < numSpecialTokens -> specialTokensById[id] ?: "<SPECIAL_$id>"
            id - numSpecialTokens < vocabTokenBytes.size -> vocabTokenStrings[id - numSpecialTokens] ?: vocabTokenBytes[id - numSpecialTokens].decodeToString()
            else -> null
        }
    }

    override val specialTokens: SpecialTokens = SpecialTokens(
        bos = specialTokenIds["<s>"],
        eos = specialTokenIds["</s>"],
        unk = specialTokenIds["<unk>"],
        pad = specialTokenIds["<pad>"],
        addBos = true,
        declared = specialTokensById.keys,
    )

    override val spec: FrontEndSpec = FrontEndSpec.Text(ALGORITHM, source, JsonObject(mapOf("num_special_tokens" to JsonPrimitive(numSpecialTokens))))

    override fun encode(text: String): IntArray {
        val tokens = ArrayList<Int>()
        var i = 0
        var segmentStart = 0
        while (i < text.length) {
            val special = specialsByLengthDesc.firstOrNull { text.regionMatches(i, it, 0, it.length) }
            if (special == null) { i++; continue }
            if (i > segmentStart) encodeSegment(text.substring(segmentStart, i), tokens)
            tokens.add(specialTokenIds.getValue(special))
            i += special.length
            segmentStart = i
        }
        if (segmentStart < text.length) encodeSegment(text.substring(segmentStart), tokens)
        return tokens.toIntArray()
    }

    private fun encodeSegment(segment: String, out: MutableList<Int>) {
        for (match in pattern.findAll(segment)) {
            val chunk = match.value
            if (chunk.isEmpty()) continue
            for (rank in bpeMerge(chunk.encodeToByteArray())) out.add(rank + numSpecialTokens)
        }
    }

    override fun decode(ids: IntArray): String {
        val bytes = mutableListOf<Byte>()
        val result = StringBuilder()
        for (id in ids) {
            if (id < numSpecialTokens) {
                if (bytes.isNotEmpty()) { result.append(bytes.toByteArray().decodeToString()); bytes.clear() }
                result.append(specialTokensById[id] ?: "<SPECIAL_$id>")
            } else {
                val rank = id - numSpecialTokens
                if (rank in vocabTokenBytes.indices) bytes.addAll(vocabTokenBytes[rank].toList())
            }
        }
        if (bytes.isNotEmpty()) result.append(bytes.toByteArray().decodeToString())
        return result.toString()
    }

    /** Decode a single token id to text (0.57 signature). */
    public fun decode(token: Int): String = decodeToken(token)

    override fun decodeToken(id: Int): String {
        if (id < numSpecialTokens) return specialTokensById[id] ?: "<SPECIAL_$id>"
        val rank = id - numSpecialTokens
        if (rank in vocabTokenBytes.indices) return vocabTokenBytes[rank].decodeToString()
        return "<UNK_$id>"
    }

    /**
     * tiktoken BPE over bytes: repeatedly merge the adjacent pair with the lowest rank.
     * @return vocab ranks (NOT token ids — the caller adds [numSpecialTokens])
     */
    private fun bpeMerge(bytes: ByteArray): List<Int> {
        if (bytes.isEmpty()) return emptyList()
        val pieces = ArrayList<ByteArray>(bytes.size)
        for (b in bytes) pieces.add(byteArrayOf(b))
        while (pieces.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until pieces.size - 1) {
                val rank = bytesToRank[ByteArrayKey(concat(pieces[i], pieces[i + 1]))]
                if (rank != null && rank < bestRank) { bestRank = rank; bestIdx = i }
            }
            if (bestIdx == -1) break
            pieces[bestIdx] = concat(pieces[bestIdx], pieces[bestIdx + 1])
            pieces.removeAt(bestIdx + 1)
        }
        return pieces.map { piece -> bytesToRank[ByteArrayKey(piece)] ?: error("BPE produced unknown byte sequence: ${piece.toList()}") }
    }

    private fun concat(a: ByteArray, b: ByteArray): ByteArray {
        val result = ByteArray(a.size + b.size)
        a.copyInto(result)
        b.copyInto(result, a.size)
        return result
    }

    public companion object {
        public const val ALGORITHM: String = "tekken"

        /** Parse a `tekken.json` string. */
        @OptIn(ExperimentalEncodingApi::class)
        public fun fromJson(json: String): TekkenTokenizer = TekkenJsonParser(json).parse()
    }
}

/** Wrapper for ByteArray that implements equals/hashCode for use as a HashMap key. */
internal class ByteArrayKey(val bytes: ByteArray) {
    override fun equals(other: Any?): Boolean = other is ByteArrayKey && bytes.contentEquals(other.bytes)
    override fun hashCode(): Int = bytes.contentHashCode()
}

/** Lightweight parser for the tekken.json format (the structure is simple and the file is large). */
@OptIn(ExperimentalEncodingApi::class)
internal class TekkenJsonParser(private val json: String) {

    fun parse(): TekkenTokenizer {
        val numSpecialTokens = extractInt("default_num_special_tokens") ?: 1000
        val patternStr = extractString("pattern")
            ?: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+(?!\\S)|\\s+"

        val vocabTokenBytes = mutableListOf<ByteArray>()
        val vocabTokenStrings = mutableListOf<String?>()
        parseVocabEntries(vocabTokenBytes, vocabTokenStrings)

        val specialTokens = mutableMapOf<String, Int>()
        val specialTokensById = mutableMapOf<Int, String>()
        parseSpecialTokens(specialTokens, specialTokensById)

        val pattern = try {
            Regex(patternStr)
        } catch (e: Exception) {
            Regex("\\S+|\\s+")
        }

        return TekkenTokenizer(
            vocabTokenBytes = vocabTokenBytes,
            vocabTokenStrings = vocabTokenStrings,
            specialTokenIds = specialTokens,
            specialTokensById = specialTokensById,
            numSpecialTokens = numSpecialTokens,
            pattern = pattern,
            source = "tekken.json",
        )
    }

    private fun parseVocabEntries(tokenBytes: MutableList<ByteArray>, tokenStrings: MutableList<String?>) {
        val vocabStart = json.indexOf("\"vocab\"")
        if (vocabStart < 0) return
        val arrayStart = json.indexOf('[', vocabStart)
        if (arrayStart < 0) return
        var pos = arrayStart + 1
        while (pos < json.length) {
            pos = skipWhitespace(pos)
            if (pos >= json.length || json[pos] == ']') break
            if (json[pos] == '{') {
                val objEnd = findMatchingBrace(pos)
                val obj = json.substring(pos, objEnd + 1)
                val tokenBytesB64 = extractStringFromObj(obj, "token_bytes")
                val tokenStr = extractStringFromObj(obj, "token_str")
                if (tokenBytesB64 != null) {
                    tokenBytes.add(Base64.decode(tokenBytesB64))
                    tokenStrings.add(tokenStr)
                }
                pos = objEnd + 1
            } else {
                pos++
            }
            pos = skipWhitespace(pos)
            if (pos < json.length && json[pos] == ',') pos++
        }
    }

    private fun parseSpecialTokens(specialTokens: MutableMap<String, Int>, specialTokensById: MutableMap<Int, String>) {
        val stStart = json.indexOf("\"special_tokens\"")
        if (stStart < 0) return
        val arrayStart = json.indexOf('[', stStart)
        if (arrayStart < 0) return
        var pos = arrayStart + 1
        while (pos < json.length) {
            pos = skipWhitespace(pos)
            if (pos >= json.length || json[pos] == ']') break
            if (json[pos] == '{') {
                val objEnd = findMatchingBrace(pos)
                val obj = json.substring(pos, objEnd + 1)
                val rank = extractIntFromObj(obj, "rank")
                val tokenStr = extractStringFromObj(obj, "token_str")
                if (rank != null && tokenStr != null) {
                    specialTokens[tokenStr] = rank
                    specialTokensById[rank] = tokenStr
                }
                pos = objEnd + 1
            } else {
                pos++
            }
            pos = skipWhitespace(pos)
            if (pos < json.length && json[pos] == ',') pos++
        }
    }

    private fun extractInt(key: String): Int? {
        val keyStr = "\"$key\""
        val idx = json.indexOf(keyStr)
        if (idx < 0) return null
        var pos = skipWhitespace(idx + keyStr.length)
        if (pos < json.length && json[pos] == ':') pos++
        pos = skipWhitespace(pos)
        val start = pos
        while (pos < json.length && (json[pos].isDigit() || json[pos] == '-')) pos++
        return json.substring(start, pos).toIntOrNull()
    }

    private fun extractString(key: String): String? {
        val keyStr = "\"$key\""
        val idx = json.indexOf(keyStr)
        if (idx < 0) return null
        var pos = skipWhitespace(idx + keyStr.length)
        if (pos < json.length && json[pos] == ':') pos++
        pos = skipWhitespace(pos)
        if (pos >= json.length || json[pos] != '"') return null
        return readJsonStringFrom(json, pos)
    }

    private fun extractStringFromObj(obj: String, key: String): String? {
        val keyStr = "\"$key\""
        val idx = obj.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        while (pos < obj.length && (obj[pos] == ' ' || obj[pos] == ':')) pos++
        if (pos >= obj.length) return null
        if (obj[pos] == 'n' && obj.startsWith("null", pos)) return null
        if (obj[pos] != '"') return null
        return readJsonStringFrom(obj, pos)
    }

    private fun extractIntFromObj(obj: String, key: String): Int? {
        val keyStr = "\"$key\""
        val idx = obj.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        while (pos < obj.length && (obj[pos] == ' ' || obj[pos] == ':')) pos++
        val start = pos
        while (pos < obj.length && (obj[pos].isDigit() || obj[pos] == '-')) pos++
        return obj.substring(start, pos).toIntOrNull()
    }

    private fun readJsonStringFrom(s: String, startPos: Int): String {
        val sb = StringBuilder()
        var pos = startPos + 1
        while (pos < s.length) {
            val c = s[pos]
            when {
                c == '"' -> return sb.toString()
                c == '\\' && pos + 1 < s.length -> {
                    pos++
                    when (s[pos]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        '/' -> sb.append('/')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'u' -> {
                            if (pos + 4 < s.length) {
                                val hex = s.substring(pos + 1, pos + 5)
                                sb.append((hex.toIntOrNull(16) ?: 0).toChar())
                                pos += 4
                            }
                        }
                    }
                }
                else -> sb.append(c)
            }
            pos++
        }
        return sb.toString()
    }

    private fun skipWhitespace(pos: Int): Int {
        var p = pos
        while (p < json.length && json[p].isWhitespace()) p++
        return p
    }

    private fun findMatchingBrace(start: Int): Int {
        var depth = 0
        var inString = false
        var pos = start
        while (pos < json.length) {
            val c = json[pos]
            when {
                inString -> if (c == '"') inString = false else if (c == '\\') pos++
                c == '"' -> inString = true
                c == '{' -> depth++
                c == '}' -> { depth--; if (depth == 0) return pos }
            }
            pos++
        }
        return json.length - 1
    }
}
