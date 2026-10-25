package sk.ainet.io.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.io.ByteArrayRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Token-id parity of a GGUF tokenizer against ids recorded earlier: guards changes to the GGUF reader and the tokenizer
 * build, which must not move a single id. Needs three files, none of them in the repo:
 *
 * - `SKAINET_GGUF_FILE`: a GGUF with tokenizer metadata;
 * - `SKAINET_TOKENIZER_CORPUS`: JSON lines, one JSON string per line, the texts to encode;
 * - `SKAINET_TOKENIZER_IDS`: the recorded ids. When the file does not exist the test writes it (record a baseline with
 *   the build to compare against), otherwise it compares line by line.
 *
 * Besides `encode(text)` per text it records the vocabulary round trip over every id and the special-token policy.
 * Without the first two variables the test prints a skip notice and passes.
 */
class GgufTokenizerIdParityTest {

    @Test
    fun idsMatchRecordedBaseline() {
        val gguf = System.getenv("SKAINET_GGUF_FILE")?.let(::File)
        val corpus = System.getenv("SKAINET_TOKENIZER_CORPUS")?.let(::File)
        val idsPath = System.getenv("SKAINET_TOKENIZER_IDS")
        if (gguf?.isFile != true || corpus?.isFile != true || idsPath.isNullOrBlank()) {
            println("[skip] GgufTokenizerIdParityTest: set SKAINET_GGUF_FILE, SKAINET_TOKENIZER_CORPUS and SKAINET_TOKENIZER_IDS")
            return
        }
        val fields = StreamingGGUFReader.open(ByteArrayRandomAccessSource(gguf.readBytes())).use { it.fields }
        val tokenizer = TextFrontEnds.fromGguf(fields)
        val vocab = tokenizer.vocabulary

        val lines = ArrayList<String>()
        lines += "specialTokens ${tokenizer.specialTokens}"
        lines += "vocab.size ${vocab.size}"
        var roundTrip = 0L
        for (id in 0 until vocab.size) {
            val piece = vocab.idToToken(id)
            roundTrip = roundTrip * 31 + (piece?.let { vocab.tokenToId(it) } ?: -2) + 7 * (piece?.hashCode() ?: 0)
        }
        lines += "vocab.roundTrip $roundTrip"
        corpus.forEachLine { line ->
            if (line.isNotBlank()) lines += tokenizer.encode(Json.parseToJsonElement(line).jsonPrimitive.content).joinToString(" ")
        }

        val ids = File(idsPath)
        if (!ids.exists()) {
            ids.writeText(lines.joinToString("\n", postfix = "\n"))
            println("[record] GgufTokenizerIdParityTest: wrote ${lines.size} lines to $ids")
            return
        }
        val expected = ids.readLines()
        assertEquals(expected.size, lines.size, "line count")
        for (i in expected.indices) assertEquals(expected[i], lines[i], "line ${i + 1}")
        println("[parity] GgufTokenizerIdParityTest: ${lines.size} lines identical (${lines.size - 3} texts)")
    }
}
