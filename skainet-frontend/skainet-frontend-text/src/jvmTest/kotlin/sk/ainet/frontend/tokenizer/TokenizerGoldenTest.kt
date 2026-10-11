package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.frontend.text.Tokenizer
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.fail

/**
 * Golden corpus recorded with the Hugging Face `tokenizers` library (0.23.3) — see
 * `src/jvmTest/resources/goldens/<fixture>.json`. Every fixture is a public tokenizer file downloaded by
 * `downloadTokenizerFixtures`; a test skips when its file is absent so offline builds stay green.
 *
 * Checked per case: `encode(text)` ids, `encode(text, addSpecialTokens = true)` ids,
 * `decode(ids)` text, `decode(idsWithSpecials, skipSpecialTokens = true)` text, and `vocabSize`.
 */
class TokenizerGoldenTest {

    private val fixturesDir = File(System.getProperty("skainet.test.fixturesDir") ?: "build/test-fixtures")
    private val json = Json { ignoreUnknownKeys = true }

    private fun golden(name: String) = json.parseToJsonElement(
        this::class.java.getResource("/goldens/$name.json")?.readText() ?: fail("golden $name.json missing"),
    ).jsonObject

    private fun load(fixture: String): Tokenizer? {
        val tokJson = File(fixturesDir, "$fixture-tokenizer.json")
        if (!tokJson.isFile) { println("[skip] TokenizerGoldenTest $fixture: ${tokJson.name} not downloaded (run downloadTokenizerFixtures)"); return null }
        val config = File(fixturesDir, "$fixture-tokenizer_config.json").takeIf { it.isFile }?.readText()
        return TextFrontEnds.fromTokenizerJson(tokJson.readText(), config)
    }

    private fun check(fixture: String, skipDecodeFor: Set<Int> = emptySet()) {
        val tok = load(fixture) ?: return
        val g = golden(fixture)
        val failures = ArrayList<String>()
        assertEquals(g["vocab_size"]!!.jsonPrimitive.int, tok.vocabSize, "$fixture vocabSize")
        for ((i, c) in g["cases"]!!.jsonArray.withIndex()) {
            val o = c.jsonObject
            val text = o["text"]!!.jsonPrimitive.content
            val ids = o["ids"]!!.jsonArray.map { it.jsonPrimitive.int }
            val idsSpecial = o["ids_with_specials"]!!.jsonArray.map { it.jsonPrimitive.int }
            val got = tok.encode(text).toList()
            if (got != ids) failures.add("case $i encode(${text.esc()}): expected $ids got $got")
            val gotSpecial = tok.encode(text, addSpecialTokens = true).toList()
            if (gotSpecial != idsSpecial) failures.add("case $i encode(${text.esc()}, specials): expected $idsSpecial got $gotSpecial")
            if (i !in skipDecodeFor) {
                val dec = tok.decode(ids.toIntArray())
                val expDec = o["decode"]!!.jsonPrimitive.content
                if (dec != expDec) failures.add("case $i decode: expected ${expDec.esc()} got ${dec.esc()}")
                val decSkip = tok.decode(idsSpecial.toIntArray(), skipSpecialTokens = true)
                val expSkip = o["decode_skip_special"]!!.jsonPrimitive.content
                if (decSkip != expSkip) failures.add("case $i decode(skip): expected ${expSkip.esc()} got ${decSkip.esc()}")
            }
        }
        if (failures.isNotEmpty()) fail("$fixture: ${failures.size} mismatches\n" + failures.joinToString("\n"))
    }

    private fun String.esc() = "\"" + replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r") + "\""

    @Test fun `gpt2 byte-level BPE matches the reference`() = check("gpt2")
    @Test fun `qwen2_5 byte-level BPE with NFC and Split regex matches the reference`() = check("qwen25")
    @Test fun `laya ModernBERT byte-level BPE with added tokens matches the reference`() = check("laya-modernbert")
    @Test fun `laya mmBERT Metaspace BPE with byte fallback matches the reference`() = check("laya-mmbert")
    @Test fun `tinyllama Metaspace BPE via normalizer matches the reference`() = check("tinyllama")
    @Test fun `t5-small Unigram matches the reference`() = check("t5-small")
    @Test fun `gemma4 E2B Metaspace BPE with byte fallback matches the reference`() = check("gemma4-e2b")
    @Test fun `bert-base-uncased WordPiece matches the reference`() {
        check("bert-base-uncased")
        val tok = load("bert-base-uncased") as? sk.ainet.frontend.tokenizer.WordPieceTokenizer ?: return
        val pair = golden("bert-base-uncased")["pair"]!!.jsonObject
        val enc = tok.encodeWithMetadata(pair["a"]!!.jsonPrimitive.content, pair["b"]!!.jsonPrimitive.content)
        assertEquals(pair["ids"]!!.jsonArray.map { it.jsonPrimitive.int }, enc.ids.toList(), "pair ids")
        assertEquals(pair["type_ids"]!!.jsonArray.map { it.jsonPrimitive.int }, enc.tokenTypeIds!!.toList(), "pair type ids")
    }

    @Test
    fun `bert vocab_txt gives the same ids as tokenizer_json`() {
        val vocab = File(fixturesDir, "bert-base-uncased-vocab.txt")
        val tokJson = load("bert-base-uncased") ?: return
        if (!vocab.isFile) return
        val fromTxt = TextFrontEnds.fromVocabTxt(vocab.readText())
        for (text in listOf("Hello world", "unaffable café 日本", "   leading")) {
            assertEquals(tokJson.encode(text).toList(), fromTxt.encode(text).toList(), text)
        }
        assertEquals(tokJson.vocabSize, fromTxt.vocabSize)
    }
}
