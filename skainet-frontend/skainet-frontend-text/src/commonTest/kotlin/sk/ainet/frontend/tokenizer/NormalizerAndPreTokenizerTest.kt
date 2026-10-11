package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.Json
import sk.ainet.frontend.UnsupportedFrontEndException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NormalizerAndPreTokenizerTest {

    private fun el(json: String) = Json.parseToJsonElement(json)

    @Test
    fun `replace prepend and sequence compose in order`() {
        val n = Normalizer.fromJson(el("""{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]}"""))
        assertEquals("▁Hello▁world", n.apply("Hello world"))
        assertEquals("", n.apply(""))
    }

    @Test
    fun `nfc composes a decomposed e acute where the platform has tables`() {
        val n = Normalizer.fromJson(el("""{"type":"NFC"}"""))
        val out = n.apply("e\u0301")
        if (UnicodeNormalization.isExact) assertEquals("é", out) else assertEquals("e\u0301", out)
    }

    @Test
    fun `unknown normalizer types fail loudly with the type name`() {
        val ex = assertFailsWith<UnsupportedFrontEndException> { Normalizer.fromJson(el("""{"type":"Nonsense"}""")) }
        assertTrue("Nonsense" in ex.message!!, ex.message)
    }

    @Test
    fun `bert normalizer lowercases strips accents and spaces CJK`() {
        val n = Normalizer.Bert(stripAccents = null, lowercase = true)
        val out = n.apply("Café 日本")
        if (UnicodeNormalization.isExact) assertEquals("cafe  日  本 ", out) else assertEquals("café  日  本 ", out)
    }

    @Test
    fun `metaspace pre-tokenizer prepends and splits at word boundaries`() {
        val p = PreTokenizer.fromJson(el("""{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"""))
        assertEquals(listOf("▁Hello", "▁world"), p.split("Hello world"))
        assertEquals(listOf("▁", "▁x"), p.split("  x"))   // the prefix is only prepended when the text does not already start with it
        val never = PreTokenizer.Metaspace(prependScheme = "never")
        assertEquals(listOf("Hello", "▁world"), never.split("Hello world"))
    }

    @Test
    fun `byte level pre-tokenizer yields byte-level glyph chunks`() {
        val p = PreTokenizer.fromJson(el("""{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}"""))
        assertEquals(listOf("Hello", "Ġworld", "Ċ"), p.split("Hello world\n"))
    }

    @Test
    fun `split regex keeps matches as chunks`() {
        val p = PreTokenizer.fromJson(el("""{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\p{N}+"},"behavior":"Isolated"},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]}"""))
        assertEquals(listOf("ab", "12", "c"), p.split("ab12c"))
        assertTrue(PreTokenizer.isByteLevel(p))
    }

    @Test
    fun `bert pre-tokenizer isolates punctuation and drops whitespace`() {
        assertEquals(listOf("hello", ",", "world", "!"), PreTokenizer.Bert.split("hello, world!"))
    }
}
