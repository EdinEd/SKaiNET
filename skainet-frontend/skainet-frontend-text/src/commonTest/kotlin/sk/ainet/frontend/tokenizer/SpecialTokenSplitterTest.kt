package sk.ainet.frontend.tokenizer

import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class SpecialTokenSplitterTest {

    /**
     * Toy base tokenizer that turns each character into one id (matching its
     * code point) and concatenates ids back to chars on decode. Lets us test
     * the splitter's segmentation logic without dragging in a real BPE/SP
     * vocab.
     */
    private object CharVocabulary : Vocabulary {
        override val size: Int = 0x10000
        override fun tokenToId(piece: String): Int? = if (piece.length == 1) piece[0].code else null
        override fun idToToken(id: Int): String? = if (id in 0 until 0x10000) id.toChar().toString() else null
    }

    private object CharBase : Tokenizer {
        override val vocabulary: Vocabulary = CharVocabulary
        override val specialTokens: SpecialTokens = SpecialTokens.NONE
        override fun encode(text: String): IntArray = IntArray(text.length) { text[it].code }
        override fun decode(ids: IntArray): String =
            buildString(ids.size) { for (id in ids) append(id.toChar()) }
    }

    @Test
    fun `lstrip and rstrip flags absorb the adjacent whitespace into the added token`() {
        // The mmBERT / ModernBERT `<mask>` is declared `lstrip: true`: the reference drops the
        // whitespace before it instead of handing it to the preceding segment (issue 1324 corpus).
        val splitter = SpecialTokenSplitter(
            CharBase,
            listOf(
                AddedToken(content = "<mask>", id = 4, special = true, lstrip = true),
                AddedToken(content = "<pad>", id = 0, special = true),
                AddedToken(content = "<r>", id = 7, special = false, rstrip = true),
            ),
        )
        assertContentEquals(
            CharBase.encode("please") + intArrayOf(4) + CharBase.encode(" this and ") + intArrayOf(0) + CharBase.encode(" that"),
            splitter.encode("please <mask> this and <pad> that"),
        )
        assertContentEquals(CharBase.encode("a") + intArrayOf(7) + CharBase.encode("b"), splitter.encode("a<r>   b"))
        // Flags change encoding only; decode still emits the literal.
        assertEquals("please<mask> this", splitter.decode(splitter.encode("please <mask> this")))
    }

    @Test
    fun `empty special tokens map is a pass-through`() {
        val splitter = SpecialTokenSplitter(CharBase, emptyMap())
        val ids = splitter.encode("hello")
        assertEquals(5, ids.size)
        assertEquals("hello", splitter.decode(ids))
    }

    @Test
    fun `single special token between text segments encodes and decodes atomically`() {
        val specials = mapOf("<|end|>" to 99999)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("hi<|end|>bye")
        // 'h','i' (2) + special (1) + 'b','y','e' (3) = 6 ids
        assertEquals(6, ids.size)
        assertEquals(99999, ids[2])
        assertEquals("hi<|end|>bye", splitter.decode(ids))
    }

    @Test
    fun `longest match wins over prefix overlap`() {
        // Both "<|im" and "<|im_start|>" registered; longer should win.
        val specials = mapOf(
            "<|im" to 7,
            "<|im_start|>" to 8,
        )
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("a<|im_start|>b")
        // 'a' (1) + special-8 (1) + 'b' (1) = 3 ids
        assertContentEquals(intArrayOf('a'.code, 8, 'b'.code), ids)
        assertEquals("a<|im_start|>b", splitter.decode(ids))
    }

    @Test
    fun `a rejected single_word match falls back to the next shorter token with the same first char`() {
        val splitter = SpecialTokenSplitter(
            CharBase,
            listOf(
                AddedToken(content = "word", id = 5, special = false, singleWord = true),
                AddedToken(content = "wo", id = 6, special = false),
                AddedToken(content = "wordy-long", id = 9, special = false),   // longer than what is left of the text
            ),
        )
        // "word" inside "swordx" is not a whole word, so the shorter "wo" matches instead
        assertContentEquals(intArrayOf('s'.code, 6, 'r'.code, 'd'.code, 'x'.code), splitter.encode("swordx"))
        assertContentEquals(intArrayOf('a'.code, ' '.code, 5), splitter.encode("a word"))
    }

    @Test
    fun `consecutive specials emit no spurious base segments`() {
        val specials = mapOf("<a>" to 1, "<b>" to 2)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("<a><b>")
        assertContentEquals(intArrayOf(1, 2), ids)
        assertEquals("<a><b>", splitter.decode(ids))
    }

    @Test
    fun `text with no specials at all returns base encode unchanged`() {
        val specials = mapOf("<unused>" to 42)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("plain")
        assertEquals(5, ids.size)
        assertEquals("plain", splitter.decode(ids))
    }

    @Test
    fun `bos and eos default to base when not overridden`() {
        val base = object : Tokenizer {
            override val vocabulary: Vocabulary = CharVocabulary
            override val specialTokens: SpecialTokens = SpecialTokens(bos = 1, eos = 2)
            override fun encode(text: String): IntArray = IntArray(0)
            override fun decode(ids: IntArray): String = ""
        }
        val splitter = SpecialTokenSplitter(base, emptyMap())
        assertEquals(1, splitter.bosTokenId)
        assertEquals(2, splitter.eosTokenId)
    }

    @Test
    fun `bos and eos overrides take precedence over base`() {
        val splitter = SpecialTokenSplitter(
            base = CharBase,
            addedTokens = emptyMap(),
            bosTokenId = 100,
            eosTokenId = 200,
        )
        assertEquals(100, splitter.bosTokenId)
        assertEquals(200, splitter.eosTokenId)
    }
}
