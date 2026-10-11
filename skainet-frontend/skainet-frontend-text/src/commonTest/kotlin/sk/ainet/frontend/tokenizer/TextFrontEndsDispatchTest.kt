package sk.ainet.frontend.tokenizer

import sk.ainet.frontend.UnsupportedFrontEndException
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.model.InMemoryCheckpointView
import sk.ainet.model.ModelFormat
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The 0.57 `TokenizerFactoryDispatchTest`, ported to the provider and `describe()` (SKEEP-006 acceptance). */
class TextFrontEndsDispatchTest {

    private fun qwenStyleFields(): Map<String, Any?> {
        val tokens = mutableListOf<String>()
        for (b in 0..255) tokens.add(ByteToUnicode.byteToUnicode[b].toString())
        val merges = mutableListOf<String>()
        fun add(a: String, b: String) { merges.add("$a $b"); tokens.add(a + b) }
        add("H", "e"); add("l", "l"); add("He", "ll"); add("Hell", "o")
        tokens.add("<|end|>")
        val types = IntArray(tokens.size) { if (it == tokens.lastIndex) 3 else 1 }.toList()
        return mapOf(
            "tokenizer.ggml.model" to "gpt2",
            "tokenizer.ggml.tokens" to tokens,
            "tokenizer.ggml.merges" to merges,
            "tokenizer.ggml.token_type" to types,
            "tokenizer.ggml.bos_token_id" to 42,
            "tokenizer.ggml.eos_token_id" to tokens.lastIndex,
            "tokenizer.ggml.add_bos_token" to true,
            "tokenizer.chat_template" to "{{ messages }}",
        )
    }

    @Test
    fun `gguf gpt2 dispatches to byte level BPE`() {
        val tok = TextFrontEnds.fromGguf(qwenStyleFields())
        assertTrue(tok is QwenByteLevelBpeTokenizer)
        val ids = tok.encode("Hello<|end|>")
        assertEquals(2, ids.size)
        assertEquals("Hello<|end|>", tok.decode(ids))
        assertEquals("Hello", tok.decode(ids, skipSpecialTokens = true))
        assertEquals(42, tok.bosTokenId)
        assertContentEquals(intArrayOf(42) + ids, tok.encode("Hello<|end|>", addSpecialTokens = true))
        assertEquals(setOf(tok.eosTokenId!!), tok.specialTokens.stopIds)
    }

    @Test
    fun `resolver builds a text front end with the chat template from gguf fields`() {
        val view = InMemoryCheckpointView(ModelFormat.GGUF, qwenStyleFields() + ("general.architecture" to "qwen2"))
        val resolver = TextFrontEnds.resolver()
        val set = resolver.resolve(view)
        assertEquals("{{ messages }}", set.text!!.chatTemplateSource)
        assertNull(set.audio); assertNull(set.image); assertNull(set.composite)
        val described = resolver.describe(view)
        assertEquals(1, described.size)
        assertEquals("bpe", (described[0] as sk.ainet.frontend.FrontEndSpec.Text).algorithm)
        assertEquals("qwen", resolver.identify(view).family)
    }

    @Test
    fun `gguf llama dispatches to SentencePiece`() {
        val fields = mapOf<String, Any?>(
            "tokenizer.ggml.model" to "llama",
            "tokenizer.ggml.tokens" to listOf("<unk>", "<s>", "</s>", "\u2581", "a"),
            "tokenizer.ggml.scores" to listOf(0.0f, 0.0f, 0.0f, -1.0f, -1.0f),
        )
        assertTrue(TextFrontEnds.fromGguf(fields) is SentencePieceTokenizer)
    }

    @Test
    fun `gguf bert dispatches to WordPiece`() {
        val fields = mapOf<String, Any?>(
            "tokenizer.ggml.model" to "bert",
            "tokenizer.ggml.tokens" to listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##s", "world"),
        )
        val tok = TextFrontEnds.fromGguf(fields)
        assertTrue(tok is WordPieceTokenizer)
        assertContentEquals(intArrayOf(4, 5, 6), tok.encode("Hellos World"))
        assertContentEquals(intArrayOf(2, 4, 5, 6, 3), tok.encode("Hellos World", addSpecialTokens = true))
    }

    @Test
    fun `gguf missing model field throws`() {
        assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.fromGguf(emptyMap()) }
        assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.fromGguf(mapOf("tokenizer.ggml.model" to "t5")) }
    }

    @Test
    fun `tokenizer_json BPE with string merges dispatches to byte level BPE`() {
        val json = """
            {"version":"1.0","added_tokens":[{"id":2,"content":"<|end|>","special":true}],
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"<|end|>":2,"ab":3},"merges":["a b"]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is QwenByteLevelBpeTokenizer)
        assertEquals(listOf(3, 2), tok.encode("ab<|end|>").toList())
    }

    @Test
    fun `tokenizer_json BPE with array merges gives the same tokenizer as string merges - issue 1323`() {
        val json = """
            {"version":"1.0","added_tokens":[{"id":2,"content":"<|end|>","special":true}],
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"<|end|>":2,"ab":3},"merges":[["a","b"]]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json) as QwenByteLevelBpeTokenizer
        assertEquals(1, tok.mergeCount)
        assertEquals(listOf(3, 2), tok.encode("ab<|end|>").toList())
        val bad = json.replace("""[["a","b"]]""", """[["a","b","c"]]""")
        assertFailsWith<IllegalArgumentException> { TextFrontEnds.fromTokenizerJson(bad) }
    }

    @Test
    fun `non-special added tokens match atomically and count in vocabSize - issue 1324`() {
        val json = """
            {"added_tokens":[{"id":5,"content":"[UNK]","special":true},{"id":6,"content":"   ","special":false,"normalized":true}],
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"ĠĠ":4},"merges":["a b","Ġ Ġ"]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertEquals(7, tok.vocabSize)
        assertEquals(listOf(6, 3), tok.encode("   ab").toList())
        assertEquals("   ab", tok.decode(tok.encode("   ab")))
        assertEquals(setOf(5), tok.specialTokens.declared)
        assertEquals("   ab", tok.decode(tok.encode("   ab[UNK]"), skipSpecialTokens = true))
    }

    @Test
    fun `normalizer block is applied before pre-tokenization - issue 1325`() {
        val json = """
            {"normalizer":{"type":"Replace","pattern":{"String":"x"},"content":"a"},
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertEquals(listOf(2), tok.encode("xb").toList())
        val unknown = json.replace(""""type":"Replace","pattern":{"String":"x"},"content":"a"""", """"type":"Weird"""")
        val ex = assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.fromTokenizerJson(unknown) }
        assertTrue("Weird" in ex.message!!)
    }

    @Test
    fun `legacy tokenizer_json without model_type infers BPE from merges`() {
        val json = """
            {"added_tokens":[{"id":2,"content":"<|end|>","special":true}],
             "pre_tokenizer":{"type":"ByteLevel"},
             "model":{"vocab":{"a":0,"b":1,"<|end|>":2,"ab":3},"merges":["a b"]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is QwenByteLevelBpeTokenizer)
        assertEquals(listOf(3, 2), tok.encode("ab<|end|>").toList())
    }

    @Test
    fun `tokenizer_json without model_type and without merges throws`() {
        assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.fromTokenizerJson("""{"model":{"vocab":{"a":0}}}""") }
    }

    @Test
    fun `metaspace BPE with byte fallback - issue 1326`() {
        val json = """
            {"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},
             "pre_tokenizer":null,
             "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
             "added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}],
             "model":{"type":"BPE","unk_token":"<unk>","byte_fallback":true,
               "vocab":{"<unk>":0,"<s>":1,"</s>":2,"<0xC3>":3,"<0xBC>":4,"▁":5,"H":6,"i":7,"▁H":8,"▁Hi":9,"▁r":10,"r":11},
               "merges":[["▁","H"],["▁H","i"],["▁","r"]]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is SpecialTokenSplitter, "added tokens wrap the Metaspace BPE: ${tok::class.simpleName}")
        assertEquals("metaspace-bpe", (tok.spec as sk.ainet.frontend.FrontEndSpec.Text).algorithm)
        assertEquals(listOf(9), tok.encode("Hi").toList())
        // ü is not in the vocab → UTF-8 bytes C3 BC as <0xC3><0xBC>
        assertEquals(listOf(10, 3, 4), tok.encode("rü").toList())
        assertEquals("rü", tok.decode(tok.encode("rü")))
        assertEquals("Hi", tok.decode(tok.encode("Hi")))
        assertEquals(" Hi", tok.decodeToken(9))
        assertEquals(listOf(1, 9, 2), tok.encode("<s>Hi</s>").toList())
        assertEquals(1, tok.bosTokenId); assertEquals(2, tok.eosTokenId)
    }

    @Test
    fun `tokenizer_json Unigram dispatches to the Viterbi lattice`() {
        val json = """
            {"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},
             "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},
             "model":{"type":"Unigram","unk_id":0,
               "vocab":[["<unk>",0.0],["▁",-3.0],["H",-3.0],["i",-3.0],["▁H",-2.5],["▁Hi",-1.0],["▁Hip",-4.0],["p",-3.0]]}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is UnigramTokenizer)
        assertEquals(listOf(5), tok.encode("Hi").toList())         // ▁Hi (-1.0) beats ▁H + i (-5.5)
        assertEquals("Hip", tok.decode(tok.encode("Hip")))          // ▁Hip (-4.0) ties ▁Hi + p (-4.0); both round-trip
        assertEquals(listOf(1, 0), tok.encode("z").toList())         // ▁ is a piece, z is unknown
        assertEquals("Hi", tok.decode(tok.encode("Hi")))
    }

    @Test
    fun `tokenizer_json WordPiece dispatches and encodes with metadata`() {
        val json = """
            {"normalizer":{"type":"BertNormalizer","lowercase":true},
             "pre_tokenizer":{"type":"BertPreTokenizer"},
             "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],
               "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},
             "added_tokens":[{"id":0,"content":"[PAD]","special":true},{"id":1,"content":"[UNK]","special":true},{"id":2,"content":"[CLS]","special":true},{"id":3,"content":"[SEP]","special":true}],
             "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
               "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"un":4,"##aff":5,"##able":6,"hello":7,",":8}}}
        """.trimIndent()
        val tok = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is WordPieceTokenizer, tok::class.simpleName)
        assertContentEquals(intArrayOf(7, 8, 4, 5, 6), tok.encode("Hello, unaffable"))
        assertContentEquals(intArrayOf(2, 7, 8, 4, 5, 6, 3), tok.encode("Hello, unaffable", addSpecialTokens = true))
        val enc = (tok as WordPieceTokenizer).encodeWithMetadata("hello", "un")
        assertContentEquals(intArrayOf(2, 7, 3, 4, 3), enc.ids)
        assertContentEquals(intArrayOf(0, 0, 0, 1, 1), enc.tokenTypeIds!!)
        assertEquals("hello, unaffable", tok.decode(tok.encode("Hello, unaffable")))
        assertEquals(1, tok.encode("zzz").single())
    }

    @Test
    fun `tokenizer_json Unigram with added_tokens is wrapped in the splitter`() {
        val json = """
            {"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}],
             "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["<s>",0.0],["</s>",0.0],["▁",-1.0],["a",-1.0]]}}
        """.trimIndent()
        val tok: Tokenizer = TextFrontEnds.fromTokenizerJson(json)
        assertTrue(tok is SpecialTokenSplitter, "expected SpecialTokenSplitter, got ${tok::class.simpleName}")
        assertEquals(listOf(1, 4, 2), tok.encode("<s>a</s>").toList())
    }

    @Test
    fun `frontend_json selects explicitly and names a missing artifact loudly`() {
        val view = InMemoryCheckpointView.ofText(
            ModelFormat.SAFETENSORS,
            mapOf("model_type" to "bert"),
            mapOf("frontend.json" to """{"modality":"text","algorithm":"wordpiece","artifact":"vocab.txt"}""", "vocab.txt" to "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n"),
        )
        val fe = TextFrontEnds.resolver().resolveText(view)
        assertContentEquals(intArrayOf(4, 5), fe.tokenizer.encode("Hello world"))
        val broken = InMemoryCheckpointView.ofText(ModelFormat.SAFETENSORS, mapOf("model_type" to "bert"), mapOf("frontend.json" to """{"modality":"text","algorithm":"wordpiece","artifact":"vocab.txt"}"""))
        val ex = assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.resolver().resolveText(broken) }
        assertTrue("vocab.txt" in ex.message!!)
    }

    @Test
    fun `a view without text evidence resolves to null and resolveText names the registry`() {
        val view = InMemoryCheckpointView(ModelFormat.ONNX, mapOf("architecture" to "yolo"))
        assertNull(TextFrontEnds.resolver().resolve(view).text)
        val ex = assertFailsWith<UnsupportedFrontEndException> { TextFrontEnds.resolver().resolveText(view) }
        assertTrue("llama" in ex.message!! && "yolo" in ex.message!!, ex.message)
    }
}
