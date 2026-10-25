package sk.ainet.io.tokenizer

import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.tokenizer.SentencePieceTokenizer
import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.io.ByteArrayRandomAccessSource
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import java.io.File
import kotlin.test.Test

/**
 * Timing harness (not an assertion) for loading a tokenizer from a GGUF file: metadata parse from memory and from the
 * file, then the tokenizer build and its larger pieces. Set `SKAINET_GGUF_FILE` to a GGUF with tokenizer metadata;
 * without it the test prints a skip notice and passes.
 *
 *   SKAINET_GGUF_FILE=/path/model.gguf ./gradlew :skainet-io:skainet-io-core:jvmTest --tests '*GgufTokenizerLoadBenchmarkTest*'
 */
class GgufTokenizerLoadBenchmarkTest {

    private fun median(runs: Int, block: () -> Unit): Long {
        repeat(2) { block() }
        val ms = LongArray(runs) { val t = System.nanoTime(); block(); (System.nanoTime() - t) / 1_000_000 }
        ms.sort()
        return ms[runs / 2]
    }

    @Test
    fun timeTokenizerLoad() {
        val path = System.getenv("SKAINET_GGUF_FILE")
        if (path.isNullOrBlank() || !File(path).isFile) {
            println("[skip] GgufTokenizerLoadBenchmarkTest: set SKAINET_GGUF_FILE to a GGUF file")
            return
        }
        val file = File(path)
        val bytes = file.readBytes()
        val runs = 5
        val fields = StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes)).use { it.fields }
        val tokens = (fields["tokenizer.ggml.tokens"] as List<*>).filterIsInstance<String>()

        val readMemory = median(runs) { StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes)).use { it.fields } }
        val readFile = median(3) { JvmRandomAccessSource.open(file).use { src -> StreamingGGUFReader.open(src).use { it.fields } } }
        val copyTokens = median(runs) { (fields["tokenizer.ggml.tokens"] as List<*>).filterIsInstance<String>() }
        val copyScores = median(runs) { (fields["tokenizer.ggml.scores"] as List<*>).mapNotNull { (it as? Number)?.toFloat() } }
        val vocabulary = median(runs) { ArrayVocabulary(tokens) }
        val sentencePiece = median(runs) { SentencePieceTokenizer.fromGgufFields(fields) }
        val frontEnd = median(runs) { TextFrontEnds.fromGguf(fields) }

        println(
            "[bench] ${file.name} ${bytes.size} bytes, ${tokens.size} tokens, median of $runs (file: 3) in ms: " +
                "read.memory=$readMemory read.file=$readFile | build.frontEnd=$frontEnd " +
                "(sentencePiece.fromGgufFields=$sentencePiece, ArrayVocabulary=$vocabulary, " +
                "tokens copy=$copyTokens, scores copy=$copyScores)"
        )
    }
}
