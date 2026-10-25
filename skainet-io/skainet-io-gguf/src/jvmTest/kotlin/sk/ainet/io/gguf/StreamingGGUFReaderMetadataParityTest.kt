package sk.ainet.io.gguf

import sk.ainet.io.ByteArrayRandomAccessSource
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufWriteRequest
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The buffered metadata parse in [StreamingGGUFReader] must produce exactly what the earlier per-value parse produced:
 * the same keys in the same order, the same value classes, the same element classes inside arrays. [LegacyFieldParser]
 * is that earlier parse, kept here as the reference.
 */
class StreamingGGUFReaderMetadataParityTest {

    @Test
    fun everyValueTypeMatchesThePerValueParse() {
        val bytes = allTypesGguf()
        assertParity(bytes)
        StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes)).use { r ->
            assertEquals(listOf(1, -2, Int.MAX_VALUE), r.fields["test.i32s"])
            assertEquals(listOf(1u, UInt.MAX_VALUE), r.fields["test.u32s"])
            assertEquals(listOf("", "ä€𝄞", "x"), r.fields["test.strings"])
            assertEquals(2, r.tensors.size)
            assertEquals(0L, r.dataOffset % 32)
        }
    }

    @Test
    fun tokenizerShapedArraysMatch() = assertParity(tokenizerGguf(262_172))

    @Test
    fun aSourceThatReturnsFewBytesPerReadGivesTheSameFields() {
        val bytes = tokenizerGguf(20_000)
        val expected = LegacyFieldParser(ByteArrayRandomAccessSource(bytes)).parse()
        val actual = StreamingGGUFReader.open(TrickleSource(bytes, maxPerRead = 7)).use { it.fields }
        assertSameFields(expected, actual)
    }

    @Test
    fun theFileSourceGivesTheSameFields() {
        val file = File.createTempFile("parity_", ".gguf").apply { deleteOnExit(); writeBytes(allTypesGguf()) }
        val expected = LegacyFieldParser(ByteArrayRandomAccessSource(file.readBytes())).parse()
        val actual = JvmRandomAccessSource.open(file).use { src -> StreamingGGUFReader.open(src).use { it.fields } }
        assertSameFields(expected, actual)
    }

    @Test
    fun aTruncatedFileFailsWithAClearError() {
        val bytes = allTypesGguf()
        val metadataEnd = StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes)).use { it.dataOffset }.toInt()
        for (cut in listOf(3, 20, 40, metadataEnd / 2, metadataEnd - 40)) {
            assertFailsWith<IllegalArgumentException>("cut at $cut") {
                StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes.copyOf(cut)))
            }
        }
    }

    @Test
    fun anArrayCountLargerThanTheFileFailsBeforeAllocating() {
        val b = Gguf()
        b.header(tensors = 0, kvs = 1)
        b.key("bad.array"); b.int(GGUFValueType.ARRAY.value); b.int(GGUFValueType.FLOAT32.value); b.long(Int.MAX_VALUE.toLong())
        b.int(0)
        val e = assertFailsWith<IllegalArgumentException> { StreamingGGUFReader.open(ByteArrayRandomAccessSource(b.bytes())) }
        assertTrue("left" in e.message.orEmpty(), e.message)
    }

    @Test
    fun parsedFieldsWriteBackThroughTheGgufWriter() {
        val fields = StreamingGGUFReader.open(ByteArrayRandomAccessSource(tokenizerGguf(5_000))).use { it.fields }
        val metadata = fields.filterKeys { !it.startsWith("GGUF.") }.mapValues { it.value!! }
        val (_, written) = GGUFWriter.writeToByteArray(GgufWriteRequest(metadata, emptyList(), emptyMap()))
        val reread = StreamingGGUFReader.open(ByteArrayRandomAccessSource(written)).use { it.fields }
        for ((key, value) in metadata) assertEquals(value, reread[key], key)
    }

    /** Opt-in check on a real model or tokenizer GGUF: `SKAINET_GGUF_FILE=/path/file.gguf`. */
    @Test
    fun aRealGgufFileMatchesWhenConfigured() {
        val path = System.getenv("SKAINET_GGUF_FILE")
        if (path.isNullOrBlank() || !File(path).isFile) {
            println("[skip] StreamingGGUFReaderMetadataParityTest: set SKAINET_GGUF_FILE to check a real GGUF file")
            return
        }
        assertParity(File(path).readBytes())
    }

    // ---------------------------------------------------------------------------------------------------------------

    private fun assertParity(bytes: ByteArray) {
        val expected = LegacyFieldParser(ByteArrayRandomAccessSource(bytes)).parse()
        val actual = StreamingGGUFReader.open(ByteArrayRandomAccessSource(bytes)).use { it.fields }
        assertSameFields(expected, actual)
    }

    private fun assertSameFields(expected: Map<String, Any?>, actual: Map<String, Any?>) {
        assertEquals(expected.keys.toList(), actual.keys.toList(), "keys and their order")
        for ((key, value) in expected) assertSameValue(value, actual[key], key)
    }

    private fun assertSameValue(expected: Any?, actual: Any?, path: String) {
        if (expected is List<*>) {
            if (actual !is List<*>) fail("$path: expected a List, got ${actual?.let { it::class }}")
            assertEquals(expected.size, actual.size, "$path: size")
            for (i in expected.indices) assertSameValue(expected[i], actual[i], "$path[$i]")
            assertEquals(expected, actual, path)
            assertEquals(expected.hashCode(), actual.hashCode(), "$path: hashCode")
            return
        }
        assertEquals(expected?.let { it::class }, actual?.let { it::class }, "$path: class")
        assertEquals(expected, actual, path)
    }

    /** A source that never returns more than [maxPerRead] bytes per buffer read, like a slow stream. */
    private class TrickleSource(private val bytes: ByteArray, private val maxPerRead: Int) : RandomAccessSource {
        override val size: Long get() = bytes.size.toLong()
        override fun readAt(position: Long, length: Int): ByteArray = bytes.copyOfRange(position.toInt(), position.toInt() + length)
        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            val n = minOf(length, maxPerRead, bytes.size - position.toInt())
            if (n <= 0) return -1
            bytes.copyInto(buffer, offset, position.toInt(), position.toInt() + n)
            return n
        }
        override fun close() {}
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Synthetic files

    private class Gguf {
        private val out = ByteArrayOutputStream()
        private fun le(n: Int, fill: ByteBuffer.() -> Unit) = out.write(ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN).apply(fill).array())
        fun byte(v: Int) = out.write(v)
        fun short(v: Int) = le(2) { putShort(v.toShort()) }
        fun int(v: Int) = le(4) { putInt(v) }
        fun long(v: Long) = le(8) { putLong(v) }
        fun float(v: Float) = le(4) { putFloat(v) }
        fun double(v: Double) = le(8) { putDouble(v) }
        fun string(s: String) { val b = s.encodeToByteArray(); long(b.size.toLong()); out.write(b) }
        fun key(k: String) = string(k)
        fun header(tensors: Long, kvs: Long) { int(GGUF_MAGIC.toInt()); int(3); long(tensors); long(kvs) }
        fun kv(k: String, type: GGUFValueType, value: Gguf.() -> Unit) { key(k); int(type.value); value() }
        fun array(k: String, element: GGUFValueType, count: Int, each: Gguf.(Int) -> Unit) {
            key(k); int(GGUFValueType.ARRAY.value); int(element.value); long(count.toLong()); for (i in 0 until count) each(i)
        }
        fun padTo(alignment: Int) { while (out.size() % alignment != 0) out.write(0) }
        fun bytes(): ByteArray = out.toByteArray()
    }

    private fun allTypesGguf(): ByteArray {
        val g = Gguf()
        g.header(tensors = 2, kvs = 26)
        g.kv("general.architecture", GGUFValueType.STRING) { string("test") }
        g.kv("test.u8", GGUFValueType.UINT8) { byte(250) }
        g.kv("test.i8", GGUFValueType.INT8) { byte(0x80) }
        g.kv("test.u16", GGUFValueType.UINT16) { short(0xFFFE) }
        g.kv("test.i16", GGUFValueType.INT16) { short(-3) }
        g.kv("test.u32", GGUFValueType.UINT32) { int(-1) }
        g.kv("test.i32", GGUFValueType.INT32) { int(-7) }
        g.kv("test.f32", GGUFValueType.FLOAT32) { float(-1.5f) }
        g.kv("test.u64", GGUFValueType.UINT64) { long(-1L) }
        g.kv("test.i64", GGUFValueType.INT64) { long(Long.MIN_VALUE) }
        g.kv("test.f64", GGUFValueType.FLOAT64) { double(Math.PI) }
        g.kv("test.bool", GGUFValueType.BOOL) { byte(1) }
        g.kv("test.long_string", GGUFValueType.STRING) { string("ab€".repeat(40_000)) }   // longer than the read block
        g.array("test.u8s", GGUFValueType.UINT8, 3) { byte(listOf(0, 127, 255)[it]) }
        g.array("test.i8s", GGUFValueType.INT8, 2) { byte(listOf(0x7F, 0x80)[it]) }
        g.array("test.bools", GGUFValueType.BOOL, 3) { byte(listOf(0, 1, 2)[it]) }
        g.array("test.u16s", GGUFValueType.UINT16, 2) { short(listOf(1, 0xFFFF)[it]) }
        g.array("test.i16s", GGUFValueType.INT16, 2) { short(listOf(-1, 300)[it]) }
        g.array("test.u32s", GGUFValueType.UINT32, 2) { int(listOf(1, -1)[it]) }
        g.array("test.i32s", GGUFValueType.INT32, 3) { int(listOf(1, -2, Int.MAX_VALUE)[it]) }
        g.array("test.f32s", GGUFValueType.FLOAT32, 4) { float(listOf(0f, -0f, Float.NaN, 1e-30f)[it]) }
        g.array("test.u64s", GGUFValueType.UINT64, 2) { long(listOf(0L, -1L)[it]) }
        g.array("test.i64s", GGUFValueType.INT64, 2) { long(listOf(Long.MAX_VALUE, -5L)[it]) }
        g.array("test.f64s", GGUFValueType.FLOAT64, 2) { double(listOf(1.0, Double.NEGATIVE_INFINITY)[it]) }
        g.array("test.strings", GGUFValueType.STRING, 3) { string(listOf("", "ä€𝄞", "x")[it]) }
        g.array("test.nested", GGUFValueType.ARRAY, 3) { i ->
            int(GGUFValueType.INT32.value); long(i.toLong()); repeat(i) { int(it * 10) }
        }
        // Two F32 tensors of 4 elements
        for ((i, name) in listOf("t.a", "t.b").withIndex()) {
            string(g, name); g.int(1); g.long(4); g.int(GGMLQuantizationType.F32.value); g.long(i * 32L)
        }
        g.padTo(32)
        repeat(64) { g.byte(0) }
        return g.bytes()
    }

    private fun string(g: Gguf, s: String) = g.string(s)

    /** Shaped like a SentencePiece tokenizer GGUF: string tokens, float scores, int token types, a few scalars. */
    private fun tokenizerGguf(n: Int): ByteArray {
        val rnd = Random(42)
        val alphabet = "abcdefghijklmnopqrstuvwxyzäöüß▁<>_0123456789€"
        val g = Gguf()
        g.header(tensors = 0, kvs = 7)
        g.kv("general.architecture", GGUFValueType.STRING) { string("gemma3") }
        g.kv("tokenizer.ggml.model", GGUFValueType.STRING) { string("llama") }
        g.array("tokenizer.ggml.tokens", GGUFValueType.STRING, n) { i ->
            string(if (i < 3) listOf("<pad>", "<eos>", "<bos>")[i] else String(CharArray(1 + rnd.nextInt(16)) { alphabet[rnd.nextInt(alphabet.length)] }))
        }
        g.array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, n) { g.float(-it.toFloat() / 7f) }
        g.array("tokenizer.ggml.token_type", GGUFValueType.INT32, n) { int(if (it < 3) 3 else 1) }
        g.kv("tokenizer.ggml.bos_token_id", GGUFValueType.UINT32) { int(2) }
        g.kv("tokenizer.ggml.add_bos_token", GGUFValueType.BOOL) { byte(1) }
        return g.bytes()
    }

    // ---------------------------------------------------------------------------------------------------------------

    /** The per-value KV parse as it was before the buffered cursor (header and KV section only). */
    private class LegacyFieldParser(private val source: RandomAccessSource) {
        private val fields = LinkedHashMap<String, Any?>()

        fun parse(): LinkedHashMap<String, Any?> {
            var offset = 8L
            val tensorCount = readULong(offset); offset += 8
            val kvCount = readULong(offset); offset += 8
            fields["GGUF.tensor_count"] = tensorCount
            fields["GGUF.kv_count"] = kvCount
            fields["GGUF.version"] = readUInt(4)
            for (i in 0 until kvCount.toInt()) {
                val keyLength = readULong(offset); offset += 8
                val keyName = source.readAt(offset, keyLength.toInt()).decodeToString(); offset += keyLength.toLong()
                val valueType = readUInt(offset).toInt(); offset += 4
                val gtype = GGUFValueType.entries.first { it.value == valueType }
                val (consumed, value) = readFieldValue(offset, gtype); offset += consumed
                if (fields.containsKey(keyName)) fields["${keyName}_dup_$i"] = value else fields[keyName] = value
            }
            return fields
        }

        private fun readFieldValue(offset: Long, gtype: GGUFValueType): Pair<Long, Any?> = when (gtype) {
            GGUFValueType.UINT8 -> 1L to source.readByteAt(offset).toUByte()
            GGUFValueType.INT8 -> 1L to source.readByteAt(offset)
            GGUFValueType.UINT16 -> 2L to readInt(offset, 2).toUShort()
            GGUFValueType.INT16 -> 2L to readInt(offset, 2).toShort()
            GGUFValueType.UINT32 -> 4L to readUInt(offset)
            GGUFValueType.INT32 -> 4L to readInt(offset, 4)
            GGUFValueType.FLOAT32 -> 4L to Float.fromBits(readInt(offset, 4))
            GGUFValueType.UINT64 -> 8L to readULong(offset)
            GGUFValueType.INT64 -> 8L to readULong(offset).toLong()
            GGUFValueType.FLOAT64 -> 8L to Double.fromBits(readULong(offset).toLong())
            GGUFValueType.BOOL -> 1L to (source.readByteAt(offset) != 0.toByte())
            GGUFValueType.STRING -> {
                val length = readULong(offset)
                (8 + length.toLong()) to source.readAt(offset + 8, length.toInt()).decodeToString()
            }
            GGUFValueType.ARRAY -> {
                var o = offset
                val elementType = readUInt(o).toInt(); o += 4
                val length = readULong(o); o += 8
                val et = GGUFValueType.entries.first { it.value == elementType }
                val result = mutableListOf<Any?>()
                for (i in 0 until length.toInt()) {
                    val (c, v) = readFieldValue(o, et); result.add(v); o += c
                }
                (o - offset) to result
            }
        }

        private fun readInt(position: Long, n: Int): Int {
            val b = source.readAt(position, n)
            var v = 0
            for (i in 0 until n) v = v or ((b[i].toInt() and 0xFF) shl (8 * i))
            return v
        }

        private fun readUInt(position: Long): UInt = readInt(position, 4).toUInt()

        private fun readULong(position: Long): ULong {
            val b = source.readAt(position, 8)
            var v = 0L
            for (i in 0 until 8) v = v or ((b[i].toLong() and 0xFF) shl (8 * i))
            return v.toULong()
        }
    }
}
