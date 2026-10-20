package sk.ainet.io.safetensors

/**
 * In-memory sharded SafeTensors fixtures for the provider-based entry points: shard bytes and the
 * index text, built in common Kotlin so the JVM, wasmJs and native tests share one checkpoint.
 *
 * Layout (the JVM file fixture's, without the F16 shard entry):
 * - `model-00001-of-00002.safetensors`: `alpha.weight` (F32 2×2), `bravo.weight` (BF16 4)
 * - `model-00002-of-00002.safetensors`: `zulu.weight` (F32 3)
 */
internal object ShardFixtures {

    class Entry(val name: String, val dtype: String, val shape: List<Long>, val bytes: ByteArray)

    const val SHARD_1 = "model-00001-of-00002.safetensors"
    const val SHARD_2 = "model-00002-of-00002.safetensors"

    val alphaValues = floatArrayOf(1f, 2f, 3f, 4f)
    val bravoValues = floatArrayOf(0.5f, -1.0f, 2.5f, -64.0f)   // exactly representable in BF16
    val zuluValues = floatArrayOf(-7f, 0f, 42f)

    /** Little-endian FP32. */
    fun fp32Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 4)
        for ((i, v) in values.withIndex()) {
            val bits = v.toRawBits()
            out[i * 4] = (bits and 0xFF).toByte()
            out[i * 4 + 1] = ((bits ushr 8) and 0xFF).toByte()
            out[i * 4 + 2] = ((bits ushr 16) and 0xFF).toByte()
            out[i * 4 + 3] = ((bits ushr 24) and 0xFF).toByte()
        }
        return out
    }

    /** Little-endian BF16: the upper half of the FP32 bits, rounded to nearest even. */
    fun bf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for ((i, v) in values.withIndex()) {
            val bits = v.toRawBits()
            val rounding = 0x7FFF + ((bits ushr 16) and 1)
            val bf16 = (bits + rounding) ushr 16
            out[i * 2] = (bf16 and 0xFF).toByte()
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** One SafeTensors file: `u64 header size | JSON header | tensor bytes` in entry order. */
    fun shard(entries: List<Entry>, metadata: Map<String, String> = emptyMap()): ByteArray {
        var offset = 0L
        val headerEntries = ArrayList<String>()
        if (metadata.isNotEmpty()) {
            headerEntries += "\"__metadata__\": {" + metadata.entries.joinToString(", ") { (k, v) -> "\"$k\": \"$v\"" } + "}"
        }
        for (e in entries) {
            val start = offset
            offset += e.bytes.size
            headerEntries += "\"${e.name}\": {\"dtype\": \"${e.dtype}\", \"shape\": [${e.shape.joinToString(", ")}], \"data_offsets\": [$start, $offset]}"
        }
        val header = headerEntries.joinToString(prefix = "{", postfix = "}", separator = ", ").encodeToByteArray()
        val out = ByteArray(8 + header.size + offset.toInt())
        for (i in 0 until 8) out[i] = ((header.size.toLong() shr (i * 8)) and 0xFF).toByte()
        header.copyInto(out, 8)
        var p = 8 + header.size
        for (e in entries) {
            e.bytes.copyInto(out, p)
            p += e.bytes.size
        }
        return out
    }

    /** The two-shard checkpoint: shard name → bytes. */
    fun twoShards(): Map<String, ByteArray> = mapOf(
        SHARD_1 to shard(
            listOf(
                Entry("alpha.weight", "F32", listOf(2L, 2L), fp32Bytes(alphaValues)),
                Entry("bravo.weight", "BF16", listOf(4L), bf16Bytes(bravoValues)),
            ),
            metadata = mapOf("format" to "pt"),
        ),
        SHARD_2 to shard(listOf(Entry("zulu.weight", "F32", listOf(3L), fp32Bytes(zuluValues)))),
    )

    /** `model.safetensors.index.json` for [twoShards]. */
    fun twoShardIndex(shards: Map<String, ByteArray> = twoShards()): String = """
        {
          "metadata": {"total_size": ${shards.values.sumOf { it.size.toLong() }}},
          "weight_map": {
            "alpha.weight": "$SHARD_1",
            "bravo.weight": "$SHARD_1",
            "zulu.weight": "$SHARD_2"
          }
        }
    """.trimIndent()
}
