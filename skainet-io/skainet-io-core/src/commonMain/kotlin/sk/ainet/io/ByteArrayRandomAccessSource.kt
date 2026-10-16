package sk.ainet.io

/**
 * A [RandomAccessSource] over bytes already in memory (SKEEP-007 §T0.4).
 *
 * The browser has no file system to map or seek in: a checkpoint arrives as one download (or as
 * bundled resource bytes) and has to be served to the synchronous readers — `StreamingGGUFReader`,
 * the SafeTensors header parser, `StreamingGgufParametersLoader` — from memory. Every web sample
 * so far wrote its own copy of this class; this is the engine's.
 *
 * The data is held as a list of [chunks] so a file larger than a single Kotlin array (`Int.MAX_VALUE`
 * bytes, and 2 GB per allocation on the web) can still be addressed with a `Long` position. A
 * single-array source is the common case and has the convenience constructor.
 *
 * Reads copy out of the chunks; nothing is retained beyond the chunks themselves, and [close] is
 * a no-op — the caller owns the lifetime of the bytes.
 */
public class ByteArrayRandomAccessSource(
    private val chunks: List<ByteArray>,
) : RandomAccessSource {

    /** Single-array convenience: the whole file in one `ByteArray`. */
    public constructor(data: ByteArray) : this(listOf(data))

    init {
        require(chunks.isNotEmpty()) { "ByteArrayRandomAccessSource needs at least one chunk (an empty array is fine)" }
        // Every chunk but the last must be the same length so a position maps to a chunk by division.
        val lead = chunks.first().size
        for (i in 0 until chunks.size - 1) {
            require(chunks[i].size == lead && lead > 0) {
                "ByteArrayRandomAccessSource: all chunks but the last must have the same non-zero length " +
                    "(chunk $i has ${chunks[i].size}, first has $lead)"
            }
        }
    }

    private val chunkSize: Long = chunks.first().size.toLong()

    override val size: Long = chunks.fold(0L) { acc, c -> acc + c.size }

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of data: position=$position, length=$length, size=$size"
        }
        val out = ByteArray(length)
        readInto(position, out, 0, length)
        return out
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0 && length >= 0 && offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }
        if (position >= size) return 0
        val available = minOf(length.toLong(), size - position).toInt()
        readInto(position, buffer, offset, available)
        return available
    }

    override fun readByteAt(position: Long): Byte {
        require(position in 0 until size) { "Position out of range: $position (size=$size)" }
        return if (chunks.size == 1) {
            chunks[0][position.toInt()]
        } else {
            chunks[(position / chunkSize).toInt()][(position % chunkSize).toInt()]
        }
    }

    private fun readInto(position: Long, buffer: ByteArray, offset: Int, length: Int) {
        if (length == 0) return
        if (chunks.size == 1) {
            val p = position.toInt()
            chunks[0].copyInto(buffer, offset, p, p + length)
            return
        }
        var remaining = length
        var dst = offset
        var pos = position
        while (remaining > 0) {
            val chunk = chunks[(pos / chunkSize).toInt()]
            val inChunk = (pos % chunkSize).toInt()
            val n = minOf(remaining, chunk.size - inChunk)
            chunk.copyInto(buffer, dst, inChunk, inChunk + n)
            remaining -= n
            dst += n
            pos += n
        }
    }

    override fun close() {
        // Nothing to release: the caller owns the arrays.
    }
}
