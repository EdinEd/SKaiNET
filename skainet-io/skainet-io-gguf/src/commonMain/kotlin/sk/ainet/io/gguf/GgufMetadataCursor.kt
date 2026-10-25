package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource

/**
 * Sequential little-endian reader over the GGUF header, KV and tensor-info sections.
 *
 * Reads the source in [WINDOW]-sized blocks through the buffer form of [RandomAccessSource.readAt] and decodes values
 * from that block, so a primitive costs a few array reads instead of one `readAt` call (and, on a file source, one
 * system call) per value. A tokenizer GGUF has arrays of ~262k strings, floats and ints; read one value at a time that
 * is over a million source calls and allocations.
 */
internal class GgufMetadataCursor(private val source: RandomAccessSource, start: Long = 0L) {

    private var buf = ByteArray(WINDOW)
    private var bufStart = start   // file offset of buf[0]
    private var bufLen = 0         // valid bytes in buf
    private var p = 0              // read index in buf

    /** File offset of the next byte to read. */
    val position: Long get() = bufStart + p

    /** Make [n] bytes available at [p], refilling (and growing) the block when they are not. */
    private fun require(n: Int) {
        if (bufLen - p >= n) return
        val at = position
        if (n > buf.size) buf = ByteArray(maxOf(n, buf.size * 2))
        var filled = 0
        val want = minOf(buf.size.toLong(), source.size - at).toInt()
        while (filled < want) {
            val r = source.readAt(at + filled, buf, filled, want - filled)
            if (r <= 0) break
            filled += r
        }
        bufStart = at
        bufLen = filled
        p = 0
        if (filled < n) throw IllegalArgumentException("Unexpected end of GGUF metadata: need $n bytes at offset $at, ${source.size - at} left")
    }

    fun byte(): Byte { require(1); return buf[p++] }

    fun short(): Short { require(2); val v = (buf[p].toInt() and 0xFF) or ((buf[p + 1].toInt() and 0xFF) shl 8); p += 2; return v.toShort() }

    fun int(): Int {
        require(4)
        val b = buf; val i = p
        p += 4
        return (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8) or
            ((b[i + 2].toInt() and 0xFF) shl 16) or ((b[i + 3].toInt() and 0xFF) shl 24)
    }

    fun long(): Long {
        require(8)
        val lo = int().toLong() and 0xFFFFFFFFL
        val hi = int().toLong()
        return lo or (hi shl 32)
    }

    fun uint(): UInt = int().toUInt()

    fun ulong(): ULong = long().toULong()

    /** A GGUF string: u64 byte length, then UTF-8 bytes. */
    fun string(): String {
        val length = checkedLength(ulong(), "string")
        require(length)
        val s = buf.decodeToString(p, p + length)
        p += length
        return s
    }

    // Hot arrays (a tokenizer has ~262k tokens, scores and token types) are decoded straight into the output list, one
    // block of at most [BLOCK] elements per call of a small method, with no per-element calls into the primitive
    // readers. Short calls give a JIT runtime a method it can compile after the first blocks instead of one long loop
    // that runs interpreted for its whole call (measured on a debuggable Android app: first parse ≈ 10 % faster).

    /** [count] GGUF strings appended to [out]; the same values as [string] per element. */
    fun strings(count: Int, out: MutableList<Any?>) {
        var left = count
        while (left > 0) {
            val done = stringsInBlock(minOf(left, BLOCK), out)
            if (done > 0) {
                left -= done
            } else {   // the next string straddles the block end (or is invalid): refill through string()
                out.add(string())
                left--
            }
        }
    }

    /** Strings whose length and bytes lie wholly inside the current block, up to [max]; 0 when the next one does not. */
    private fun stringsInBlock(max: Int, out: MutableList<Any?>): Int {
        val b = buf
        val end = bufLen
        var i = p
        var n = 0
        while (n < max && end - i >= 8) {
            val lo = (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8) or
                ((b[i + 2].toInt() and 0xFF) shl 16) or ((b[i + 3].toInt() and 0xFF) shl 24)
            val hi = (b[i + 4].toInt() and 0xFF) or ((b[i + 5].toInt() and 0xFF) shl 8) or
                ((b[i + 6].toInt() and 0xFF) shl 16) or ((b[i + 7].toInt() and 0xFF) shl 24)
            if (hi != 0 || lo < 0 || end - i - 8 < lo) break   // too long or not wholly in this block: string() handles it
            out.add(b.decodeToString(i + 8, i + 8 + lo))
            i += 8 + lo
            n++
        }
        p = i
        return n
    }

    /** [count] 32-bit elements appended to [out] as `Int` ([INT32]), `UInt` ([UINT32]) or `Float` ([FLOAT32]). */
    fun int32s(count: Int, kind: Int, out: MutableList<Any?>) {
        var left = count
        while (left > 0) {
            require(4)
            left -= int32sInBlock(minOf(left, BLOCK, (bufLen - p) / 4), kind, out)
        }
    }

    private fun int32sInBlock(n: Int, kind: Int, out: MutableList<Any?>): Int {
        val b = buf
        var q = p
        for (k in 0 until n) {
            val v = (b[q].toInt() and 0xFF) or ((b[q + 1].toInt() and 0xFF) shl 8) or
                ((b[q + 2].toInt() and 0xFF) shl 16) or ((b[q + 3].toInt() and 0xFF) shl 24)
            out.add(
                when (kind) {
                    FLOAT32 -> Float.fromBits(v)
                    UINT32 -> v.toUInt()
                    else -> v
                }
            )
            q += 4
        }
        p = q
        return n
    }

    /** An array or string length as an Int index, or a clear error for a corrupt or oversized count. */
    fun checkedLength(value: ULong, what: String): Int {
        if (value > Int.MAX_VALUE.toULong()) throw IllegalArgumentException("GGUF $what length $value at offset ${position - 8} is too large")
        return value.toInt()
    }

    /** Fail before allocating when [count] elements of at least [minBytes] each cannot fit in the rest of the source. */
    fun checkFits(count: Int, minBytes: Int, what: String) {
        val left = source.size - position
        if (count.toLong() * minBytes > left) throw IllegalArgumentException("GGUF $what of $count elements at offset $position needs ${count.toLong() * minBytes} bytes, $left left")
    }

    // Bulk reads for the other fixed-size array elements (small arrays in practice): into a primitive array.

    fun bytes(count: Int): ByteArray {
        val out = ByteArray(count)
        var done = 0
        while (done < count) {
            if (bufLen - p == 0) require(1)
            val n = minOf(count - done, bufLen - p)
            buf.copyInto(out, done, p, p + n)
            p += n; done += n
        }
        return out
    }

    fun shorts(count: Int): ShortArray = ShortArray(count).also { out -> for (i in 0 until count) out[i] = short() }

    fun longs(count: Int): LongArray = LongArray(count).also { out -> for (i in 0 until count) out[i] = long() }

    companion object {
        private const val WINDOW = 64 * 1024
        private const val BLOCK = 2048
        const val INT32 = 0
        const val UINT32 = 1
        const val FLOAT32 = 2
    }
}
