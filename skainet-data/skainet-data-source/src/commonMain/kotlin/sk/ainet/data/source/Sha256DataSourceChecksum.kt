package sk.ainet.data.source

/**
 * SHA-256 in common Kotlin (FIPS 180-4), for the targets that have no platform digest to call —
 * a browser's `crypto.subtle` is asynchronous and cannot serve the incremental [DataSourceHash]
 * contract — and as the one implementation every target may share. Byte-identical to
 * `MessageDigest.getInstance("SHA-256")` (the JVM test checks it); a few MB/s slower, which is
 * irrelevant next to the download it verifies.
 */
public object Sha256DataSourceChecksum : DataSourceChecksum {
    override fun newSha256(): DataSourceHash = Sha256DataSourceHash()
}

/** One SHA-256 computation: [update] any number of times, then [hex] once. */
public class Sha256DataSourceHash : DataSourceHash {
    private val state = IntArray(8) { INITIAL[it] }
    private val block = ByteArray(64)
    private var blockLength = 0
    private var totalBytes = 0L
    private val schedule = IntArray(64)
    private var digest: String? = null

    override fun update(bytes: ByteArray, startIndex: Int, endIndex: Int) {
        require(startIndex in 0..endIndex && endIndex <= bytes.size) { "range [$startIndex, $endIndex) out of ${bytes.size}" }
        check(digest == null) { "hex() has been called; this hash is finished" }
        var i = startIndex
        totalBytes += (endIndex - startIndex).toLong()
        while (i < endIndex) {
            val n = minOf(64 - blockLength, endIndex - i)
            bytes.copyInto(block, blockLength, i, i + n)
            blockLength += n
            i += n
            if (blockLength == 64) {
                compress()
                blockLength = 0
            }
        }
    }

    override fun hex(): String {
        digest?.let { return it }
        val bitLength = totalBytes * 8
        block[blockLength++] = 0x80.toByte()
        if (blockLength > 56) {
            block.fill(0, blockLength, 64)
            compress()
            blockLength = 0
        }
        block.fill(0, blockLength, 56)
        for (i in 0 until 8) block[56 + i] = (bitLength ushr (56 - 8 * i)).toByte()
        compress()
        val out = StringBuilder(64)
        for (word in state) {
            for (shift in intArrayOf(24, 16, 8, 0)) {
                val b = (word ushr shift) and 0xFF
                out.append(HEX[b ushr 4]).append(HEX[b and 0xF])
            }
        }
        return out.toString().also { digest = it }
    }

    private fun compress() {
        val w = schedule
        for (t in 0 until 16) {
            val o = t * 4
            w[t] = ((block[o].toInt() and 0xFF) shl 24) or ((block[o + 1].toInt() and 0xFF) shl 16) or
                ((block[o + 2].toInt() and 0xFF) shl 8) or (block[o + 3].toInt() and 0xFF)
        }
        for (t in 16 until 64) {
            val w15 = w[t - 15]
            val w2 = w[t - 2]
            val s0 = w15.rotateRight(7) xor w15.rotateRight(18) xor (w15 ushr 3)
            val s1 = w2.rotateRight(17) xor w2.rotateRight(19) xor (w2 ushr 10)
            w[t] = w[t - 16] + s0 + w[t - 7] + s1
        }
        var a = state[0]; var b = state[1]; var c = state[2]; var d = state[3]
        var e = state[4]; var f = state[5]; var g = state[6]; var h = state[7]
        for (t in 0 until 64) {
            val bigS1 = e.rotateRight(6) xor e.rotateRight(11) xor e.rotateRight(25)
            val ch = (e and f) xor (e.inv() and g)
            val temp1 = h + bigS1 + ch + K[t] + w[t]
            val bigS0 = a.rotateRight(2) xor a.rotateRight(13) xor a.rotateRight(22)
            val maj = (a and b) xor (a and c) xor (b and c)
            val temp2 = bigS0 + maj
            h = g; g = f; f = e; e = d + temp1
            d = c; c = b; b = a; a = temp1 + temp2
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d
        state[4] += e; state[5] += f; state[6] += g; state[7] += h
    }

    private companion object {
        private const val HEX = "0123456789abcdef"

        private val INITIAL: IntArray = longArrayOf(
            0x6a09e667L, 0xbb67ae85L, 0x3c6ef372L, 0xa54ff53aL, 0x510e527fL, 0x9b05688cL, 0x1f83d9abL, 0x5be0cd19L,
        ).toIntArrayTruncating()

        private val K: IntArray = longArrayOf(
            0x428a2f98L, 0x71374491L, 0xb5c0fbcfL, 0xe9b5dba5L, 0x3956c25bL, 0x59f111f1L, 0x923f82a4L, 0xab1c5ed5L,
            0xd807aa98L, 0x12835b01L, 0x243185beL, 0x550c7dc3L, 0x72be5d74L, 0x80deb1feL, 0x9bdc06a7L, 0xc19bf174L,
            0xe49b69c1L, 0xefbe4786L, 0x0fc19dc6L, 0x240ca1ccL, 0x2de92c6fL, 0x4a7484aaL, 0x5cb0a9dcL, 0x76f988daL,
            0x983e5152L, 0xa831c66dL, 0xb00327c8L, 0xbf597fc7L, 0xc6e00bf3L, 0xd5a79147L, 0x06ca6351L, 0x14292967L,
            0x27b70a85L, 0x2e1b2138L, 0x4d2c6dfcL, 0x53380d13L, 0x650a7354L, 0x766a0abbL, 0x81c2c92eL, 0x92722c85L,
            0xa2bfe8a1L, 0xa81a664bL, 0xc24b8b70L, 0xc76c51a3L, 0xd192e819L, 0xd6990624L, 0xf40e3585L, 0x106aa070L,
            0x19a4c116L, 0x1e376c08L, 0x2748774cL, 0x34b0bcb5L, 0x391c0cb3L, 0x4ed8aa4aL, 0x5b9cca4fL, 0x682e6ff3L,
            0x748f82eeL, 0x78a5636fL, 0x84c87814L, 0x8cc70208L, 0x90befffaL, 0xa4506cebL, 0xbef9a3f7L, 0xc67178f2L,
        ).toIntArrayTruncating()

        private fun LongArray.toIntArrayTruncating(): IntArray = IntArray(size) { this[it].toInt() }
    }
}
