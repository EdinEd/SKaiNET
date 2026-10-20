package sk.ainet.data.source

import java.security.MessageDigest
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

/** The common SHA-256 against the JDK's, over lengths that cover every padding case. */
class Sha256DataSourceChecksumJvmTest {
    @Test
    fun matchesMessageDigestForEveryLengthAround64AndRandomSizes() {
        val random = Random(2026)
        val lengths = (0..130).toList() + listOf(255, 256, 1023, 4096, 65_537, 1_000_003)
        for (n in lengths) {
            val bytes = random.nextBytes(n)
            val expected = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            val hash = Sha256DataSourceChecksum.newSha256()
            // Feed in uneven pieces so block boundaries fall mid-update.
            var i = 0
            while (i < n) {
                val end = minOf(n, i + 1 + random.nextInt(97))
                hash.update(bytes, i, end)
                i = end
            }
            assertEquals(expected, hash.hex(), "length $n")
        }
    }
}
