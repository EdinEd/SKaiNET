package sk.ainet.data.source

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class Sha256DataSourceChecksumTest {

    private fun sha256(bytes: ByteArray): String = Sha256DataSourceChecksum.newSha256().apply { update(bytes) }.hex()

    @Test
    fun theStandardVectorsMatch() {
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", sha256(ByteArray(0)))
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", sha256("abc".encodeToByteArray()))
        // 56 bytes: the message that forces the padding into a second block.
        assertEquals(
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".encodeToByteArray()),
        )
        assertEquals("41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3", sha256(ByteArray(1000) { 'a'.code.toByte() }))
    }

    @Test
    fun incrementalUpdatesEqualOneShot() {
        val data = ByteArray(3000) { (it * 31 + 7).toByte() }
        val whole = sha256(data)
        for (chunk in intArrayOf(1, 7, 63, 64, 65, 500)) {
            val hash = Sha256DataSourceChecksum.newSha256()
            var i = 0
            while (i < data.size) {
                val end = minOf(i + chunk, data.size)
                hash.update(data, i, end)
                i = end
            }
            assertEquals(whole, hash.hex(), "chunk $chunk")
        }
    }

    @Test
    fun hexIsIdempotentAndEndsTheHash() {
        val hash = Sha256DataSourceChecksum.newSha256()
        hash.update("payload".encodeToByteArray())
        assertEquals("239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5", hash.hex())
        assertEquals(hash.hex(), hash.hex())
        assertFailsWith<IllegalStateException> { hash.update(byteArrayOf(1)) }
    }
}
