package sk.ainet.io

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** SKEEP-007 §T0.4: the engine's in-memory [RandomAccessSource], single array and chunked. */
class ByteArrayRandomAccessSourceTest {
    private val bytes = ByteArray(100) { it.toByte() }

    @Test
    fun `single array reads any window`() {
        val src = ByteArrayRandomAccessSource(bytes)
        assertEquals(100L, src.size)
        assertContentEquals(bytes.copyOfRange(10, 25), src.readAt(10, 15))
        assertContentEquals(ByteArray(0), src.readAt(100, 0))
        assertEquals(42.toByte(), src.readByteAt(42))
        val buf = ByteArray(10) { 0x7F }
        assertEquals(4, src.readAt(96, buf, 2, 8), "short read at the end")
        assertContentEquals(byteArrayOf(0x7F, 0x7F, 96, 97, 98, 99, 0x7F, 0x7F, 0x7F, 0x7F), buf)
        assertEquals(0, src.readAt(100, buf))
    }

    @Test
    fun `chunked source reads across chunk boundaries`() {
        val chunks = listOf(bytes.copyOfRange(0, 32), bytes.copyOfRange(32, 64), bytes.copyOfRange(64, 96), bytes.copyOfRange(96, 100))
        val src = ByteArrayRandomAccessSource(chunks)
        assertEquals(100L, src.size)
        assertContentEquals(bytes.copyOfRange(30, 70), src.readAt(30, 40))
        assertContentEquals(bytes.copyOfRange(63, 65), src.readAt(63, 2))
        assertContentEquals(bytes.copyOfRange(95, 100), src.readAt(95, 5))
        assertEquals(64.toByte(), src.readByteAt(64))
        val buf = ByteArray(12)
        assertEquals(8, src.readAt(92, buf, 0, 12))
        assertContentEquals(bytes.copyOfRange(92, 100), buf.copyOfRange(0, 8))
    }

    @Test
    fun `bounds and chunk geometry are validated`() {
        val src = ByteArrayRandomAccessSource(bytes)
        assertFailsWith<IllegalArgumentException> { src.readAt(90, 11) }
        assertFailsWith<IllegalArgumentException> { src.readAt(-1, 1) }
        assertFailsWith<IllegalArgumentException> { src.readByteAt(100) }
        assertFailsWith<IllegalArgumentException> { ByteArrayRandomAccessSource(emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            ByteArrayRandomAccessSource(listOf(ByteArray(10), ByteArray(5), ByteArray(3)))
        }
    }
}
