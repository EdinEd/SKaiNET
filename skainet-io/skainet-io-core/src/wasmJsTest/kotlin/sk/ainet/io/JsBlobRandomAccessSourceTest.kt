@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class)

package sk.ainet.io

import kotlinx.coroutines.test.runTest
import org.w3c.files.Blob
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** A Blob of [n] bytes from a 32-bit LCG seeded with [seed]; [expectedByte] is the same sequence in Kotlin. */
@JsFun(
    """(n, seed) => {
        const a = new Uint8Array(n);
        let s = seed | 0;
        for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) | 0; a[i] = (s >>> 16) & 0xff; }
        return new Blob([a]);
    }""",
)
private external fun patternBlob(n: Int, seed: Int): Blob

/**
 * A `FileReaderSync` for the main thread, for the test alone: a synchronous `XMLHttpRequest` on a
 * Blob URL with the `x-user-defined` charset, the pre-`FileReaderSync` way to read bytes
 * synchronously. Lets the worker-only path of the source run under Karma, which has no worker.
 */
@JsFun(
    """() => {
        globalThis.FileReaderSync = class {
            readAsArrayBuffer(blob) {
                const url = URL.createObjectURL(blob);
                try {
                    const x = new XMLHttpRequest();
                    x.open('GET', url, false);
                    x.overrideMimeType('text/plain; charset=x-user-defined');
                    x.send(null);
                    const s = x.responseText;
                    const out = new Uint8Array(s.length);
                    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
                    return out.buffer;
                } finally {
                    URL.revokeObjectURL(url);
                }
            }
        };
    }""",
)
private external fun installFileReaderSyncPolyfill()

@JsFun("() => { delete globalThis.FileReaderSync; }")
private external fun removeFileReaderSyncPolyfill()

@JsFun("() => performance.now()")
private external fun now(): Double

/**
 * The Kotlin/Wasm Blob source (JVM ↔ wasm IO parity): `Long` positions, bulk copies through linear
 * memory, blocking reads from the preload on the main thread and from the Blob inside a worker.
 */
class JsBlobRandomAccessSourceTest {

    private val size = 9 * 1024 * 1024 + 123
    private val preload = 1024 * 1024
    private val seed = 0x5EED

    private fun expectedByte(i: Int): Byte {
        var s = seed
        // The LCG is sequential; the tests only ask for a few positions, so a rerun from the seed is fine.
        for (k in 0..i) s = s * 1103515245 + 12345
        return ((s ushr 16) and 0xFF).toByte()
    }

    private fun assertPattern(bytes: ByteArray, from: Int, position: Long, length: Int, label: String) {
        var s = seed
        var i = 0
        while (i < position) { s = s * 1103515245 + 12345; i++ }
        for (k in 0 until length) {
            s = s * 1103515245 + 12345
            val expected = ((s ushr 16) and 0xFF).toByte()
            if (bytes[from + k] != expected) {
                throw AssertionError("$label: byte at blob position ${position + k} is ${bytes[from + k]}, expected $expected")
            }
        }
    }

    @Test
    fun theSourceReportsTheBlobsSizeAndItsPreload() = runTest {
        val source = JsBlobRandomAccessSource.open(patternBlob(size, seed), preloadSize = preload)
        assertEquals(size.toLong(), source.size)
        assertEquals(preload, source.preloadedSize)
        assertFalse(source.supportsSyncReads, "Karma runs the test on the main thread, which has no FileReaderSync")

        val whole = JsBlobRandomAccessSource.open(patternBlob(4096, seed), preloadSize = preload)
        assertEquals(4096, whole.preloadedSize, "a Blob smaller than the preload is preloaded whole")
        assertEquals(0, JsBlobRandomAccessSource.open(patternBlob(4096, seed), preloadSize = 0).preloadedSize)
    }

    @Test
    fun blockingReadsServeThePreloadAndRefuseTheRestOnTheMainThread() = runTest {
        val source = JsBlobRandomAccessSource.open(patternBlob(size, seed), preloadSize = preload)

        val head = source.readAt(0L, 16)
        assertPattern(head, 0, 0L, 16, "head")
        assertEquals(expectedByte(0), head[0])

        val tail = source.readAt((preload - 8).toLong(), 8)
        assertPattern(tail, 0, (preload - 8).toLong(), 8, "end of the preload")

        val into = ByteArray(40)
        assertEquals(24, source.readAt(1000L, into, 8, 24))
        assertPattern(into, 8, 1000L, 24, "read into a buffer at an offset")
        assertEquals(0, into[0].toInt())
        assertEquals(0, into[39].toInt())

        assertEquals(0, source.readAt(5L, ByteArray(0), 0, 0))
        assertEquals(expectedByte(77), source.readByteAt(77L))

        val error = assertFailsWith<UnsupportedOperationException> { source.readAt((preload - 4).toLong(), 8) }
        assertTrue(error.message!!.contains("Web Worker"), error.message)
        assertTrue(error.message!!.contains("read()"), error.message)
        assertFailsWith<IllegalArgumentException> { source.readAt(size.toLong() - 4, 8) }
        assertFailsWith<IllegalArgumentException> { source.readAt(-1L, 8) }
        assertFailsWith<IllegalArgumentException> { source.readAt(0L, ByteArray(4), 2, 4) }
    }

    @Test
    fun suspendingReadsReachAnyRangeInBulk() = runTest {
        val source = JsBlobRandomAccessSource.open(patternBlob(size, seed), preloadSize = preload)

        // Crosses the preload boundary: served from the Blob as one range.
        val across = source.read((preload - 100).toLong(), 300)
        assertPattern(across, 0, (preload - 100).toLong(), 300, "across the preload boundary")

        // Deep in the Blob, odd length, odd position.
        val deep = source.read(5_000_001L, 777)
        assertPattern(deep, 0, 5_000_001L, 777, "deep read")

        // The last bytes.
        val last = source.read(size.toLong() - 5, 5)
        assertPattern(last, 0, size.toLong() - 5, 5, "last bytes")
        assertEquals(0, source.read(size.toLong(), 0).size)

        // Into a buffer at an offset.
        val into = ByteArray(64)
        assertEquals(32, source.read(3_333_333L, into, 16, 32))
        assertPattern(into, 16, 3_333_333L, 32, "suspending read into a buffer")

        // A read spanning two 4 MB stage boundaries, checked byte for byte.
        val big = 8 * 1024 * 1024 + 500
        val t0 = now()
        val bulk = source.read(500_000L, big)
        val ms = now() - t0
        assertEquals(big, bulk.size)
        assertPattern(bulk, 0, 500_000L, big, "bulk read")
        println("[JsBlobRandomAccessSourceTest] suspending read of ${big / (1024 * 1024)} MB: ${ms.toInt()} ms (${(big / 1048576.0 / (ms / 1000.0)).toInt()} MB/s)")

        assertFailsWith<IllegalArgumentException> { source.read(size.toLong() - 4, 8) }
    }

    @Test
    fun insideAWorkerBlockingReadsComeFromTheBlob() = runTest {
        installFileReaderSyncPolyfill()
        try {
            assertTrue(JsBlobRandomAccessSource.hasFileReaderSync())
            val source = JsBlobRandomAccessSource.open(patternBlob(size, seed), preloadSize = 0)
            assertTrue(source.supportsSyncReads)
            assertEquals(0, source.preloadedSize)

            val deep = source.readAt(5_000_000L, 100)
            assertPattern(deep, 0, 5_000_000L, 100, "blocking read from the Blob")

            // Across a 4 MB stage boundary, with the buffer form.
            val into = ByteArray(200)
            assertEquals(120, source.readAt((4 * 1024 * 1024 - 50).toLong(), into, 40, 120))
            assertPattern(into, 40, (4 * 1024 * 1024 - 50).toLong(), 120, "blocking read across a stage")

            val sync = JsBlobRandomAccessSource.openSync(patternBlob(size, seed), preloadSize = 4096)
            assertEquals(size.toLong(), sync.size)
            assertEquals(4096, sync.preloadedSize)
            assertPattern(sync.readAt(0L, 4096), 0, 0L, 4096, "openSync preload")
            assertPattern(sync.readAt(size.toLong() - 10, 10), 0, size.toLong() - 10, 10, "openSync tail")
        } finally {
            removeFileReaderSyncPolyfill()
        }
        assertFalse(JsBlobRandomAccessSource.hasFileReaderSync())
        assertFailsWith<UnsupportedOperationException> { JsBlobRandomAccessSource.openSync(patternBlob(16, seed)) }
    }
}
