package sk.ainet.io

import kotlinx.coroutines.await
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Int8Array
import org.w3c.files.Blob
import kotlin.js.Promise

/**
 * JavaScript/Browser implementation of [RandomAccessSource] using Blob API.
 *
 * Uses Blob.slice() for efficient random access to file data without loading
 * the entire file into memory.
 *
 * **Important**: Browser Blob operations are inherently asynchronous.
 * This implementation pre-loads metadata section during [open] for synchronous
 * access via [readAt], and provides [readAtAsync] for accessing data beyond
 * the preloaded buffer.
 *
 * Usage:
 * ```kotlin
 * // From file input
 * val file: File = document.getElementById("fileInput").files[0]
 * val source = JsBlobRandomAccessSource.open(file)
 *
 * // Sync access for metadata (within preloaded buffer)
 * val header = source.readAt(0, 24)
 *
 * // Async access for tensor data
 * val tensorData = source.readAtAsync(tensorOffset, tensorSize)
 * ```
 */
public class JsBlobRandomAccessSource private constructor(
    private val blob: Blob,
    private val preloadedBuffer: ByteArray
) : RandomAccessSource, SuspendingRandomAccessSource {

    override val size: Long = blob.size.toLong()

    /**
     * The size of the preloaded buffer in bytes.
     * Reads within this range are synchronous.
     */
    public val preloadedSize: Int = preloadedBuffer.size

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }

        if (length == 0) return ByteArray(0)

        // Within the preloaded window: served from the heap. Compared as Long — a position past
        // 2 GB must not be truncated into the window.
        if (position + length <= preloadedBuffer.size) {
            val pos = position.toInt()
            return preloadedBuffer.copyOfRange(pos, pos + length)
        }

        // Beyond preloaded data - throw helpful error
        throw UnsupportedOperationException(
            "Synchronous read beyond preloaded buffer (${preloadedBuffer.size} bytes). " +
            "Position: $position, Length: $length. " +
            "Use readAtAsync() for tensor data loading, or increase preloadSize when opening."
        )
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }

        if (length == 0) return 0

        val data = readAt(position, length)
        data.copyInto(buffer, offset, 0, data.size)
        return data.size
    }

    override fun close() {
        // Blob doesn't need explicit closing
    }

    /**
     * Asynchronously read bytes from any position.
     *
     * Use this for loading tensor data that may be beyond the preloaded buffer.
     * If the requested range is within the preloaded buffer, returns from cache.
     *
     * @param position Starting byte offset (0-indexed)
     * @param length Number of bytes to read
     * @return ByteArray of the requested data
     */
    /**
     * The suspending read (#1037): unlike [readAt], this is not limited to the preloaded window —
     * a range outside it is fetched from the blob instead of failing.
     */
    override suspend fun read(position: Long, length: Int): ByteArray = readAtAsync(position, length)

    /** Suspending read into [buffer]; see [readAt]. */
    override suspend fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        val bytes = readAtAsync(position, length)
        bytes.copyInto(buffer, offset)
        return bytes.size
    }

    public suspend fun readAtAsync(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }

        if (length == 0) return ByteArray(0)

        // Within the preloaded window: served from the heap (fast path), compared as Long.
        if (position + length <= preloadedBuffer.size) {
            val pos = position.toInt()
            return preloadedBuffer.copyOfRange(pos, pos + length)
        }

        // Beyond it: slice + arrayBuffer. `Blob.slice` takes JavaScript numbers (53 bits), so a Blob
        // over 2 GB is addressed as a Double, not through the stdlib's Int overload.
        val slice = blob.asDynamic().slice(position.toDouble(), (position + length).toDouble()) as Blob
        val arrayBuffer = (slice.asDynamic().arrayBuffer() as Promise<ArrayBuffer>).await()
        return arrayBufferToByteArray(arrayBuffer)
    }

    public companion object {
        /**
         * Default preload size: 50MB should cover metadata for most models.
         * ONNX metadata is typically 1-10MB.
         */
        public const val DEFAULT_PRELOAD_SIZE: Int = 50 * 1024 * 1024

        /**
         * Open a Blob for streaming random access.
         *
         * Pre-loads the first [preloadSize] bytes for synchronous metadata parsing.
         * Use [readAtAsync] for accessing data beyond the preloaded buffer.
         *
         * @param blob The Blob or File to read from
         * @param preloadSize How much to preload for sync access (default 50MB)
         * @return JsBlobRandomAccessSource ready for use
         */
        public suspend fun open(blob: Blob, preloadSize: Int = DEFAULT_PRELOAD_SIZE): JsBlobRandomAccessSource {
            val actualPreloadSize = minOf(preloadSize.toLong(), blob.size.toLong()).toInt()
            val slice = blob.slice(0, actualPreloadSize)
            val arrayBuffer = (slice.asDynamic().arrayBuffer() as Promise<ArrayBuffer>).await()
            val bytes = arrayBufferToByteArray(arrayBuffer)
            return JsBlobRandomAccessSource(blob, bytes)
        }

        /** A Kotlin/JS `ByteArray` *is* an `Int8Array`: no copy, no per-element loop. */
        private fun arrayBufferToByteArray(buffer: ArrayBuffer): ByteArray =
            Int8Array(buffer).unsafeCast<ByteArray>()
    }
}
