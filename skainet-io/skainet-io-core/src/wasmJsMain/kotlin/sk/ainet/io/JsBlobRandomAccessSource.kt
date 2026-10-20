@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class, kotlin.wasm.unsafe.UnsafeWasmMemoryApi::class)

package sk.ainet.io

import kotlinx.coroutines.await
import org.khronos.webgl.ArrayBuffer
import org.w3c.files.Blob
import kotlin.js.Promise
import kotlin.wasm.unsafe.Pointer
import kotlin.wasm.unsafe.wasmMemory
import kotlin.wasm.unsafe.withScopedMemoryAllocator

/**
 * [RandomAccessSource] over a browser `Blob` — a `File` from an `<input>`, a fetched body, a Cache
 * API entry — for Kotlin/Wasm.
 *
 * Positions and [size] are `Long`: a Blob may be larger than 2 GB (one shard of a sharded checkpoint
 * routinely is), and `Blob.slice` takes JavaScript numbers, which carry 53 bits. Only a single read
 * is bounded by the 2 GB `ByteArray` it lands in.
 *
 * Bytes cross into Kotlin through wasm linear memory — one `Uint8Array.set` per 4 MB stage, then a
 * wasm loop into the `ByteArray` — never one JavaScript call per byte. (Kotlin/Wasm has no direct
 * `Int8Array` → `ByteArray` adapter; the per-byte loop this replaces was fifty million interop calls
 * for the default 50 MB preload.)
 *
 * A Blob is asynchronous on the main thread, so there are two ways to read it:
 *
 *  - [read] (suspending, [SuspendingRandomAccessSource]) works on any thread and for any range.
 *  - [readAt] (blocking, [RandomAccessSource]) is served from the preloaded window — the first
 *    [preloadedSize] bytes, enough for a header — and, **inside a Web Worker**, from the Blob itself
 *    through `FileReaderSync`, the browser's only synchronous file read, which exists on worker
 *    threads alone. [supportsSyncReads] says which case this source is in; a blocking read beyond the
 *    window on the main thread throws with that explanation rather than blocking the page.
 *
 * The blocking readers of the format modules (`StreamingSafeTensorsReader`, `StreamingGGUFReader`)
 * therefore load a whole model from a Blob when the load runs in a worker — the same code path the
 * JVM takes over a file — and parse headers from the preload when it runs on the main thread.
 */
public class JsBlobRandomAccessSource private constructor(
    private val blob: Blob,
    private val preloadedBuffer: ByteArray,
    override val size: Long,
    /**
     * Whether blocking reads reach the whole Blob: `true` on a worker thread (`FileReaderSync` is
     * defined), `false` on the main thread, where only the preloaded window is readable blockingly.
     */
    public val supportsSyncReads: Boolean,
) : RandomAccessSource, SuspendingRandomAccessSource {

    /** The size of the preloaded window in bytes; blocking reads inside it never touch the Blob. */
    public val preloadedSize: Int = preloadedBuffer.size

    override fun readAt(position: Long, length: Int): ByteArray {
        checkRange(position, length)
        val out = ByteArray(length)
        readAt(position, out, 0, length)
        return out
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        checkRange(position, length)
        checkBuffer(buffer, offset, length)
        if (length == 0) return 0
        if (position + length <= preloadedSize) {
            preloadedBuffer.copyInto(buffer, offset, position.toInt(), position.toInt() + length)
            return length
        }
        if (!supportsSyncReads) {
            throw UnsupportedOperationException(
                "Blocking read beyond the preloaded $preloadedSize bytes (position=$position, length=$length) on a " +
                    "thread without FileReaderSync — the browser's main thread. Use the suspending read(), raise " +
                    "preloadSize when opening, or run the load in a Web Worker, where Blob reads are synchronous.",
            )
        }
        readSync(position, buffer, offset, length)
        return length
    }

    override fun close() {
        // A Blob has no handle to release.
    }

    /** The suspending read: any range, any thread (#1037). */
    override suspend fun read(position: Long, length: Int): ByteArray {
        checkRange(position, length)
        val out = ByteArray(length)
        read(position, out, 0, length)
        return out
    }

    /** Suspending read into [buffer]; see [readAt]. */
    override suspend fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        checkRange(position, length)
        checkBuffer(buffer, offset, length)
        if (length == 0) return 0
        if (position + length <= preloadedSize) {
            preloadedBuffer.copyInto(buffer, offset, position.toInt(), position.toInt() + length)
            return length
        }
        val slice = sliceBlob(blob, position.toDouble(), (position + length).toDouble())
        val arrayBuffer = blobArrayBuffer(slice).await<ArrayBuffer>()
        copyArrayBuffer(arrayBuffer, buffer, offset, length)
        return length
    }

    /** [read] under its original name. */
    public suspend fun readAtAsync(position: Long, length: Int): ByteArray = read(position, length)

    private fun checkRange(position: Long, length: Int) {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) { "Read beyond end of file: position=$position, length=$length, size=$size" }
    }

    private fun checkBuffer(buffer: ByteArray, offset: Int, length: Int) {
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(offset + length <= buffer.size) { "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}" }
    }

    /** `FileReaderSync` over the Blob, staged through linear memory. Worker threads only. */
    private fun readSync(position: Long, dest: ByteArray, destOffset: Int, length: Int) {
        var done = 0
        while (done < length) {
            val n = minOf(length - done, STAGE_BYTES)
            withScopedMemoryAllocator { allocator ->
                val base = allocator.allocate(n)
                jsReadBlobSyncInto(blob, (position + done).toDouble(), n, wasmMemory, base.address.toInt())
                unstage(base, dest, destOffset + done, n)
            }
            done += n
        }
    }

    public companion object {
        /** Default preload: 50 MB covers the header of every model format this project reads. */
        public const val DEFAULT_PRELOAD_SIZE: Int = 50 * 1024 * 1024

        private const val STAGE_BYTES = 4 * 1024 * 1024

        /**
         * Open [blob] for random access, preloading its first [preloadSize] bytes (or all of it when
         * smaller) for blocking reads. Suspends for the preload; works on any thread.
         */
        public suspend fun open(blob: Blob, preloadSize: Int = DEFAULT_PRELOAD_SIZE): JsBlobRandomAccessSource {
            require(preloadSize >= 0) { "preloadSize must be non-negative: $preloadSize" }
            val blobSize = getBlobSize(blob).toLong()
            val preload = minOf(preloadSize.toLong(), blobSize).toInt()
            val window = ByteArray(preload)
            if (preload > 0) {
                val slice = sliceBlob(blob, 0.0, preload.toDouble())
                val arrayBuffer = blobArrayBuffer(slice).await<ArrayBuffer>()
                copyArrayBuffer(arrayBuffer, window, 0, preload)
            }
            return JsBlobRandomAccessSource(blob, window, blobSize, hasFileReaderSync())
        }

        /**
         * Open [blob] without suspending — for code that runs inside a Web Worker and is not in a
         * coroutine. Every read is blocking there, so nothing is preloaded unless [preloadSize] says
         * so. Throws on a thread without `FileReaderSync` (the main thread); use [open] there.
         */
        public fun openSync(blob: Blob, preloadSize: Int = 0): JsBlobRandomAccessSource {
            require(preloadSize >= 0) { "preloadSize must be non-negative: $preloadSize" }
            if (!hasFileReaderSync()) {
                throw UnsupportedOperationException(
                    "openSync needs FileReaderSync, which exists only on Web Worker threads; on the main thread use the suspending open()",
                )
            }
            val blobSize = getBlobSize(blob).toLong()
            val preload = minOf(preloadSize.toLong(), blobSize).toInt()
            val source = JsBlobRandomAccessSource(blob, ByteArray(0), blobSize, true)
            if (preload == 0) return source
            val window = ByteArray(preload)
            source.readSync(0L, window, 0, preload)
            return JsBlobRandomAccessSource(blob, window, blobSize, true)
        }

        /** Whether this thread can read a Blob synchronously (a Web Worker). */
        public fun hasFileReaderSync(): Boolean = jsHasFileReaderSync()

        /** Copy [length] bytes of [buffer] into [dest] at [destOffset], 4 MB stages through linear memory. */
        private fun copyArrayBuffer(buffer: ArrayBuffer, dest: ByteArray, destOffset: Int, length: Int) {
            require(getArrayBufferByteLength(buffer) >= length) {
                "the Blob delivered ${getArrayBufferByteLength(buffer)} bytes, $length were requested"
            }
            var done = 0
            while (done < length) {
                val n = minOf(length - done, STAGE_BYTES)
                withScopedMemoryAllocator { allocator ->
                    val base = allocator.allocate(n)
                    jsCopyBufferInto(buffer, done, n, wasmMemory, base.address.toInt())
                    unstage(base, dest, destOffset + done, n)
                }
                done += n
            }
        }

        /** Linear memory at [base] → [dest]: four bytes per load, the tail byte by byte. */
        private fun unstage(base: Pointer, dest: ByteArray, destOffset: Int, n: Int) {
            var p: Pointer = base
            var i = 0
            while (i + 4 <= n) {
                val v = p.loadInt()
                val o = destOffset + i
                dest[o] = (v and 0xFF).toByte()
                dest[o + 1] = ((v ushr 8) and 0xFF).toByte()
                dest[o + 2] = ((v ushr 16) and 0xFF).toByte()
                dest[o + 3] = ((v ushr 24) and 0xFF).toByte()
                p += 4
                i += 4
            }
            while (i < n) {
                dest[destOffset + i] = p.loadByte()
                p += 1
                i++
            }
        }
    }
}

@JsFun("(blob) => blob.size")
private external fun getBlobSize(blob: Blob): Double

@JsFun("(blob) => blob.arrayBuffer()")
private external fun blobArrayBuffer(blob: Blob): Promise<ArrayBuffer>

@JsFun("(buf) => buf.byteLength")
private external fun getArrayBufferByteLength(buf: ArrayBuffer): Int

@JsFun("(blob, start, end) => blob.slice(start, end)")
private external fun sliceBlob(blob: Blob, start: Double, end: Double): Blob

@JsFun("() => typeof FileReaderSync !== 'undefined'")
private external fun jsHasFileReaderSync(): Boolean

/** `buf[srcOffset, srcOffset + length)` → Kotlin linear memory at [ptr]. */
@JsFun("(buf, srcOffset, length, kmem, ptr) => new Uint8Array(kmem.buffer).set(new Uint8Array(buf, srcOffset, length), ptr)")
private external fun jsCopyBufferInto(buf: ArrayBuffer, srcOffset: Int, length: Int, kmem: JsAny, ptr: Int)

/** `blob[start, start + length)` through `FileReaderSync` → Kotlin linear memory at [ptr]. Worker threads only. */
@JsFun("(blob, start, length, kmem, ptr) => new Uint8Array(kmem.buffer).set(new Uint8Array(new FileReaderSync().readAsArrayBuffer(blob.slice(start, start + length))), ptr)")
private external fun jsReadBlobSyncInto(blob: Blob, start: Double, length: Int, kmem: JsAny, ptr: Int)
