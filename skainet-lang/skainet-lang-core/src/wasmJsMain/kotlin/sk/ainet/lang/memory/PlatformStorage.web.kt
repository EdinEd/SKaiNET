package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * A binding of [MemoryDomain.HOST_OFFHEAP] for Kotlin/Wasm, installed by a backend that owns
 * memory outside the Kotlin heap — the SIMD128 kernel sidecar's linear memory
 * (`skainet-backend-wasm-simd`, `WasmSimd.install()`), where activations can stay resident between
 * kernels instead of crossing the boundary per op. This module cannot depend on that backend, so
 * the binding is a hook, exactly like the ops override in `skainet-backend-cpu`.
 */
public fun interface OffHeapAllocator {
    /** Allocate [bytes] zeroed bytes off the Kotlin heap, owned by a scope of kind [scope]. */
    public fun allocate(bytes: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage

    /**
     * Bytes this allocator's memory currently holds — what [MemoryProbe.rssBytes] adds for the
     * off-heap domain, since a browser has no process counter that would see them. `null` when the
     * backend cannot say.
     */
    public fun heldBytes(): Long? = null
}

/**
 * JS / Wasm: the Kotlin heap, plus whatever [offHeapAllocator] a backend installs for
 * [MemoryDomain.HOST_OFFHEAP] (SKEEP-003 §4.8.4, SKEEP-007). With no allocator installed off-heap
 * requests resolve to the heap (the planner records the fallback); with one they go off the Kotlin
 * heap into the backend's memory and [supports] says so, so the planner can place activations
 * there. Files arrive through fetch/range requests into heap slabs, so [mapFile] is unsupported.
 */
public actual object PlatformStorage {
    /**
     * The off-heap binding, or `null` for the heap fallback. Set by `WasmSimd.install()`, cleared
     * by its `uninstall()`; a consumer with its own off-heap memory may set it as well.
     */
    public var offHeapAllocator: OffHeapAllocator? = null

    public actual fun supports(domain: MemoryDomain): Boolean =
        domain == MemoryDomain.HOST_HEAP || (domain == MemoryDomain.HOST_OFFHEAP && offHeapAllocator != null)
    public actual val supportsMappedFiles: Boolean get() = false

    public actual fun allocate(bytes: Long, domain: MemoryDomain, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage {
        require(bytes >= 0) { "bytes must be >= 0" }
        when (domain) {
            MemoryDomain.MMAP_FILE, MemoryDomain.DEVICE_LOCAL -> throw IllegalArgumentException("$domain is not allocatable on this target")
            MemoryDomain.HOST_OFFHEAP, MemoryDomain.HOST_PINNED, MemoryDomain.UNIFIED -> {
                offHeapAllocator?.let { return it.allocate(bytes, scope, origin, sink) }
            }
            MemoryDomain.HOST_HEAP -> Unit
        }
        require(bytes <= Int.MAX_VALUE.toLong()) { "heap storage is limited to 2 GB per array, requested $bytes bytes" }
        return Storage.Heap.bytes(bytes.toInt(), scope, origin, sink)
    }

    public actual fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
        throw UnsupportedOperationException("memory-mapped files are not available on this target (use the suspend RandomAccessSource into heap slabs)")

    public val info: PlatformStorageInfo
        get() = PlatformStorageInfo(if (offHeapAllocator != null) "backend linear memory (installed allocator)" else "heap (fallback)", "none")
}
