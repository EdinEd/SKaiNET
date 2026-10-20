@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class, kotlin.wasm.unsafe.UnsafeWasmMemoryApi::class)

package sk.ainet.lang.memory

import kotlin.wasm.unsafe.wasmMemory

/**
 * Kotlin/Wasm: what the page holds for this module. There is no process to ask — a tab has no
 * `/proc` — so the resident set is assembled from the three places SKaiNET's bytes live:
 *
 *  - the JavaScript heap, where Kotlin/Wasm allocates every object (a `FloatArray` is a GC array,
 *    not linear memory): `performance.memory.usedJSHeapSize`, which Chromium exposes and other
 *    engines do not — [jsHeapUsedBytes] is `null` there;
 *  - this module's own linear memory, the staging area for interop copies ([linearMemoryBytes]);
 *  - the memory an installed [OffHeapAllocator] holds — the kernel sidecar's linear memory, where
 *    resident weights and activations live — through [OffHeapAllocator.heldBytes] ([offHeapBytes]).
 *
 * The sum is what a tab is charged for this model. Page faults are not observable in a browser,
 * so the fault counters stay `null`, as the common contract allows.
 */
public actual object MemoryProbe {

    /** JS heap in use (when the engine reports it) + this module's linear memory + off-heap bytes held. */
    public actual fun rssBytes(): Long? =
        linearMemoryBytes() + (jsHeapUsedBytes() ?: 0L) + (offHeapBytes() ?: 0L)

    public actual fun majorFaults(): Long? = null
    public actual fun minorFaults(): Long? = null

    /** Bytes of this Kotlin/Wasm module's linear memory (interop staging; grows, never shrinks). */
    public fun linearMemoryBytes(): Long = jsByteLength(wasmMemory).toLong()

    /** `performance.memory.usedJSHeapSize`, or `null` where the engine does not expose it. */
    public fun jsHeapUsedBytes(): Long? = jsHeapUsed().takeIf { it >= 0 }?.toLong()

    /** `performance.memory.jsHeapSizeLimit` — what the engine will let the heap grow to — or `null`. */
    public fun jsHeapLimitBytes(): Long? = jsHeapLimit().takeIf { it >= 0 }?.toLong()

    /** Bytes held by the installed [PlatformStorage.offHeapAllocator], or `null` without one (or one that cannot say). */
    public fun offHeapBytes(): Long? = PlatformStorage.offHeapAllocator?.heldBytes()
}

@JsFun("(kmem) => kmem.buffer.byteLength")
private external fun jsByteLength(kmem: JsAny): Double

@JsFun("() => (typeof performance !== 'undefined' && performance.memory) ? performance.memory.usedJSHeapSize : -1")
private external fun jsHeapUsed(): Double

@JsFun("() => (typeof performance !== 'undefined' && performance.memory) ? performance.memory.jsHeapSizeLimit : -1")
private external fun jsHeapLimit(): Double
