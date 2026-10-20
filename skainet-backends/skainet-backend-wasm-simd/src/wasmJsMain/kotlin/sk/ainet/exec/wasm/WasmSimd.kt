package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.tensor.ops.WasmCpuOpsOverride
import sk.ainet.lang.memory.PlatformStorage

/**
 * Entry point of the browser kernel pack (SKEEP-007 §T1).
 *
 * ```kotlin
 * WasmSimd.install()                 // once, before the first DirectCpuExecutionContext
 * val ctx = DirectCpuExecutionContext()
 * ```
 *
 * [install] loads the SIMD128 sidecar (pthreads build when the page is cross-origin isolated),
 * registers [WasmKernelProvider], [WasmSimdKernelPack] and [WasmTernaryKernelPack], binds
 * `MemoryDomain.HOST_OFFHEAP` to the sidecar's linear memory ([WasmLinearStorage], so activations stay
 * resident between kernels) and makes every context built afterwards use [WasmCpuOps]. Without it —
 * or before it completes — the engine runs the common scalar lane, so an application may start
 * rendering before the sidecar is up and simply get faster later.
 */
public object WasmSimd {
    public val isInstalled: Boolean get() = WasmKernels.isReady && WasmCpuOpsOverride.factory != null

    /**
     * How many sidecar bytes resident activations may hold before [WasmCpuOps] returns heap outputs
     * instead (the kernels still run in the sidecar; only the result is copied back). Ambient
     * activations are freed when their tensors are collected, which is not prompt, so this is the
     * bound on what a long notebook-style session can pin in linear memory. 768 MB by default; the
     * wasm32 sidecar has 4 GB in total, shared with the resident weights.
     */
    public var residentBudgetBytes: Long = 768L * 1024 * 1024

    public suspend fun install(url: String? = null, threads: Boolean? = null) {
        WasmKernels.initialize(url, threads)
        KernelRegistry.register(WasmKernelProvider)
        // A table filled before this pack existed would keep serving scalar kernels: rebuild it so
        // the auto-install on the next dispatch picks the sidecar provider and this pack's kernels.
        KernelDispatch.clearForTesting()
        WasmSimdKernelPack.install()
        WasmTernaryKernelPack.install()
        PlatformStorage.offHeapAllocator = WasmLinearStorage.Allocator
        WasmCpuOpsOverride.factory = { factory, schedule -> WasmCpuOps(factory, schedule) }
        // The same one-line banner the JVM factory prints, so a console shows which tier runs.
        println("[SKaiNET] Using SIMD128 sidecar kernels (${WasmKernels.features()})")
    }

    /** What the installed sidecar runs on; `mode == "none"` when not installed. */
    public fun features(): WasmKernels.SidecarFeatures = WasmKernels.features()

    /** Back to the scalar lane; frees every resident weight. Tests and model switching. */
    public fun uninstall() {
        WasmCpuOpsOverride.factory = null
        PlatformStorage.offHeapAllocator = null
        KernelDispatch.clearForTesting()
        WasmKernels.clearForTesting()
    }
}
