package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.ViewKernelPack
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The browser kernel pack (SKEEP-007 §T1): one [WasmRowMajorMatmulKernel] per packed format, keyed
 * exactly like the JVM FFM and Android JNI packs, so `DefaultCpuOpsBase.matmulWeightTransposed`
 * finds an exact kernel for the canonical row-major view and never relays out or decodes.
 *
 * There is no `ServiceLoader` on wasm: [WasmSimd.install] calls [install] after the sidecar is up.
 */
public object WasmSimdKernelPack : ViewKernelPack {
    override val name: String = "wasm-simd128"

    /** Formats the sidecar exports a `_rm` kernel for. */
    public val formats: Map<TensorEncoding, String> = linkedMapOf(
        TensorEncoding.Q4_K to "_skainet_q4k_matmul_rm",
        TensorEncoding.Q5_K to "_skainet_q5k_matmul_rm",
        TensorEncoding.Q6_K to "_skainet_q6k_matmul_rm",
        TensorEncoding.Q8_0 to "_skainet_q8_0_matmul_rm",
        TensorEncoding.Q4_0 to "_skainet_q4_0_matmul_rm",
        TensorEncoding.Q5_0 to "_skainet_q5_0_matmul_rm",
        TensorEncoding.Q5_1 to "_skainet_q5_1_matmul_rm",
    )

    override fun install() {
        for ((encoding, fn) in formats) {
            KernelDispatch.register(WasmRowMajorMatmulKernel(encoding, fn))
        }
    }
}
