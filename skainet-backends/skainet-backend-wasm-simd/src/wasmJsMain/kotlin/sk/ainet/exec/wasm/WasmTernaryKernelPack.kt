package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.BitNetGemvNative
import sk.ainet.backend.api.kernel.TernaryF32GemvNative
import sk.ainet.backend.api.kernel.TernaryF32KernelPack
import sk.ainet.backend.api.kernel.TernaryKernelPacks
import sk.ainet.backend.api.kernel.TernaryLmheadNative
import sk.ainet.backend.api.kernel.TernaryPlanesKernelPack
import sk.ainet.backend.api.kernel.ViewKernelPack

/**
 * The ternary kernels of the browser pack — the wasm sibling of the JVM `FfmTernaryKernelPackFactory`
 * and the Android JNI ternary pack. Three native faces over the sidecar's exports:
 *
 * - [WasmTernaryF32Gemv] — `skainet_ternary_f32_gemv`, exact FP32 activations × `BITNET_B1_58`;
 * - [WasmTernaryLmhead] — `skainet_ternary_lmhead_stage1`, the fused four-plane `BITNET_PLANES` lm_head;
 * - [WasmBitNetGemv] — `skainet_bitnet_gemv_tq2_0`, int8 activations × `TQ2_0`.
 *
 * Each hands its weight to the sidecar **once** ([WasmKernels.weightHandle]) and stages the
 * activation and output per call, like every other kernel in this module. Without this pack a
 * BitNet weight in a browser went to the decoding reference kernel — correct, and hours per
 * projection at model size. Installed by [WasmSimd.install] after the sidecar is up; there is no
 * `ServiceLoader` on wasm.
 */
public object WasmTernaryKernelPack : ViewKernelPack {
    override val name: String = "wasm-simd128-ternary"

    override fun install() {
        if (!WasmKernels.isReady) return
        TernaryF32KernelPack.install(WasmTernaryF32Gemv)
        TernaryPlanesKernelPack.install(WasmTernaryLmhead)
        TernaryKernelPacks.install(WasmBitNetGemv)
    }
}

/** `skainet_ternary_f32_gemv` in the sidecar: the sequential b1.58 payload, scale applied by the pack. */
public object WasmTernaryF32Gemv : TernaryF32GemvNative {
    override val name: String get() = "wasm-simd128"

    override fun gemvPacked(
        activation: FloatArray, activationOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        require(inputDim % 4 == 0) { "WasmTernaryF32Gemv: inputDim must be a multiple of 4; got $inputDim" }
        if (outputDim == 0) return
        if (inputDim == 0) {
            out.fill(0f, outOffset, outOffset + outputDim)
            return
        }
        val weightPtr = WasmKernels.weightHandle(weight, 0, weight.size) + weightByteOffset
        val inPtr = WasmKernels.alloc(inputDim * 4)
        val outPtr = WasmKernels.alloc(outputDim * 4)
        try {
            WasmKernels.writeFloats(inPtr, activation, activationOffset, inputDim)
            WasmKernels.ternaryF32Gemv(inPtr, weightPtr, inputDim, outputDim, outPtr)
            WasmKernels.readFloats(outPtr, out, outOffset, outputDim)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }
}

/** `skainet_ternary_lmhead_stage1` in the sidecar: planes and row scales read from the resident weight. */
public object WasmTernaryLmhead : TernaryLmheadNative {
    override val name: String get() = "wasm-simd128"

    override fun lmheadStage1(
        activation: FloatArray, activationOffset: Int,
        weight: ByteArray, planesByteOffset: Int,
        planeStrideBytes: Int, rowScaleByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        require(inputDim % 4 == 0) { "WasmTernaryLmhead: inputDim must be a multiple of 4; got $inputDim" }
        require(rowScaleByteOffset % 2 == 0) { "WasmTernaryLmhead: rowScaleByteOffset must be 2-byte aligned; got $rowScaleByteOffset" }
        if (outputDim == 0) return
        val weightPtr = WasmKernels.weightHandle(weight, 0, weight.size)
        val inPtr = WasmKernels.alloc(maxOf(inputDim, 1) * 4)
        val outPtr = WasmKernels.alloc(outputDim * 4)
        try {
            if (inputDim > 0) WasmKernels.writeFloats(inPtr, activation, activationOffset, inputDim)
            WasmKernels.ternaryLmheadStage1(
                inPtr, weightPtr + planesByteOffset, planeStrideBytes, weightPtr + rowScaleByteOffset,
                inputDim, outputDim, outPtr,
            )
            WasmKernels.readFloats(outPtr, out, outOffset, outputDim)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }
}

/** `skainet_bitnet_gemv_tq2_0` in the sidecar: one token's int8 codes against resident TQ2_0 blocks. */
public object WasmBitNetGemv : BitNetGemvNative {
    override val name: String get() = "wasm-simd128"

    override fun gemvTq2_0(
        activation: ByteArray, activationOffset: Int, activationScale: Float,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        if (outputDim == 0) return
        if (inputDim == 0) {
            out.fill(0f, outOffset, outOffset + outputDim)
            return
        }
        val weightPtr = WasmKernels.weightHandle(weight, 0, weight.size) + weightByteOffset
        val actPtr = WasmKernels.alloc(inputDim)
        val outPtr = WasmKernels.alloc(outputDim * 4)
        try {
            WasmKernels.writeBytes(actPtr, activation, activationOffset, inputDim)
            WasmKernels.bitnetGemvTq2_0(actPtr, activationScale, weightPtr, inputDim, outputDim, outPtr)
            WasmKernels.readFloats(outPtr, out, outOffset, outputDim)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(actPtr)
        }
    }
}
