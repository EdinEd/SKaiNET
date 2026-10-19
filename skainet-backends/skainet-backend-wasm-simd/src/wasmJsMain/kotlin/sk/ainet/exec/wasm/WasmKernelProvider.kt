package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulWtKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5KMatmulKernel
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_1MatmulKernel
import sk.ainet.backend.api.kernel.Q6KMatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel

/**
 * The sidecar as a [KernelProvider] (SKEEP-007 §T1): the dense FP32 GEMM and the feed-order packed
 * kernels, so `KernelPacks.install()` and `chooseQuantizedMatmulHeap` serve from SIMD128 as well.
 * Priority 50 — above the scalar provider (0), the same tier the JVM's Panama provider sits at.
 */
public object WasmKernelProvider : KernelProvider {
    override val name: String = "wasm-simd128"
    override val priority: Int = 50
    override fun isAvailable(): Boolean = WasmKernels.isReady

    override fun matmulFp32(): Fp32MatmulKernel = WasmFp32MatmulKernel
    override fun matmulFp32Wt(): Fp32MatmulWtKernel = WasmFp32MatmulWtKernel
    override fun matmulBf16(): Bf16MatmulKernel = WasmBf16MatmulKernel
    override fun matmulFp16(): Fp16MatmulKernel = WasmFp16MatmulKernel

    override fun matmulQ8_0(): Q8_0MatmulKernel = object : Q8_0MatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q8_0_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ4_0(): Q4_0MatmulKernel = object : Q4_0MatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q4_0_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ4K(): Q4KMatmulKernel = object : Q4KMatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q4k_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ5K(): Q5KMatmulKernel = object : Q5KMatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q5k_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ6K(): Q6KMatmulKernel = object : Q6KMatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q6k_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ5_0(): Q5_0MatmulKernel = object : Q5_0MatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q5_0_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }
    override fun matmulQ5_1(): Q5_1MatmulKernel = object : Q5_1MatmulKernel {
        override fun matmul(input: FloatArray, inputOffset: Int, weight: ByteArray, weightByteOffset: Int, inputDim: Int, outputDim: Int, output: FloatArray, outputOffset: Int) =
            feedOrder("_skainet_q5_1_matmul", input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
    }

    /** One activation row against a feed-order packed weight; the weight is resident after the first call. */
    private fun feedOrder(
        fn: String,
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        val weightPtr = WasmKernels.weightHandle(weight, weightByteOffset, weight.size - weightByteOffset)
        val inPtr = WasmKernels.alloc(inputDim * 4)
        val outPtr = WasmKernels.alloc(outputDim * 4)
        try {
            WasmKernels.writeFloats(inPtr, input, inputOffset, inputDim)
            WasmKernels.packedMatmul(fn, inPtr, weightPtr, inputDim, outputDim, outPtr)
            WasmKernels.readFloats(outPtr, output, outputOffset, outputDim)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }
}

/**
 * `C(m,n) = A(m,k) · B(k,n)` with `B` as 16-bit floats (`skainet_bf16_matmul` / `skainet_fp16_matmul`).
 * A contiguous `B` (`bByteStride == 2 * n`, whole array) is a weight and stays resident after the
 * first call; a strided or windowed `B` is staged per call, as the activation always is.
 */
internal class WasmNarrowFloatMatmulKernel(private val fn: String) {
    fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) { "m, n, k must be non-negative; got m=$m n=$n k=$k" }
        if (m == 0 || n == 0) return
        val rowBytes = n * 2
        val resident = bByteStride == rowBytes && bByteOffset == 0 && b.size == k * rowBytes
        val aPtr = WasmKernels.alloc(m * k * 4)
        val bPtr = if (resident) WasmKernels.weightHandle(b, 0, b.size) else WasmKernels.alloc(k * rowBytes)
        val cPtr = WasmKernels.alloc(m * n * 4)
        try {
            if (aStride == k) WasmKernels.writeFloats(aPtr, a, aOffset, m * k)
            else for (i in 0 until m) WasmKernels.writeFloats(aPtr + i * k * 4, a, aOffset + i * aStride, k)
            if (!resident) {
                if (bByteStride == rowBytes) WasmKernels.writeBytes(bPtr, b, bByteOffset, k * rowBytes)
                else for (p in 0 until k) WasmKernels.writeBytes(bPtr + p * rowBytes, b, bByteOffset + p * bByteStride, rowBytes)
            }
            WasmKernels.narrowFloatMatmul(fn, aPtr, k, bPtr, rowBytes, cPtr, n, m, n, k)
            if (outStride == n) WasmKernels.readFloats(cPtr, out, outOffset, m * n)
            else for (i in 0 until m) WasmKernels.readFloats(cPtr + i * n * 4, out, outOffset + i * outStride, n)
        } finally {
            WasmKernels.free(cPtr)
            if (!resident) WasmKernels.free(bPtr)
            WasmKernels.free(aPtr)
        }
    }
}

/** BF16 weights in the sidecar (`skainet_bf16_matmul`). */
public object WasmBf16MatmulKernel : Bf16MatmulKernel {
    private val impl = WasmNarrowFloatMatmulKernel("_skainet_bf16_matmul")
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ): Unit = impl.matmul(a, aOffset, aStride, b, bByteOffset, bByteStride, out, outOffset, outStride, m, n, k)
}

/** FP16 weights in the sidecar (`skainet_fp16_matmul`). */
public object WasmFp16MatmulKernel : Fp16MatmulKernel {
    private val impl = WasmNarrowFloatMatmulKernel("_skainet_fp16_matmul")
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ): Unit = impl.matmul(a, aOffset, aStride, b, bByteOffset, bByteStride, out, outOffset, outStride, m, n, k)
}

/** `C(m,n) = A(m,k) · B(k,n)` in the sidecar; both operands cross per call (dense weights are not cached). */
public object WasmFp32MatmulKernel : Fp32MatmulKernel {
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) { "m, n, k must be non-negative; got m=$m n=$n k=$k" }
        if (m == 0 || n == 0) return
        // Stage dense blocks contiguously (stride == width) so the kernel reads them as such.
        val aPtr = WasmKernels.alloc(m * k * 4)
        val bPtr = WasmKernels.alloc(k * n * 4)
        val cPtr = WasmKernels.alloc(m * n * 4)
        try {
            if (aStride == k) WasmKernels.writeFloats(aPtr, a, aOffset, m * k)
            else for (i in 0 until m) WasmKernels.writeFloats(aPtr + i * k * 4, a, aOffset + i * aStride, k)
            if (bStride == n) WasmKernels.writeFloats(bPtr, b, bOffset, k * n)
            else for (p in 0 until k) WasmKernels.writeFloats(bPtr + p * n * 4, b, bOffset + p * bStride, n)
            WasmKernels.fp32Matmul(aPtr, k, bPtr, n, cPtr, n, m, n, k)
            if (outStride == n) WasmKernels.readFloats(cPtr, out, outOffset, m * n)
            else for (i in 0 until m) WasmKernels.readFloats(cPtr + i * n * 4, out, outOffset + i * outStride, n)
        } finally {
            WasmKernels.free(cPtr)
            WasmKernels.free(bPtr)
            WasmKernels.free(aPtr)
        }
    }
}

/**
 * `out(m,n) = x(m,k) · w(n,k)ᵀ` in the sidecar (`skainet_fp32_matmul_wt`). A whole-array weight is a
 * parameter and stays resident after its first call ([WasmKernels.denseWeightHandle]); a windowed
 * weight is staged per call, as the activation always is.
 */
public object WasmFp32MatmulWtKernel : Fp32MatmulWtKernel {
    override fun matmulWt(
        x: FloatArray, xOffset: Int,
        w: FloatArray, wOffset: Int,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) { "m, n, k must be non-negative; got m=$m n=$n k=$k" }
        if (m == 0 || n == 0) return
        val resident = wOffset == 0 && w.size == n * k
        val inPtr = WasmKernels.alloc(maxOf(m * k, 1) * 4)
        val wPtr = if (resident) WasmKernels.denseWeightHandle(w) else WasmKernels.alloc(maxOf(n * k, 1) * 4)
        val outPtr = WasmKernels.alloc(m * n * 4)
        try {
            if (k > 0) {
                WasmKernels.writeFloats(inPtr, x, xOffset, m * k)
                if (!resident) WasmKernels.writeFloats(wPtr, w, wOffset, n * k)
            }
            WasmKernels.fp32MatmulWt(inPtr, wPtr, outPtr, m, n, k)
            WasmKernels.readFloats(outPtr, out, outOffset, m * n)
        } finally {
            WasmKernels.free(outPtr)
            if (!resident) WasmKernels.free(wPtr)
            WasmKernels.free(inPtr)
        }
    }
}
