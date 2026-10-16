package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
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
