package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.Fp32AttentionKernel
import sk.ainet.backend.api.kernel.Fp32NormKernel
import sk.ainet.backend.api.kernel.Fp32SoftmaxKernel

/*
 * The fused row kernels of the sidecar (`f32_fused.c`) behind the SPI faces of `KernelProvider`
 * (SKEEP-007 fused ops): the same accessors the JVM FFM provider answers, for callers that hold
 * heap arrays. Every face stages its operands into the sidecar per call and reads the output
 * back; [WasmCpuOps] bypasses them and hands the kernels resident operands directly.
 */

/** `skainet_f32_softmax_rows` over heap arrays. */
public object WasmFp32SoftmaxKernel : Fp32SoftmaxKernel {
    override fun softmaxRows(input: FloatArray, inputOffset: Int, out: FloatArray, outOffset: Int, rows: Int, cols: Int, log: Boolean) {
        require(rows >= 0 && cols >= 0) { "rows and cols must be non-negative; got rows=$rows cols=$cols" }
        if (rows == 0 || cols == 0) return
        val n = rows * cols
        val inPtr = WasmKernels.alloc(n * 4)
        val outPtr = WasmKernels.alloc(n * 4)
        try {
            WasmKernels.writeFloats(inPtr, input, inputOffset, n)
            WasmKernels.f32SoftmaxRows(inPtr, outPtr, rows, cols, log)
            WasmKernels.readFloats(outPtr, out, outOffset, n)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }
}

/** `skainet_f32_layer_norm` / `skainet_f32_rms_norm` over heap arrays. */
public object WasmFp32NormKernel : Fp32NormKernel {
    override fun layerNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?, beta: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int, eps: Float,
    ): Unit = run(input, inputOffset, gamma, beta, out, outOffset, rows, cols, eps, rms = false)

    override fun rmsNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int, eps: Float,
    ): Unit = run(input, inputOffset, gamma, null, out, outOffset, rows, cols, eps, rms = true)

    private fun run(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?, beta: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int, eps: Float, rms: Boolean,
    ) {
        require(rows >= 0 && cols >= 0) { "rows and cols must be non-negative; got rows=$rows cols=$cols" }
        if (rows == 0 || cols == 0) return
        require(gamma == null || gamma.size >= cols) { "gamma has ${gamma?.size} values, the row has $cols" }
        require(beta == null || beta.size >= cols) { "beta has ${beta?.size} values, the row has $cols" }
        val n = rows * cols
        val inPtr = WasmKernels.alloc(n * 4)
        val outPtr = WasmKernels.alloc(n * 4)
        val gPtr = if (gamma != null) WasmKernels.alloc(cols * 4) else 0
        val bPtr = if (beta != null) WasmKernels.alloc(cols * 4) else 0
        try {
            WasmKernels.writeFloats(inPtr, input, inputOffset, n)
            if (gamma != null) WasmKernels.writeFloats(gPtr, gamma, 0, cols)
            if (beta != null) WasmKernels.writeFloats(bPtr, beta, 0, cols)
            if (rms) WasmKernels.f32RmsNorm(inPtr, gPtr, outPtr, rows, cols, eps)
            else WasmKernels.f32LayerNorm(inPtr, gPtr, bPtr, outPtr, rows, cols, eps)
            WasmKernels.readFloats(outPtr, out, outOffset, n)
        } finally {
            if (bPtr != 0) WasmKernels.free(bPtr)
            if (gPtr != 0) WasmKernels.free(gPtr)
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }
}

/** `skainet_f32_sdpa` over heap arrays. */
public object WasmFp32AttentionKernel : Fp32AttentionKernel {
    override fun sdpa(
        query: FloatArray, key: FloatArray, value: FloatArray, out: FloatArray,
        batch: Int, heads: Int, kvHeads: Int,
        seqQ: Int, seqKV: Int, headDim: Int,
        scale: Float, causal: Boolean, window: Int, rightContext: Int,
        keyLengths: IntArray?,
        mask: FloatArray?, maskDims: IntArray?,
    ) {
        require(batch >= 0 && heads >= 0 && kvHeads >= 0 && seqQ >= 0 && seqKV >= 0 && headDim >= 0) { "dimensions must be non-negative" }
        require(kvHeads > 0 && heads % kvHeads == 0) { "heads ($heads) must be a multiple of kvHeads ($kvHeads)" }
        require(keyLengths == null || keyLengths.size == batch) { "keyLengths has ${keyLengths?.size} rows, batch is $batch" }
        require((mask == null) == (maskDims == null) && (maskDims == null || maskDims.size == 4)) { "mask and maskDims (4 dims) go together" }
        val outCount = batch * heads * seqQ * headDim
        require(outCount <= out.size) { "output needs $outCount floats, array has ${out.size}" }
        if (outCount == 0) return
        val kvCount = batch * kvHeads * seqKV * headDim
        val md = maskDims ?: intArrayOf(1, 1, 1, 1)
        val maskCount = md[0] * md[1] * md[2] * md[3]
        val qPtr = WasmKernels.alloc(outCount * 4)
        val kPtr = WasmKernels.alloc(maxOf(kvCount, 1) * 4)
        val vPtr = WasmKernels.alloc(maxOf(kvCount, 1) * 4)
        val outPtr = WasmKernels.alloc(outCount * 4)
        val lPtr = if (keyLengths != null) WasmKernels.alloc(batch * 4) else 0
        val mPtr = if (mask != null) WasmKernels.alloc(maxOf(maskCount, 1) * 4) else 0
        try {
            WasmKernels.writeFloats(qPtr, query, 0, outCount)
            if (kvCount > 0) {
                WasmKernels.writeFloats(kPtr, key, 0, kvCount)
                WasmKernels.writeFloats(vPtr, value, 0, kvCount)
            }
            if (keyLengths != null) WasmKernels.writeInts(lPtr, keyLengths, 0, batch)
            if (mask != null && maskCount > 0) WasmKernels.writeFloats(mPtr, mask, 0, maskCount)
            WasmKernels.f32Sdpa(
                qPtr, kPtr, vPtr, outPtr, batch, heads, kvHeads, seqQ, seqKV, headDim,
                scale, causal, window, rightContext, lPtr, mPtr, md[0], md[1], md[2], md[3],
            )
            WasmKernels.readFloats(outPtr, out, 0, outCount)
        } finally {
            if (mPtr != 0) WasmKernels.free(mPtr)
            if (lPtr != 0) WasmKernels.free(lPtr)
            WasmKernels.free(outPtr)
            WasmKernels.free(vPtr)
            WasmKernels.free(kPtr)
            WasmKernels.free(qPtr)
        }
    }
}
