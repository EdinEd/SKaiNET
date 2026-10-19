package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Fp32AttentionKernel
import sk.ainet.backend.api.kernel.Fp32NormKernel
import sk.ainet.backend.api.kernel.Fp32SoftmaxKernel

/*
 * Native (FFM) bindings of the fused dense FP32 row kernels (`f32_fused.c`, SKEEP-007 fused ops):
 * softmax rows, layer norm / RMS norm and banded scaled dot-product attention. The same symbols the
 * wasm sidecar exports; on the JVM they replace the scalar Kotlin loops of `DefaultCpuOpsBase`
 * for these ops (attention there is scalar per head, parallel over heads through the schedule;
 * here the C kernel vectorises the dot products and threads over query rows in the row pool).
 * Every face stages its operands off-heap per call and copies the output back, like
 * [NativeFp32MatmulKernel]; the operands of these ops are activations, so there is nothing to keep
 * resident.
 */

private fun bind(symbol: String, descriptor: FunctionDescriptor): MethodHandle? {
    val lookup = NativeLibraryLoader.lookup() ?: return null
    val sym = lookup.find(symbol).orElse(null) ?: return null
    return runCatching { Linker.nativeLinker().downcallHandle(sym, descriptor) }.getOrNull()
}

/** [count] floats of [src] from [offset] as a fresh confined segment; `MemorySegment.NULL` for an empty count. */
private fun stageFloats(arena: Arena, src: FloatArray, offset: Int, count: Long, what: String): MemorySegment {
    require(offset >= 0 && offset + count <= src.size) { "$what needs $count floats from $offset, array has ${src.size}" }
    if (count == 0L) return MemorySegment.NULL
    val seg = arena.allocate(count * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())
    MemorySegment.copy(src, offset, seg, ValueLayout.JAVA_FLOAT, 0L, count.toInt())
    return seg
}

private fun outSegment(arena: Arena, count: Long): MemorySegment =
    arena.allocate(maxOf(count, 1L) * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())

private fun readFloats(seg: MemorySegment, dst: FloatArray, offset: Int, count: Long, what: String) {
    require(offset >= 0 && offset + count <= dst.size) { "$what needs $count floats from $offset, array has ${dst.size}" }
    if (count > 0) MemorySegment.copy(seg, ValueLayout.JAVA_FLOAT, 0L, dst, offset, count.toInt())
}

/** `skainet_f32_softmax_rows` through FFM. */
internal object NativeFp32SoftmaxKernel : Fp32SoftmaxKernel {

    fun isAvailable(): Boolean = handle != null

    override fun softmaxRows(input: FloatArray, inputOffset: Int, out: FloatArray, outOffset: Int, rows: Int, cols: Int, log: Boolean) {
        require(rows >= 0 && cols >= 0) { "NativeFp32SoftmaxKernel: rows and cols must be non-negative; got rows=$rows cols=$cols" }
        if (rows == 0 || cols == 0) return
        val mh = handle ?: error("NativeFp32SoftmaxKernel invoked while native library unavailable")
        val n = rows.toLong() * cols
        Arena.ofConfined().use { arena ->
            val inSeg = stageFloats(arena, input, inputOffset, n, "NativeFp32SoftmaxKernel: input")
            val outSeg = outSegment(arena, n)
            mh.invoke(inSeg, outSeg, rows, cols, if (log) 1 else 0)
            readFloats(outSeg, out, outOffset, n, "NativeFp32SoftmaxKernel: output")
        }
    }

    private val handle: MethodHandle? by lazy {
        bind(
            "skainet_f32_softmax_rows",
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,  // in
                ValueLayout.ADDRESS,  // out
                ValueLayout.JAVA_INT, // rows
                ValueLayout.JAVA_INT, // cols
                ValueLayout.JAVA_INT, // log_form
            ),
        )
    }
}

/** `skainet_f32_layer_norm` and `skainet_f32_rms_norm` through FFM. */
internal object NativeFp32NormKernel : Fp32NormKernel {

    fun isAvailable(): Boolean = layerNormHandle != null && rmsNormHandle != null

    override fun layerNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?, beta: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int, eps: Float,
    ) {
        require(rows >= 0 && cols >= 0) { "NativeFp32NormKernel: rows and cols must be non-negative; got rows=$rows cols=$cols" }
        if (rows == 0 || cols == 0) return
        val mh = layerNormHandle ?: error("NativeFp32NormKernel invoked while native library unavailable")
        val n = rows.toLong() * cols
        Arena.ofConfined().use { arena ->
            val inSeg = stageFloats(arena, input, inputOffset, n, "NativeFp32NormKernel: input")
            val gSeg = gamma?.let { stageFloats(arena, it, 0, cols.toLong(), "NativeFp32NormKernel: gamma") } ?: MemorySegment.NULL
            val bSeg = beta?.let { stageFloats(arena, it, 0, cols.toLong(), "NativeFp32NormKernel: beta") } ?: MemorySegment.NULL
            val outSeg = outSegment(arena, n)
            mh.invoke(inSeg, gSeg, bSeg, outSeg, rows, cols, eps)
            readFloats(outSeg, out, outOffset, n, "NativeFp32NormKernel: output")
        }
    }

    override fun rmsNorm(
        input: FloatArray, inputOffset: Int,
        gamma: FloatArray?,
        out: FloatArray, outOffset: Int,
        rows: Int, cols: Int, eps: Float,
    ) {
        require(rows >= 0 && cols >= 0) { "NativeFp32NormKernel: rows and cols must be non-negative; got rows=$rows cols=$cols" }
        if (rows == 0 || cols == 0) return
        val mh = rmsNormHandle ?: error("NativeFp32NormKernel invoked while native library unavailable")
        val n = rows.toLong() * cols
        Arena.ofConfined().use { arena ->
            val inSeg = stageFloats(arena, input, inputOffset, n, "NativeFp32NormKernel: input")
            val gSeg = gamma?.let { stageFloats(arena, it, 0, cols.toLong(), "NativeFp32NormKernel: gamma") } ?: MemorySegment.NULL
            val outSeg = outSegment(arena, n)
            mh.invoke(inSeg, gSeg, outSeg, rows, cols, eps)
            readFloats(outSeg, out, outOffset, n, "NativeFp32NormKernel: output")
        }
    }

    private val layerNormHandle: MethodHandle? by lazy {
        bind(
            "skainet_f32_layer_norm",
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,    // in
                ValueLayout.ADDRESS,    // gamma (nullable)
                ValueLayout.ADDRESS,    // beta (nullable)
                ValueLayout.ADDRESS,    // out
                ValueLayout.JAVA_INT,   // rows
                ValueLayout.JAVA_INT,   // cols
                ValueLayout.JAVA_FLOAT, // eps
            ),
        )
    }

    private val rmsNormHandle: MethodHandle? by lazy {
        bind(
            "skainet_f32_rms_norm",
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,    // in
                ValueLayout.ADDRESS,    // gamma (nullable)
                ValueLayout.ADDRESS,    // out
                ValueLayout.JAVA_INT,   // rows
                ValueLayout.JAVA_INT,   // cols
                ValueLayout.JAVA_FLOAT, // eps
            ),
        )
    }
}

/** `skainet_f32_sdpa` through FFM. */
internal object NativeFp32AttentionKernel : Fp32AttentionKernel {

    fun isAvailable(): Boolean = handle != null

    override fun sdpa(
        query: FloatArray, key: FloatArray, value: FloatArray, out: FloatArray,
        batch: Int, heads: Int, kvHeads: Int,
        seqQ: Int, seqKV: Int, headDim: Int,
        scale: Float, causal: Boolean, window: Int, rightContext: Int,
        keyLengths: IntArray?,
        mask: FloatArray?, maskDims: IntArray?,
    ) {
        require(batch >= 0 && heads >= 0 && kvHeads >= 0 && seqQ >= 0 && seqKV >= 0 && headDim >= 0) {
            "NativeFp32AttentionKernel: dimensions must be non-negative"
        }
        require(kvHeads > 0 && heads % kvHeads == 0) { "NativeFp32AttentionKernel: heads ($heads) must be a multiple of kvHeads ($kvHeads)" }
        require(keyLengths == null || keyLengths.size == batch) { "NativeFp32AttentionKernel: keyLengths has ${keyLengths?.size} rows, batch is $batch" }
        require((mask == null) == (maskDims == null) && (maskDims == null || maskDims.size == 4)) {
            "NativeFp32AttentionKernel: mask and maskDims (4 dims) go together"
        }
        val outCount = batch.toLong() * heads * seqQ * headDim
        require(outCount <= out.size) { "NativeFp32AttentionKernel: output needs $outCount floats, array has ${out.size}" }
        if (outCount == 0L) return
        val mh = handle ?: error("NativeFp32AttentionKernel invoked while native library unavailable")
        val qCount = batch.toLong() * heads * seqQ * headDim
        val kvCount = batch.toLong() * kvHeads * seqKV * headDim
        Arena.ofConfined().use { arena ->
            val qSeg = stageFloats(arena, query, 0, qCount, "NativeFp32AttentionKernel: query")
            val kSeg = stageFloats(arena, key, 0, kvCount, "NativeFp32AttentionKernel: key")
            val vSeg = stageFloats(arena, value, 0, kvCount, "NativeFp32AttentionKernel: value")
            val outSeg = outSegment(arena, outCount)
            val lSeg = if (keyLengths != null && batch > 0) {
                val seg = arena.allocate(batch.toLong() * Integer.BYTES, ValueLayout.JAVA_INT.byteAlignment())
                MemorySegment.copy(keyLengths, 0, seg, ValueLayout.JAVA_INT, 0L, batch)
                seg
            } else MemorySegment.NULL
            val md = maskDims ?: intArrayOf(1, 1, 1, 1)
            val mSeg = if (mask != null) {
                stageFloats(arena, mask, 0, md[0].toLong() * md[1] * md[2] * md[3], "NativeFp32AttentionKernel: mask")
            } else MemorySegment.NULL
            mh.invoke(
                qSeg, kSeg, vSeg, outSeg,
                batch, heads, kvHeads, seqQ, seqKV, headDim,
                scale, if (causal) 1 else 0, window, rightContext,
                lSeg, mSeg, md[0], md[1], md[2], md[3],
            )
            readFloats(outSeg, out, 0, outCount, "NativeFp32AttentionKernel: output")
        }
    }

    private val handle: MethodHandle? by lazy {
        bind(
            "skainet_f32_sdpa",
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS,    // q
                ValueLayout.ADDRESS,    // k
                ValueLayout.ADDRESS,    // v
                ValueLayout.ADDRESS,    // out
                ValueLayout.JAVA_INT,   // batch
                ValueLayout.JAVA_INT,   // heads
                ValueLayout.JAVA_INT,   // kv_heads
                ValueLayout.JAVA_INT,   // seq_q
                ValueLayout.JAVA_INT,   // seq_kv
                ValueLayout.JAVA_INT,   // head_dim
                ValueLayout.JAVA_FLOAT, // scale
                ValueLayout.JAVA_INT,   // causal
                ValueLayout.JAVA_INT,   // window
                ValueLayout.JAVA_INT,   // right_context
                ValueLayout.ADDRESS,    // lengths (nullable)
                ValueLayout.ADDRESS,    // mask (nullable)
                ValueLayout.JAVA_INT,   // mask_b
                ValueLayout.JAVA_INT,   // mask_h
                ValueLayout.JAVA_INT,   // mask_q
                ValueLayout.JAVA_INT,   // mask_k
            ),
        )
    }
}
