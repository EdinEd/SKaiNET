package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Fp32MatmulWtKernel
import sk.ainet.backend.api.kernel.Fp32MemSegMatmulWtKernel

/**
 * Native (FFM) `x · Wᵀ` over a row-major `[out, in]` FP32 weight — the JVM binding of
 *
 *   void skainet_fp32_matmul_wt(
 *       const float* in, int32_t in_offset,
 *       const float* w,  int32_t w_offset,
 *       float* out,      int32_t out_offset,
 *       int32_t m, int32_t n, int32_t k);
 *
 * which the wasm sidecar has served since 0.61.0 and the JVM never bound: `DefaultCpuOpsJvm`
 * transposed every dense weight once into `[k, n]` for the GEMM kernels and cached the copy. The C
 * kernel computes every output as one contiguous dot product over the weight's own row (NEON or
 * auto-vectorised portable body, four rows per tile), threads over output rows through the shared
 * row pool, and accumulates in ascending `p` — bit-identical to transpose-then-GEMM.
 *
 * Two faces. The heap face ([Fp32MatmulWtKernel]) stages both operands off-heap per call, like
 * [NativeFp32MatmulKernel]; for a decode step that copy is the weight, so it is the slow one. The
 * segment face ([Fp32MemSegMatmulWtKernel]) takes the weight where it lies and copies only the
 * `m × k` activation and the `m × n` output, which is what the ops layer uses once it has made a
 * weight resident off-heap (the JVM counterpart of the sidecar's resident dense weights).
 */
internal object NativeFp32MatmulWtKernel : Fp32MatmulWtKernel, Fp32MemSegMatmulWtKernel {

    fun isAvailable(): Boolean = handle != null

    override fun matmulWt(
        x: FloatArray, xOffset: Int,
        w: FloatArray, wOffset: Int,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) { "NativeFp32MatmulWtKernel: m, n, k must be non-negative; got m=$m n=$n k=$k" }
        if (m == 0 || n == 0) return
        val mh = handle ?: error("NativeFp32MatmulWtKernel invoked while native library unavailable")
        val wFloats = n.toLong() * k
        require(wOffset >= 0 && wOffset + wFloats <= w.size) {
            "NativeFp32MatmulWtKernel: weight needs $wFloats floats from $wOffset, array has ${w.size}"
        }
        Arena.ofConfined().use { arena ->
            val align = ValueLayout.JAVA_FLOAT.byteAlignment()
            val wSeg: MemorySegment = if (wFloats > 0) arena.allocate(wFloats * java.lang.Float.BYTES, align) else MemorySegment.NULL
            if (wFloats > 0) MemorySegment.copy(w, wOffset, wSeg, ValueLayout.JAVA_FLOAT, 0L, wFloats.toInt())
            invoke(mh, arena, x, xOffset, wSeg, out, outOffset, m, n, k)
        }
    }

    override fun matmulWt(
        x: FloatArray, xOffset: Int,
        weight: MemorySegment, weightByteOffset: Long,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) { "NativeFp32MatmulWtKernel: m, n, k must be non-negative; got m=$m n=$n k=$k" }
        if (m == 0 || n == 0) return
        val mh = handle ?: error("NativeFp32MatmulWtKernel invoked while native library unavailable")
        val wBytes = n.toLong() * k * java.lang.Float.BYTES
        require(weightByteOffset >= 0 && weightByteOffset + wBytes <= weight.byteSize()) {
            "NativeFp32MatmulWtKernel: weight segment too small — needs $wBytes bytes from offset " +
                "$weightByteOffset, segment is ${weight.byteSize()} bytes"
        }
        Arena.ofConfined().use { arena ->
            invoke(mh, arena, x, xOffset, if (wBytes > 0) weight.asSlice(weightByteOffset) else MemorySegment.NULL, out, outOffset, m, n, k)
        }
    }

    /** Stage the activation and the output, run, copy the output back. The weight is already a segment. */
    private fun invoke(
        mh: MethodHandle, arena: Arena,
        x: FloatArray, xOffset: Int, wSeg: MemorySegment,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    ) {
        val xFloats = m.toLong() * k
        val outFloats = m.toLong() * n
        require(xOffset >= 0 && xOffset + xFloats <= x.size) {
            "NativeFp32MatmulWtKernel: activation needs $xFloats floats from $xOffset, array has ${x.size}"
        }
        require(outOffset >= 0 && outOffset + outFloats <= out.size) {
            "NativeFp32MatmulWtKernel: output needs $outFloats floats from $outOffset, array has ${out.size}"
        }
        val align = ValueLayout.JAVA_FLOAT.byteAlignment()
        val xSeg: MemorySegment = if (xFloats > 0) arena.allocate(xFloats * java.lang.Float.BYTES, align) else MemorySegment.NULL
        val outSeg: MemorySegment = arena.allocate(outFloats * java.lang.Float.BYTES, align)
        if (xFloats > 0) MemorySegment.copy(x, xOffset, xSeg, ValueLayout.JAVA_FLOAT, 0L, xFloats.toInt())
        mh.invoke(xSeg, 0, wSeg, 0, outSeg, 0, m, n, k)
        MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0L, out, outOffset, outFloats.toInt())
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_fp32_matmul_wt").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // in
            ValueLayout.JAVA_INT,   // in_offset
            ValueLayout.ADDRESS,    // w
            ValueLayout.JAVA_INT,   // w_offset
            ValueLayout.ADDRESS,    // out
            ValueLayout.JAVA_INT,   // out_offset
            ValueLayout.JAVA_INT,   // m
            ValueLayout.JAVA_INT,   // n
            ValueLayout.JAVA_INT,   // k
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
