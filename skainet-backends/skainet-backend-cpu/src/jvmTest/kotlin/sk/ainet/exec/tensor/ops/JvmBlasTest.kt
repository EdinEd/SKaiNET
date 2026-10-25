package sk.ainet.exec.tensor.ops

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * The platform BLAS binding and the three dense FP32 products the JVM ops send to it. Runs wherever a
 * `cblas_sgemm` resolves — Accelerate on macOS, OpenBLAS or the reference BLAS on Linux, or the library
 * named by `-Dskainet.cpu.blas.library` — and reports itself skipped otherwise, so a CI host without a
 * BLAS neither fails nor pretends. The reference is a double-accumulated product; the bar is the
 * rounding tolerance the kernel parity tests use, since a BLAS orders its sums as it likes.
 */
class JvmBlasTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(dataFactory)

    private fun available(): Boolean {
        if (JvmBlas.isAvailable()) return true
        println("[skip] no cblas_sgemm resolved (set -Dskainet.cpu.blas.library=<path>); BLAS tests not run")
        return false
    }

    private fun reference(
        a: FloatArray, aOff: Int, lda: Int, transA: Boolean,
        b: FloatArray, bOff: Int, ldb: Int, transB: Boolean,
        m: Int, n: Int, k: Int,
    ): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (j in 0 until n) {
            var acc = 0.0
            for (p in 0 until k) {
                val av = if (transA) a[aOff + p * lda + i] else a[aOff + i * lda + p]
                val bv = if (transB) b[bOff + j * ldb + p] else b[bOff + p * ldb + j]
                acc += av.toDouble() * bv
            }
            out[i * n + j] = acc.toFloat()
        }
        return out
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String) {
        assertEquals(expected.size, actual.size, "$what size")
        var maxAbs = 0f
        for (v in expected) maxAbs = maxOf(maxAbs, abs(v))
        val tol = maxAbs * 1e-5f + 1e-6f
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${actual[i]} vs ${expected[i]} (tol $tol)")
        }
    }

    private fun segmentOf(arena: Arena, values: FloatArray): MemorySegment {
        val seg = arena.allocate(values.size.toLong() * Float.SIZE_BYTES, 64)
        MemorySegment.copy(values, 0, seg, ValueLayout.JAVA_FLOAT, 0L, values.size)
        return seg
    }

    @Test
    fun segmentFaceMatchesTheReferenceForBothTransposes() {
        if (!available()) return
        println("[blas] ${JvmBlas.libraryDescription}")
        val rng = Random(31)
        Arena.ofConfined().use { arena ->
            for ((m, n, k) in listOf(Triple(13, 37, 300), Triple(1, 1024, 256), Triple(6, 16, 1), Triple(63, 1021, 1023))) {
                val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
                // B as [k, n] for the plain product and as [n, k] (a Linear weight) for x · Wᵀ.
                val bKN = FloatArray(k * n) { rng.nextFloat() - 0.5f }
                val bNK = FloatArray(n * k) { rng.nextFloat() - 0.5f }
                val aSeg = segmentOf(arena, a)
                val cSeg = arena.allocate(m.toLong() * n * Float.SIZE_BYTES, 64)
                val out = FloatArray(m * n)

                assertTrue(JvmBlas.sgemm(false, false, m, n, k, aSeg, 0L, k, segmentOf(arena, bKN), 0L, n, cSeg, 0L, n))
                MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, m * n)
                assertClose(reference(a, 0, k, false, bKN, 0, n, false, m, n, k), out, "NN m=$m n=$n k=$k")

                assertTrue(JvmBlas.sgemm(false, true, m, n, k, aSeg, 0L, k, segmentOf(arena, bNK), 0L, k, cSeg, 0L, n))
                MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, m * n)
                assertClose(reference(a, 0, k, false, bNK, 0, k, true, m, n, k), out, "NT m=$m n=$n k=$k")
            }
            // k = 0 zeroes the block rather than reading anything.
            val cSeg = arena.allocate(3L * 4 * Float.SIZE_BYTES, 64).also { it.fill(1) }
            assertTrue(JvmBlas.sgemm(false, false, 3, 4, 0, cSeg, 0L, 1, cSeg, 0L, 4, cSeg, 0L, 4))
            for (i in 0 until 12) assertEquals(0f, cSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i.toLong()))
        }
    }

    @Test
    fun heapFaceTakesOffsetsAndStridesAndLeavesThePaddingAlone() {
        if (!available()) return
        val rng = Random(32)
        val m = 7; val n = 19; val k = 33
        val aStride = k + 5; val bStride = n + 3; val outStride = n + 7
        val aOff = 2; val bOff = 1; val outOff = 3
        val a = FloatArray(aOff + m * aStride) { rng.nextFloat() - 0.5f }
        val b = FloatArray(bOff + k * bStride) { rng.nextFloat() - 0.5f }
        val out = FloatArray(outOff + m * outStride) { 123f }
        assertTrue(JvmBlas.sgemmRowMajorNN(m, n, k, a, aOff, aStride, b, bOff, bStride, out, outOff, outStride))
        val block = FloatArray(m * n)
        for (i in 0 until m) System.arraycopy(out, outOff + i * outStride, block, i * n, n)
        assertClose(reference(a, aOff, aStride, false, b, bOff, bStride, false, m, n, k), block, "heap NN")
        for (idx in 0 until outOff) assertEquals(123f, out[idx], "prefix $idx")
        for (i in 0 until m) for (j in n until outStride) {
            if (outOff + i * outStride + j < out.size) assertEquals(123f, out[outOff + i * outStride + j], "pad [$i, $j]")
        }
    }

    private fun heapTensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values), FP32::class)

    @Suppress("UNCHECKED_CAST")
    private fun segmentTensor(shape: Shape, values: FloatArray, arena: Arena): Tensor<FP32, Float> {
        val data = MemorySegmentTensorData<FP32>(shape, arena)
        data.copyFromFloatArray(values)
        return VoidOpsTensor(data as TensorData<FP32, Float>, FP32::class)
    }

    private fun floatsOf(t: Tensor<*, *>): FloatArray = when (val d = t.data) {
        is FloatArrayTensorData<*> -> d.buffer.copyOf(t.shape.volume)
        is MemorySegmentTensorData<*> -> d.copyToFloatArray()
        else -> error("unexpected data ${d::class}")
    }

    /**
     * With BLAS switched on, the three routed products — heap `A · B`, segment `A · B`, and `x · Wᵀ` on
     * a dense `[out, in]` weight — agree with the reference, and the same products with BLAS switched
     * off (the bundled kernels) agree with them within rounding. Shapes sit above the small-work floor.
     */
    @Test
    fun opsProductsAgreeWithBlasOnAndOff() {
        if (!available()) return
        val rng = Random(33)
        val m = 48; val n = 320; val k = 256 // 3.9 M multiply-adds: above the BLAS floor, below the small-shape loop's ceiling
        val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bKN = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
        val expectedNN = reference(x, 0, k, false, bKN, 0, n, false, m, n, k)
        val expectedNT = reference(x, 0, k, false, w, 0, k, true, m, n, k)

        val previous = System.getProperty("skainet.cpu.blas.enabled")
        try {
            val results = mutableMapOf<String, FloatArray>()
            for (blas in listOf(true, false)) {
                System.setProperty("skainet.cpu.blas.enabled", blas.toString())
                assertEquals(blas, JvmCpuBackendConfig.blasEnabled)
                Arena.ofConfined().use { arena ->
                    val heap = ops.matmul(heapTensor(Shape(m, k), x), heapTensor(Shape(k, n), bKN))
                    val seg = ops.matmul(segmentTensor(Shape(m, k), x, arena), segmentTensor(Shape(k, n), bKN, arena))
                    val wt = ops.matmulWeightTransposed(heapTensor(Shape(m, k), x), heapTensor(Shape(n, k), w))
                    results["heap blas=$blas"] = floatsOf(heap).also { assertClose(expectedNN, it, "heap A·B blas=$blas") }
                    results["segment blas=$blas"] = floatsOf(seg).also { assertClose(expectedNN, it, "segment A·B blas=$blas") }
                    results["wt blas=$blas"] = floatsOf(wt).also { assertClose(expectedNT, it, "x·Wᵀ blas=$blas") }
                }
            }
            for (face in listOf("heap", "segment", "wt")) {
                assertClose(results["$face blas=false"]!!, results["$face blas=true"]!!, "$face on vs off")
            }
        } finally {
            if (previous == null) System.clearProperty("skainet.cpu.blas.enabled") else System.setProperty("skainet.cpu.blas.enabled", previous)
        }
    }
}
