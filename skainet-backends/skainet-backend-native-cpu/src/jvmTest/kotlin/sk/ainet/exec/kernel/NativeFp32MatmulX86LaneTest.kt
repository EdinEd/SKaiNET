package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tails, the strides and the placement independence of `skainet_fp32_matmul`. The x86-64 AVX2/FMA
 * lane tiles six rows of A by sixteen columns of C in a 256-deep k-block, packs each sixteen-column
 * panel of B, finishes the rows left under six with 2×16 and 1×16 tiles and the columns left under
 * sixteen through a lane-masked stripe, so every edge a tile can fall off is a shape here: `m` under,
 * at and past one tile, `n` around eight and sixteen, `k` around the k-block. The reference is a
 * double-accumulated product. Every C element takes one fused multiply-add per `p` in ascending `p`
 * whichever tile holds it, so a block of the output computed on its own (another slice, another
 * thread, a sub-matrix through strides and offsets) is checked to be bit-identical to the same
 * elements of the full product. On a host without AVX2 the same tests exercise the portable lane.
 */
class NativeFp32MatmulX86LaneTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeFp32MatmulKernel.isAvailable(), "bundled libskainet_kernels must resolve skainet_fp32_matmul")
    }

    private fun reference(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        m: Int, n: Int, k: Int,
    ): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (j in 0 until n) {
            var acc = 0.0
            for (p in 0 until k) acc += a[aOffset + i * aStride + p].toDouble() * b[bOffset + p * bStride + j]
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

    /** The m×n block of a strided, offset C buffer as a contiguous array. */
    private fun block(c: FloatArray, cOffset: Int, cStride: Int, m: Int, n: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) System.arraycopy(c, cOffset + i * cStride, out, i * n, n)
        return out
    }

    @Test
    fun everyTileEdgeMatchesTheDoubleReference() {
        val rng = Random(21)
        val ms = listOf(1, 2, 3, 5, 6, 7, 13, 63)
        val ns = listOf(1, 7, 8, 15, 16, 17, 33, 1021)
        val ks = listOf(1, 255, 257, 1023)
        for (m in ms) for (n in ns) for (k in ks) {
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val out = FloatArray(m * n)
            NativeFp32MatmulKernel.matmul(a, 0, k, b, 0, n, out, 0, n, m, n, k)
            assertClose(reference(a, 0, k, b, 0, n, m, n, k), out, "m=$m n=$n k=$k")
        }
    }

    @Test
    fun stridedOperandsAndAnOffsetOutputLeaveThePaddingAlone() {
        val rng = Random(22)
        for ((m, n, k) in listOf(Triple(13, 37, 300), Triple(6, 16, 256), Triple(7, 17, 257), Triple(1, 1, 1))) {
            val aStride = k + 5
            val bStride = n + 3
            val cStride = n + 7
            val aOffset = 2
            val bOffset = 1
            val cOffset = 3
            val a = FloatArray(aOffset + m * aStride) { rng.nextFloat() - 0.5f }
            val b = FloatArray(bOffset + k * bStride) { rng.nextFloat() - 0.5f }
            val c = FloatArray(cOffset + m * cStride) { 123f }
            NativeFp32MatmulKernel.matmul(a, aOffset, aStride, b, bOffset, bStride, c, cOffset, cStride, m, n, k)
            assertClose(reference(a, aOffset, aStride, b, bOffset, bStride, m, n, k), block(c, cOffset, cStride, m, n), "m=$m n=$n k=$k")
            for (idx in 0 until cOffset) assertEquals(123f, c[idx], "prefix $idx must be untouched")
            for (i in 0 until m) for (j in n until cStride) {
                if (cOffset + i * cStride + j < c.size) assertEquals(123f, c[cOffset + i * cStride + j], "pad [$i, $j] must be untouched")
            }
        }
    }

    @Test
    fun anOutputDoesNotDependOnTheBlockThatComputesIt() {
        val rng = Random(23)
        // Shapes whose full call is one block (under the row-pool threshold) and one whose columns
        // the pool splits across threads (n ≥ 512 > m). The partial calls move every output to a
        // different place inside a tile — 5 and 3 are not multiples of 6 or 16 — and change which
        // tile variant computes it (6×16, 2×16, 1×16 or the masked stripe).
        for ((m, n, k) in listOf(Triple(13, 37, 300), Triple(7, 19, 1023), Triple(2, 18, 8), Triple(14, 1030, 64))) {
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val full = FloatArray(m * n)
            NativeFp32MatmulKernel.matmul(a, 0, k, b, 0, n, full, 0, n, m, n, k)

            // Rows [0, cut) and [cut, m) as two calls through offsets into the same strided output.
            val byRows = FloatArray(m * n)
            val cut = minOf(5, m - 1)
            NativeFp32MatmulKernel.matmul(a, 0, k, b, 0, n, byRows, 0, n, cut, n, k)
            NativeFp32MatmulKernel.matmul(a, cut * k, k, b, 0, n, byRows, cut * n, n, m - cut, n, k)
            assertContentEquals(full, byRows, "m=$m n=$n k=$k: a row boundary must not change a bit")

            // Columns [0, 3) and [3, n) as two calls through offsets and the full strides.
            val byColumns = FloatArray(m * n)
            val col = 3
            NativeFp32MatmulKernel.matmul(a, 0, k, b, 0, n, byColumns, 0, n, m, col, k)
            NativeFp32MatmulKernel.matmul(a, 0, k, b, col, n, byColumns, col, n, m, n - col, k)
            assertContentEquals(full, byColumns, "m=$m n=$n k=$k: a column boundary must not change a bit")

            // One row at a time: the 1×16 tile and the one-row stripe against the 6×16 tile.
            val rows = FloatArray(m * n)
            for (i in 0 until m) NativeFp32MatmulKernel.matmul(a, i * k, k, b, 0, n, rows, i * n, n, 1, n, k)
            assertContentEquals(full, rows, "m=$m n=$n k=$k: a row-at-a-time call must not change a bit")
        }
    }

    /**
     * `-Dskainet.runBench=true`: one native call per shape with all three operands resident off-heap,
     * so the C row pool is the only threading and no heap copy is timed (the SPI face copies A, B and
     * C on every call; see [NativeFp32MatmulKernel]).
     */
    @Test
    fun singleCallBench() {
        if (System.getProperty("skainet.runBench") != "true") return
        val lookup = NativeLibraryLoader.lookup() ?: error("native library not loaded")
        val symbol = lookup.find("skainet_fp32_matmul").orElseThrow()
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
        )
        val handle = Linker.nativeLinker().downcallHandle(symbol, descriptor)
        val shapes = listOf(
            Triple(63, 1024, 1024), Triple(183, 1024, 1024), Triple(63, 2624, 1024), Triple(63, 1024, 2624),
            Triple(1024, 1024, 1024), Triple(512, 512, 512), Triple(192, 192, 64), Triple(1, 4096, 4096),
        )
        println("[gemm-x86] host ${System.getProperty("os.name")} ${System.getProperty("os.arch")} JDK ${System.getProperty("java.version")}")
        Arena.ofConfined().use { arena ->
            for ((m, n, k) in shapes) {
                val rng = Random(m * 31 + n)
                fun segment(count: Int, fill: Boolean): MemorySegment {
                    val seg = arena.allocate(count.toLong() * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())
                    if (fill) {
                        val values = FloatArray(count) { rng.nextFloat() - 0.5f }
                        MemorySegment.copy(values, 0, seg, ValueLayout.JAVA_FLOAT, 0L, count)
                    }
                    return seg
                }
                val aSeg = segment(m * k, fill = true)
                val bSeg = segment(k * n, fill = true)
                val cSeg = segment(m * n, fill = false)
                val flop = 2.0 * m * n * k
                val reps = maxOf(3, (2e9 / flop).toInt().coerceAtMost(400))
                val call = { handle.invoke(aSeg, 0, k, bSeg, 0, n, cSeg, 0, n, m, n, k); Unit }
                repeat(reps / 2 + 1) { call() }
                val samples = LongArray(5)
                for (s in samples.indices) {
                    val t0 = System.nanoTime()
                    repeat(reps) { call() }
                    samples[s] = (System.nanoTime() - t0) / reps
                }
                val ns = samples.sorted()[2]
                println("[gemm-x86] m=$m n=$n k=$k: %.3f ms/call %.1f GFLOP/s".format(ns / 1e6, flop / ns))
            }
        }
    }
}
