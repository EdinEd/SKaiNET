package sk.ainet.exec.kernel

import java.lang.foreign.Arena
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
 * The tails and the slice independence of `skainet_fp32_matmul_wt`. The x86-64 AVX2/FMA lane tiles
 * four weight rows by two activation rows with eight-lane sums, so every edge the tile can fall off
 * is a shape here: `k` off a multiple of eight, `n` off a multiple of four, `m` odd and one. The
 * reference is a double-accumulated row dot; the kernel's own order (lanes, then a fixed horizontal
 * sum, then the scalar tail) is checked to be the same whichever slice an output lands in, which is
 * what the row pool and the SKEEP-005 column split rely on. On a host without AVX2 the same tests
 * exercise the portable lane.
 */
class NativeFp32MatmulWtX86LaneTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeFp32MatmulWtKernel.isAvailable(), "bundled libskainet_kernels must resolve skainet_fp32_matmul_wt")
    }

    private fun reference(x: FloatArray, w: FloatArray, m: Int, n: Int, k: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (o in 0 until n) {
            var acc = 0.0
            for (p in 0 until k) acc += x[i * k + p].toDouble() * w[o * k + p]
            out[i * n + o] = acc.toFloat()
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

    @Test
    fun everyTileEdgeMatchesTheRowDotReference() {
        val rng = Random(11)
        val ms = listOf(1, 2, 3, 63)
        val ns = listOf(1, 2, 3, 5, 1021)
        val ks = listOf(1, 7, 9, 1023)
        for (m in ms) for (n in ns) for (k in ks) {
            val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
            val out = FloatArray(m * n)
            NativeFp32MatmulWtKernel.matmulWt(x, 0, w, 0, out, 0, m, n, k)
            assertClose(reference(x, w, m, n, k), out, "m=$m n=$n k=$k")
        }
    }

    @Test
    fun anOutputDoesNotDependOnTheSliceThatComputesIt() {
        val rng = Random(12)
        // Shapes under the row-pool threshold, so the full call is one slice; the split calls put
        // every output at a different offset inside a tile (3 is not a multiple of 4) and change
        // which tile variant computes it (4×2, 4×1 or the single-row tail).
        for ((m, n, k) in listOf(Triple(63, 13, 1024), Triple(5, 9, 33), Triple(2, 11, 8), Triple(1, 6, 1023))) {
            val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
            val full = FloatArray(m * n)
            NativeFp32MatmulWtKernel.matmulWt(x, 0, w, 0, full, 0, m, n, k)
            val split = FloatArray(m * n)
            val cut = 3
            val left = FloatArray(m * cut)
            val right = FloatArray(m * (n - cut))
            NativeFp32MatmulWtKernel.matmulWt(x, 0, w, 0, left, 0, m, cut, k)
            NativeFp32MatmulWtKernel.matmulWt(x, 0, w, cut * k, right, 0, m, n - cut, k)
            for (i in 0 until m) {
                System.arraycopy(left, i * cut, split, i * n, cut)
                System.arraycopy(right, i * (n - cut), split, i * n + cut, n - cut)
            }
            assertContentEquals(full, split, "m=$m n=$n k=$k: a slice boundary must not change a bit")
            // One activation row at a time against all rows at once: the 4×1 and single-row
            // variants against the 4×2 tile.
            val rows = FloatArray(m * n)
            for (i in 0 until m) NativeFp32MatmulWtKernel.matmulWt(x, i * k, w, 0, rows, i * n, 1, n, k)
            assertContentEquals(full, rows, "m=$m n=$n k=$k: a row-at-a-time call must not change a bit")
        }
    }

    /**
     * `-Dskainet.runBench=true`: one native call per shape through the segment face (the weight
     * resident off-heap, as the ops layer holds it), so the C row pool is the only threading and
     * no weight copy is timed.
     */
    @Test
    fun singleCallBench() {
        if (System.getProperty("skainet.runBench") != "true") return
        val shapes = listOf(
            Triple(63, 1024, 1024), Triple(63, 2624, 1024), Triple(63, 1024, 2624),
            Triple(183, 1024, 1024), Triple(1, 1024, 1024), Triple(63, 256, 1024),
        )
        println("[wt-x86] host ${System.getProperty("os.name")} ${System.getProperty("os.arch")} JDK ${System.getProperty("java.version")}")
        Arena.ofConfined().use { arena ->
            for ((m, n, k) in shapes) {
                val rng = Random(m * 31 + n)
                val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
                val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
                val wSeg: MemorySegment = arena.allocate(w.size.toLong() * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())
                MemorySegment.copy(w, 0, wSeg, ValueLayout.JAVA_FLOAT, 0L, w.size)
                val out = FloatArray(m * n)
                val flop = 2.0 * m * n * k
                val reps = maxOf(3, (2e9 / flop).toInt().coerceAtMost(400))
                val call = { NativeFp32MatmulWtKernel.matmulWt(x, 0, wSeg, 0L, out, 0, m, n, k) }
                repeat(reps / 2 + 1) { call() }
                val samples = LongArray(5)
                for (s in samples.indices) {
                    val t0 = System.nanoTime()
                    repeat(reps) { call() }
                    samples[s] = (System.nanoTime() - t0) / reps
                }
                val ns = samples.sorted()[2]
                println("[wt-x86] m=$m n=$n k=$k: %.3f ms/call %.1f GFLOP/s".format(ns / 1e6, flop / ns))
            }
        }
    }
}
