package sk.ainet.exec.kernel

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.kernels.cinterop.skainet_fp32_matmul_wt

/**
 * The tails and the slice independence of the C `skainet_fp32_matmul_wt`, through cinterop so the
 * test runs on every Kotlin/Native target that embeds the archive — on macosArm64 (CI) and
 * linuxArm64 that is the NEON lane: four weight rows by three activation rows in twelve four-lane
 * chains, a 4×1 tile for the activation rows left over and a single-row dot for the weight rows
 * left over. Shapes put `k` off a multiple of four, `n` off a multiple of four, `m` at one, two,
 * three and past a tile. Every variant reduces an output the same way (lanes, horizontal sum,
 * fused scalar tail), so a slice of the weight's rows computed on its own, and one activation row
 * at a time, are checked bit-for-bit against the whole product — what the row pool and the
 * SKEEP-005 column split rely on. On linuxX64 the same tests exercise the AVX2 lane.
 */
@OptIn(ExperimentalForeignApi::class)
class NativeKnFp32MatmulWtTileTest {

    private fun matmulWt(x: FloatArray, xOffset: Int, w: FloatArray, wOffset: Int, out: FloatArray, outOffset: Int, m: Int, n: Int, k: Int) {
        x.usePinned { xPin ->
            w.usePinned { wPin ->
                out.usePinned { oPin ->
                    skainet_fp32_matmul_wt(xPin.addressOf(0), xOffset, wPin.addressOf(0), wOffset, oPin.addressOf(0), outOffset, m, n, k)
                }
            }
        }
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
        val rng = Random(51)
        for (m in listOf(1, 2, 3, 4, 7, 20)) for (n in listOf(1, 2, 3, 4, 5, 9, 61)) for (k in listOf(1, 3, 4, 5, 255, 1023)) {
            val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
            val out = FloatArray(m * n)
            matmulWt(x, 0, w, 0, out, 0, m, n, k)
            assertClose(reference(x, w, m, n, k), out, "m=$m n=$n k=$k")
        }
    }

    @Test
    fun anOutputDoesNotDependOnTheSliceThatComputesIt() {
        val rng = Random(52)
        for ((m, n, k) in listOf(Triple(20, 13, 1023), Triple(7, 9, 33), Triple(2, 11, 6), Triple(1, 6, 1023), Triple(5, 600, 64))) {
            val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
            val full = FloatArray(m * n)
            matmulWt(x, 0, w, 0, full, 0, m, n, k)

            // The weight's rows split at 3: every output lands at another offset inside a 4-row tile.
            val cut = 3
            val left = FloatArray(m * cut)
            val right = FloatArray(m * (n - cut))
            matmulWt(x, 0, w, 0, left, 0, m, cut, k)
            matmulWt(x, 0, w, cut * k, right, 0, m, n - cut, k)
            val split = FloatArray(m * n)
            for (i in 0 until m) {
                left.copyInto(split, i * n, i * cut, i * cut + cut)
                right.copyInto(split, i * n + cut, i * (n - cut), i * (n - cut) + (n - cut))
            }
            assertContentEquals(full, split, "m=$m n=$n k=$k: a slice boundary must not change a bit")

            // One activation row at a time: the 4×1 tile and the single-row dot against the 4×3 tile.
            val rows = FloatArray(m * n)
            for (i in 0 until m) matmulWt(x, i * k, w, 0, rows, i * n, 1, n, k)
            assertContentEquals(full, rows, "m=$m n=$n k=$k: a row-at-a-time call must not change a bit")
        }
    }
}
