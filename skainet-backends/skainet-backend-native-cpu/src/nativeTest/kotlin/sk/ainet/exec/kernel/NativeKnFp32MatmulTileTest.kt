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
import sk.ainet.kernels.cinterop.skainet_fp32_matmul

/**
 * The tails, the strides and the placement independence of the C `skainet_fp32_matmul`, through
 * cinterop so the test runs on every Kotlin/Native target that embeds the archive — on macosArm64
 * (CI) and linuxArm64 that is the NEON lane: an 8×8 tile of lane-indexed FMAs over a packed
 * 8-column panel in a 256-deep k-block, 4×8 and 1×8 tiles for the rows left over and a zero-padded
 * panel plus C stripe for the columns left over. Shapes put `m` under, at and past a tile, `n`
 * around eight, `k` around the k-block and off a multiple of four (the lane-indexed body takes four
 * `p` at a time). Every element takes one fused step per `p` in ascending `p` whichever tile holds
 * it, so a block of the output computed on its own is checked bit-for-bit against the full product.
 * On linuxX64 the same tests exercise the AVX2 lane, on a host without it the portable loop.
 */
@OptIn(ExperimentalForeignApi::class)
class NativeKnFp32MatmulTileTest {

    private fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        c: FloatArray, cOffset: Int, cStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                c.usePinned { cPin ->
                    skainet_fp32_matmul(
                        aPin.addressOf(0), aOffset, aStride,
                        bPin.addressOf(0), bOffset, bStride,
                        cPin.addressOf(0), cOffset, cStride,
                        m, n, k,
                    )
                }
            }
        }
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

    private fun block(c: FloatArray, cOffset: Int, cStride: Int, m: Int, n: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) c.copyInto(out, i * n, cOffset + i * cStride, cOffset + i * cStride + n)
        return out
    }

    @Test
    fun everyTileEdgeMatchesTheDoubleReference() {
        val rng = Random(41)
        for (m in listOf(1, 3, 4, 5, 8, 9, 13, 17)) for (n in listOf(1, 7, 8, 9, 15, 16, 17, 61)) for (k in listOf(1, 3, 255, 257)) {
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val out = FloatArray(m * n)
            matmul(a, 0, k, b, 0, n, out, 0, n, m, n, k)
            assertClose(reference(a, 0, k, b, 0, n, m, n, k), out, "m=$m n=$n k=$k")
        }
    }

    @Test
    fun stridedOperandsAndAnOffsetOutputLeaveThePaddingAlone() {
        val rng = Random(42)
        for ((m, n, k) in listOf(Triple(17, 21, 300), Triple(8, 8, 256), Triple(9, 9, 257), Triple(1, 1, 1))) {
            val aStride = k + 5; val bStride = n + 3; val cStride = n + 7
            val aOffset = 2; val bOffset = 1; val cOffset = 3
            val a = FloatArray(aOffset + m * aStride) { rng.nextFloat() - 0.5f }
            val b = FloatArray(bOffset + k * bStride) { rng.nextFloat() - 0.5f }
            val c = FloatArray(cOffset + m * cStride) { 123f }
            matmul(a, aOffset, aStride, b, bOffset, bStride, c, cOffset, cStride, m, n, k)
            assertClose(reference(a, aOffset, aStride, b, bOffset, bStride, m, n, k), block(c, cOffset, cStride, m, n), "m=$m n=$n k=$k")
            for (idx in 0 until cOffset) assertEquals(123f, c[idx], "prefix $idx must be untouched")
            for (i in 0 until m) for (j in n until cStride) {
                if (cOffset + i * cStride + j < c.size) assertEquals(123f, c[cOffset + i * cStride + j], "pad [$i, $j] must be untouched")
            }
        }
    }

    @Test
    fun anOutputDoesNotDependOnTheBlockThatComputesIt() {
        val rng = Random(43)
        // 17 rows: two 8-row tiles and a 1-row tile with packing on; the partial calls move every
        // output to another place inside a tile (5 and 3 are not multiples of 8), switch packing
        // off (fewer than two tiles) and change which tile variant computes it.
        for ((m, n, k) in listOf(Triple(17, 21, 300), Triple(9, 19, 1023), Triple(2, 10, 6), Triple(18, 1030, 64))) {
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val full = FloatArray(m * n)
            matmul(a, 0, k, b, 0, n, full, 0, n, m, n, k)

            val byRows = FloatArray(m * n)
            val cut = minOf(5, m - 1)
            matmul(a, 0, k, b, 0, n, byRows, 0, n, cut, n, k)
            matmul(a, cut * k, k, b, 0, n, byRows, cut * n, n, m - cut, n, k)
            assertContentEquals(full, byRows, "m=$m n=$n k=$k: a row boundary must not change a bit")

            val byColumns = FloatArray(m * n)
            val col = 3
            matmul(a, 0, k, b, 0, n, byColumns, 0, n, m, col, k)
            matmul(a, 0, k, b, col, n, byColumns, col, n, m, n - col, k)
            assertContentEquals(full, byColumns, "m=$m n=$n k=$k: a column boundary must not change a bit")

            val rows = FloatArray(m * n)
            for (i in 0 until m) matmul(a, i * k, k, b, 0, n, rows, i * n, n, 1, n, k)
            assertContentEquals(full, rows, "m=$m n=$n k=$k: a row-at-a-time call must not change a bit")
        }
    }
}
