package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * SKEEP-007 §T0.2: `x · Wᵀ` for a dense heap FP32 `[out, in]` weight computed from the weight's own rows.
 *
 * The tiled row-dot keeps one sequential accumulator per output, so it is bit-identical to the
 * transpose-then-matmul path it replaces on the tiers that take it — and that is what these tests
 * pin, for every tile remainder (m, n not multiples of 4) and for rank-1 and rank-3 activations.
 */
class DenseWeightTransposedDirectTest {
    private val ctx = DirectCpuExecutionContext()

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 250f
        }
    }

    /** `out[i, j] = Σ_p a[i, p] · w[j, p]`, ascending p, one accumulator — the definition. */
    private fun reference(a: FloatArray, w: FloatArray, m: Int, n: Int, k: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (j in 0 until n) {
            var s = 0f
            for (p in 0 until k) s += a[i * k + p] * w[j * k + p]
            out[i * n + j] = s
        }
        return out
    }

    @Test
    fun `tiled row-dot is bit-identical to the sequential definition for every tile remainder`() {
        for ((m, n, k) in listOf(Triple(1, 1, 1), Triple(4, 4, 8), Triple(5, 7, 33), Triple(9, 3, 64), Triple(2, 13, 5), Triple(8, 8, 0))) {
            val a = pseudo(m * k, 7 + m)
            val w = pseudo(n * k, 11 + n)
            val out = FloatArray(m * n) { Float.NaN }
            denseRowDotTiled(a, 0, w, 0, out, m, n, k)
            assertContentEquals(reference(a, w, m, n, k), out, "m=$m n=$n k=$k")
        }
    }

    @Test
    fun `tiled row-dot honours array offsets`() {
        val m = 6; val n = 5; val k = 17
        val aPad = 3; val wPad = 5
        val a = pseudo(m * k, 3)
        val w = pseudo(n * k, 5)
        val aShifted = FloatArray(aPad) { 99f } + a
        val wShifted = FloatArray(wPad) { -99f } + w
        val out = FloatArray(m * n)
        denseRowDotTiled(aShifted, aPad, wShifted, wPad, out, m, n, k)
        assertContentEquals(reference(a, w, m, n, k), out)
    }

    @Test
    fun `the primitive agrees with transpose then matmul for rank-1 rank-2 and rank-3 activations`() {
        val n = 7; val k = 12
        val w = ctx.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, pseudo(n * k, 21))
        val shapes = listOf(Shape(k), Shape(5, k), Shape(2, 3, k))
        for (shape in shapes) {
            val x = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, pseudo(shape.volume, 33))
            val viaPrimitive = ctx.ops.matmulWeightTransposed(x, w)
            val viaTranspose = ctx.ops.matmul(x, ctx.ops.transpose(w))
            assertEquals(viaTranspose.shape, viaPrimitive.shape, "shape for x=$shape")
            assertContentEquals(
                viaTranspose.data.copyToFloatArray(),
                viaPrimitive.data.copyToFloatArray(),
                "values for x=$shape",
            )
        }
    }
}
