package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.MemorySegmentBackedData
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * `x · Wᵀ` on the JVM ops layer with a dense `[out, in]` weight goes through the native direct
 * kernel (`skainet_fp32_matmul_wt`) when the FFM provider is loaded — heap weights after one
 * resident off-heap copy, segment-backed weights where they lie — and agrees with the
 * transpose-then-matmul result it replaces, for every activation rank a `Linear` layer sees.
 */
class JvmDirectWeightTransposedTest {

    @BeforeTest
    fun setUp() {
        KernelRegistry.clearForTesting()
        KernelServiceLoader.installAll()
        assertTrue(NativeFp32MatmulWtKernel.isAvailable(), "bundled libskainet_kernels must resolve")
    }

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String) {
        assertEquals(expected.size, actual.size, "$what size")
        var maxAbs = 0f
        for (v in expected) maxAbs = maxOf(maxAbs, abs(v))
        val tol = maxAbs * 1e-5f + 1e-6f
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${actual[i]} vs ${expected[i]} (tol $tol)")
        }
    }

    private fun check(ctx: DirectCpuExecutionContext, xShape: Shape, n: Int, seed: Int, what: String) {
        val rng = Random(seed)
        val k = xShape.dimensions.last()
        val x: Tensor<FP32, Float> = ctx.fromFloatArray(xShape, FP32::class, FloatArray(xShape.volume) { rng.nextFloat() - 0.5f })
        val w: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(n, k), FP32::class, FloatArray(n * k) { rng.nextFloat() - 0.5f })
        val got = ctx.ops.matmulWeightTransposed(x, w)
        val expectedShape = if (xShape.rank == 1) Shape(n) else Shape(*(xShape.dimensions.copyOf(xShape.rank - 1) + n))
        assertEquals(expectedShape, got.shape, "$what shape")
        val expected = ctx.ops.matmul(x, ctx.ops.transpose(w)).data.copyToFloatArray()
        assertClose(expected, got.data.copyToFloatArray(), what)
        // A second call reuses the resident weight: same answer.
        assertClose(expected, ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray(), "$what again")
    }

    @Test
    fun heap_weights_match_transpose_then_matmul_for_every_activation_rank() {
        val ctx = DirectCpuExecutionContext()
        assertEquals("native-ffm", KernelRegistry.bestAvailable()?.name, "the FFM provider must win on this runner")
        check(ctx, Shape(40), 70, 1, "rank-1")
        check(ctx, Shape(3, 40), 70, 2, "rank-2")
        check(ctx, Shape(2, 3, 40), 600, 3, "rank-3, threaded rows")
        check(ctx, Shape(1, 1), 1, 4, "1×1")
    }

    @Test
    fun segment_backed_weights_are_read_where_they_lie() {
        val ctx = DirectCpuExecutionContext(tensorDataFactory = MemorySegmentTensorDataFactory())
        val w: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(5, 8), FP32::class, FloatArray(40) { it.toFloat() })
        assertTrue(w.data is MemorySegmentBackedData, "this factory stages tensors off-heap: ${w.data::class.simpleName}")
        check(ctx, Shape(4, 32), 48, 5, "segment-backed")
        check(ctx, Shape(32), 520, 6, "segment-backed rank-1, threaded rows")
    }
}
