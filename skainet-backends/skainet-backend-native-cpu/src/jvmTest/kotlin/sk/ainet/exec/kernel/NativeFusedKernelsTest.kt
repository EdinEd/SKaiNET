package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.AttentionMask
import sk.ainet.lang.tensor.ops.FusedNormOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * The fused row kernels on the JVM ops layer (`f32_fused.c` through FFM): softmax / log-softmax,
 * the normalisations behind `LayerNormalization` and the `FusedNormOps` hook, and scaled
 * dot-product attention with every mask rule — all routed to the native kernels when the FFM
 * provider is loaded, and agreeing with the common scalar implementation within the rounding of a
 * vectorised accumulation. Routing is proved by bit-identity with the kernel called directly.
 */
class NativeFusedKernelsTest {

    @BeforeTest
    fun setUp() {
        KernelRegistry.clearForTesting()
        KernelServiceLoader.installAll()
        assertTrue(NativeFp32SoftmaxKernel.isAvailable() && NativeFp32NormKernel.isAvailable() && NativeFp32AttentionKernel.isAvailable(),
            "bundled libskainet_kernels must resolve the fused kernels")
    }

    @AfterTest
    fun tearDown() = KernelRegistry.clearForTesting()

    /** The common scalar implementation, the reference every tier is checked against. */
    private val reference = DefaultCpuOps(DenseTensorDataFactory(), Schedule.Sequential)

    private fun fill(size: Int, seed: Int, scale: Float = 1f) = FloatArray(size) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f * scale }

    private fun floats(t: Tensor<*, *>) = t.data.copyToFloatArray()

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String, rel: Float = 1e-5f, abs: Float = 1e-6f) {
        assertEquals(expected.size, actual.size, "$what size")
        var maxAbs = 0f
        for (v in expected) maxAbs = maxOf(maxAbs, abs(v))
        val tol = maxAbs * rel + abs
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${actual[i]} vs ${expected[i]} (tol $tol)")
        }
    }

    @Test
    fun softmax_and_log_softmax_run_native_and_match_the_common_loop() {
        val ctx = DirectCpuExecutionContext()
        val rows = 64; val cols = 256
        val arr = fill(rows * cols, 1, 4f)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, arr)
        val sm = floats(ctx.ops.softmax(x, -1))
        val lsm = floats(ctx.ops.logSoftmax(x, -1))
        assertClose(floats(reference.softmax(x, -1)), sm, "softmax")
        assertClose(floats(reference.logSoftmax(x, -1)), lsm, "logSoftmax", abs = 1e-5f)
        // Routed to the kernel: bit-identical to calling it directly.
        val direct = FloatArray(rows * cols)
        NativeFp32SoftmaxKernel.softmaxRows(arr, 0, direct, 0, rows, cols, log = false)
        assertContentEquals(direct, sm, "softmax went through the native kernel")
        NativeFp32SoftmaxKernel.softmaxRows(arr, 0, direct, 0, rows, cols, log = true)
        assertContentEquals(direct, lsm, "logSoftmax went through the native kernel")
        // Every row sums to one.
        for (r in 0 until rows) {
            var s = 0.0
            for (c in 0 until cols) s += sm[r * cols + c]
            assertTrue(abs(s - 1.0) < 1e-4, "row $r sums to $s")
        }
        // A small tensor and a non-last dim keep the common loop; the answer is the same.
        val small = ctx.fromFloatArray<FP32, Float>(Shape(4, 8), FP32::class, fill(32, 2))
        assertClose(floats(reference.softmax(small, -1)), floats(ctx.ops.softmax(small, -1)), "small softmax")
        assertClose(floats(reference.softmax(x, 0)), floats(ctx.ops.softmax(x, 0)), "softmax over dim 0")
    }

    @Test
    fun layer_norm_runs_fused_through_the_module_and_matches_the_decomposition() {
        val ctx = DirectCpuExecutionContext()
        assertTrue(ctx.ops is FusedNormOps, "the JVM ops carry the fused-norm hook")
        val rows = 6; val cols = 768
        val xArr = fill(rows * cols, 3, 2f)
        val gArr = FloatArray(cols) { 0.5f + (it % 7) * 0.1f }
        val bArr = FloatArray(cols) { ((it % 5) - 2) * 0.05f }
        val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, xArr)
        val gamma = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, gArr)
        val beta = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, bArr)
        val ln = LayerNormalization<FP32, Float>(intArrayOf(cols), eps = 1e-5, elementwiseAffine = true, name = "ln", initGamma = gamma, initBeta = beta)
        val got = floats(ln.forward(x, ctx))
        // The decomposition's arithmetic, in double.
        val ref = FloatArray(rows * cols)
        for (r in 0 until rows) {
            var mean = 0.0
            for (c in 0 until cols) mean += xArr[r * cols + c]
            mean /= cols
            var v = 0.0
            for (c in 0 until cols) { val d = xArr[r * cols + c] - mean; v += d * d }
            v /= cols
            val inv = 1.0 / sqrt(v + 1e-5)
            for (c in 0 until cols) ref[r * cols + c] = ((xArr[r * cols + c] - mean) * inv * gArr[c] + bArr[c]).toFloat()
        }
        assertClose(ref, got, "layerNorm", rel = 1e-4f, abs = 1e-4f)
        // The hook answered, not the decomposition: bit-identical to the kernel called directly.
        val direct = FloatArray(rows * cols)
        NativeFp32NormKernel.layerNorm(xArr, 0, gArr, bArr, direct, 0, rows, cols, 1e-5f)
        assertContentEquals(direct, got, "layerNorm went through the native kernel")

        // RMS norm through the hook.
        val rms = floats(checkNotNull((ctx.ops as FusedNormOps).rmsNorm(x, gamma, 1e-6f, 1)) { "rmsNorm must be served" })
        val rmsRef = FloatArray(rows * cols)
        for (r in 0 until rows) {
            var ms = 0.0
            for (c in 0 until cols) { val d = xArr[r * cols + c].toDouble(); ms += d * d }
            ms /= cols
            val inv = 1.0 / sqrt(ms + 1e-6)
            for (c in 0 until cols) rmsRef[r * cols + c] = (xArr[r * cols + c] * inv * gArr[c]).toFloat()
        }
        assertClose(rmsRef, rms, "rmsNorm", rel = 1e-4f, abs = 1e-4f)

        // Two trailing dims and no affine: one row of 32 per [2, 3] prefix.
        val x3Arr = fill(2 * 3 * 4 * 8, 5)
        val x3 = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 4, 8), FP32::class, x3Arr)
        val ln2 = LayerNormalization<FP32, Float>(intArrayOf(4, 8), eps = 1e-5, elementwiseAffine = false, name = "ln2")
        val got3 = floats(ln2.forward(x3, ctx))
        val ref3 = FloatArray(x3Arr.size)
        for (r in 0 until 6) {
            var mean = 0.0
            for (c in 0 until 32) mean += x3Arr[r * 32 + c]
            mean /= 32
            var v = 0.0
            for (c in 0 until 32) { val d = x3Arr[r * 32 + c] - mean; v += d * d }
            v /= 32
            val inv = 1.0 / sqrt(v + 1e-5)
            for (c in 0 until 32) ref3[r * 32 + c] = ((x3Arr[r * 32 + c] - mean) * inv).toFloat()
        }
        assertClose(ref3, got3, "layerNorm over two dims", rel = 1e-4f, abs = 1e-4f)
    }

    private class Inputs(val q: Tensor<FP32, Float>, val k: Tensor<FP32, Float>, val v: Tensor<FP32, Float>)

    private fun inputs(ctx: DirectCpuExecutionContext, batch: Int, heads: Int, kvHeads: Int, seqQ: Int, seqKV: Int, headDim: Int, seed: Int = 0) = Inputs(
        ctx.fromFloatArray(Shape(batch, heads, seqQ, headDim), FP32::class, fill(batch * heads * seqQ * headDim, 1 + seed)),
        ctx.fromFloatArray(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 2 + seed)),
        ctx.fromFloatArray(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 3 + seed)),
    )

    @Suppress("UNCHECKED_CAST")
    private fun lengthsTensor(ctx: DirectCpuExecutionContext, lengths: IntArray): Tensor<DType, *> =
        ctx.fromIntArray<Int32, Int>(Shape(lengths.size), Int32::class, lengths) as Tensor<DType, *>

    @Test
    fun attention_runs_native_with_every_mask_rule_and_matches_the_common_kernel() {
        val ctx = DirectCpuExecutionContext()
        val batch = 3
        for ((heads, kvHeads) in listOf(4 to 4, 4 to 2)) for ((seqQ, seqKV) in listOf(6 to 6, 3 to 10, 1 to 7)) for (causal in listOf(false, true)) {
            val x = inputs(ctx, batch, heads, kvHeads, seqQ, seqKV, 8)
            for (lengths in listOf(null, intArrayOf(seqKV, (seqKV + 1) / 2, 1))) for ((window, rightContext) in listOf(0 to 0, 3 to 0, 4 to 2)) {
                val what = "heads=$heads kv=$kvHeads seqQ=$seqQ seqKV=$seqKV causal=$causal lengths=${lengths?.toList()} window=$window rc=$rightContext"
                val kl = lengths?.let { lengthsTensor(ctx, it) }
                val got = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = null, scale = 0f, causal = causal, keyLengths = kl, window = window, rightContext = rightContext)
                assertEquals(Shape(batch, heads, seqQ, 8), got.shape, what)
                val want = reference.scaledDotProductAttention(x.q, x.k, x.v, mask = null, scale = 0f, causal = causal, keyLengths = kl, window = window, rightContext = rightContext)
                assertClose(floats(want), floats(got), what, rel = 1e-4f, abs = 1e-5f)
                // The structured rules equal the dense -1e30 mask on the fused kernel too.
                val dense = ctx.fromFloatArray<FP32, Float>(Shape(batch, 1, seqQ, seqKV), FP32::class, AttentionMask.additive(batch, seqQ, seqKV, causal, window, rightContext, lengths))
                val masked = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = dense, scale = 0f, causal = false)
                assertClose(floats(got), floats(masked), "$what vs dense mask", rel = 1e-5f, abs = 1e-6f)
            }
        }
        // Mask broadcast shapes: a per-key bias and a [seqQ, seqKV] matrix.
        val x = inputs(ctx, 2, 4, 2, 5, 9, 16, seed = 4)
        for (mask in listOf(
            ctx.fromFloatArray<FP32, Float>(Shape(1, 1, 1, 9), FP32::class, fill(9, 6)),
            ctx.fromFloatArray<FP32, Float>(Shape(5, 9), FP32::class, fill(45, 7)),
            ctx.fromFloatArray<FP32, Float>(Shape(2, 4, 5, 9), FP32::class, fill(2 * 4 * 5 * 9, 8)),
        )) {
            val got = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = mask, scale = 0.3f, causal = true)
            val want = reference.scaledDotProductAttention(x.q, x.k, x.v, mask = mask, scale = 0.3f, causal = true)
            assertClose(floats(want), floats(got), "mask ${mask.shape}", rel = 1e-4f, abs = 1e-5f)
        }
    }

    @Test
    fun fully_padded_row_stays_finite_and_a_large_call_threads_in_the_native_pool() {
        val ctx = DirectCpuExecutionContext()
        val p = inputs(ctx, 2, 2, 2, 3, 5, 4)
        val out = floats(ctx.ops.scaledDotProductAttention(p.q, p.k, p.v, keyLengths = lengthsTensor(ctx, intArrayOf(0, 5))))
        assertTrue(out.all { it.isFinite() }, "fully padded row produced a non-finite value")
        val dense = ctx.fromFloatArray<FP32, Float>(Shape(2, 1, 3, 5), FP32::class, AttentionMask.additive(2, 3, 5, false, 0, 0, intArrayOf(0, 5)))
        assertClose(floats(reference.scaledDotProductAttention(p.q, p.k, p.v, mask = dense)), out, "fully padded", rel = 1e-5f, abs = 1e-6f)

        // Laya's attention shape: 12 heads × 150 queries = 1800 rows, above the row pool's threading threshold.
        val x = inputs(ctx, 1, 12, 12, 150, 150, 64, seed = 9)
        val got = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, causal = true)
        val want = reference.scaledDotProductAttention(x.q, x.k, x.v, causal = true)
        assertClose(floats(want), floats(got), "laya shape", rel = 1e-4f, abs = 1e-5f)
        // Routed to the kernel: bit-identical to calling it directly.
        val direct = FloatArray(12 * 150 * 64)
        NativeFp32AttentionKernel.sdpa(
            floats(x.q), floats(x.k), floats(x.v), direct,
            1, 12, 12, 150, 150, 64, (1.0 / sqrt(64.0)).toFloat(), true, 0, 0, null, null, null,
        )
        assertContentEquals(direct, floats(got), "attention went through the native kernel")
    }
}
