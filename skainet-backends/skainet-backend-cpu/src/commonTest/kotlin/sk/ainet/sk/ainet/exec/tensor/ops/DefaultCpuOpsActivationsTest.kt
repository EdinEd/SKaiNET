package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.exec.tensor.ops.DefaultCpuOps

class DefaultCpuOpsActivationsTest {
    private val dataFactory = DenseTensorDataFactory()
    private val cpuOps = DefaultCpuOps(dataFactory)

    // Helpers to build input tensors with data (mirror existing tests style)
    private fun fTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    private fun iTensor(shape: Shape, values: IntArray): VoidOpsTensor<Int32, Int> {
        val data = dataFactory.fromIntArray<Int32, Int>(shape, Int32::class, values)
        return VoidOpsTensor(data, Int32::class)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + kotlin.math.exp(-x))
    private fun silu(x: Float): Float = x * sigmoid(x)
    private fun tanh(x: Float): Float = kotlin.math.tanh(x)

    private fun assertAlmostEquals(expected: Float, actual: Float, eps: Float = 1e-5f, msg: String = "") {
        assertTrue(kotlin.math.abs(expected - actual) <= eps, msg.ifEmpty { "Expected $expected, got $actual" })
    }

    @Test
    fun sigmoid_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.sigmoid(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { sigmoid(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "sigmoid at $i mismatch")
        }
    }

    @Test
    fun sigmoid_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 3), floatArrayOf(
            -1f, 0f, 1f,
            2f, -2f, 0.5f
        ))
        val out = cpuOps.sigmoid(input)
        assertEquals(Shape(2, 3), out.shape)
        // spot-check a couple values
        assertAlmostEquals(sigmoid(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(sigmoid(0.5f), out.data[1, 2] as Float, 1e-6f)
    }

    @Test
    fun silu_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.silu(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { silu(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "silu at $i mismatch")
        }
    }

    @Test
    fun silu_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 2), floatArrayOf(-1f, 0f, 1f, 2f))
        val out = cpuOps.silu(input)
        assertEquals(Shape(2, 2), out.shape)
        assertAlmostEquals(silu(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(silu(2f), out.data[1, 1] as Float, 1e-6f)
    }

    @Test
    fun sigmoid_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(3), intArrayOf(1, 2, 3))
        var threw = false
        try {
            cpuOps.sigmoid(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            // Optional: ensure message mentions unsupported dtype
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 sigmoid")
    }

    @Test
    fun silu_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(2), intArrayOf(1, 2))
        var threw = false
        try {
            cpuOps.silu(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 silu")
    }

    @Test
    fun tanh_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.tanh(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { tanh(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "tanh at $i mismatch")
        }
    }

    @Test
    fun tanh_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 3), floatArrayOf(
            -1f, 0f, 1f,
            2f, -2f, 0.5f
        ))
        val out = cpuOps.tanh(input)
        assertEquals(Shape(2, 3), out.shape)
        assertAlmostEquals(tanh(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(tanh(0.5f), out.data[1, 2] as Float, 1e-6f)
    }

    @Test
    fun tanh_fp32_saturates_at_extremes() {
        // tanh(x) → 1 as x → +∞, → -1 as x → -∞. kotlin.math.tanh is numerically stable.
        val input = fTensor(Shape(2), floatArrayOf(-100f, 100f))
        val out = cpuOps.tanh(input)
        assertAlmostEquals(-1f, out.data[0] as Float, 1e-6f)
        assertAlmostEquals(1f, out.data[1] as Float, 1e-6f)
    }

    @Test
    fun tanh_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(2), intArrayOf(1, 2))
        var threw = false
        try {
            cpuOps.tanh(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 tanh")
    }

    // ---- erf / exact GELU (#1342) ----------------------------------------

    /** Abramowitz–Stegun 7.1.26 in double, the host reference the kernel must reproduce. */
    private fun erfRef(x: Double): Double {
        val sign = if (x < 0) -1.0 else 1.0
        val a = kotlin.math.abs(x)
        val t = 1.0 / (1.0 + 0.3275911 * a)
        val poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
        return sign * (1.0 - poly * t * kotlin.math.exp(-a * a))
    }

    private fun geluExactRef(x: Float): Float {
        val xd = x.toDouble()
        return (0.5 * xd * (1.0 + erfRef(xd / kotlin.math.sqrt(2.0)))).toFloat()
    }

    private fun geluTanhRef(x: Float): Float {
        val inner = x + 0.044715f * (x * x * x)
        return 0.5f * x * (1.0f + kotlin.math.tanh(0.7978845608f * inner))
    }

    @Test
    fun erf_matches_reference_table() {
        val xs = floatArrayOf(0f, 0.5f, -0.5f, 1f, -1f, 1.5f, -1.5f, 2f, -2f, 3f, -3f, 4f, -4f, 6f, -6f)
        val table = floatArrayOf(
            0f, 0.5204998778f, -0.5204998778f, 0.8427007929f, -0.8427007929f, 0.9661051465f, -0.9661051465f,
            0.9953222650f, -0.9953222650f, 0.9999779095f, -0.9999779095f, 0.9999999846f, -0.9999999846f, 1f, -1f,
        )
        val out = cpuOps.erf(fTensor(Shape(xs.size), xs))
        assertEquals(Shape(xs.size), out.shape)
        assertEquals(FP32::class, out.dtype)
        for (i in xs.indices) {
            assertAlmostEquals(table[i], out.data[i] as Float, 2e-7f, "erf(${xs[i]}) mismatch: ${out.data[i]}")
        }
    }

    @Test
    fun erf_is_odd_and_saturates() {
        val xs = FloatArray(41) { -4f + it * 0.2f }
        val pos = cpuOps.erf(fTensor(Shape(xs.size), xs))
        val neg = cpuOps.erf(fTensor(Shape(xs.size), FloatArray(xs.size) { -xs[it] }))
        for (i in xs.indices) {
            assertAlmostEquals(-(pos.data[i] as Float), neg.data[i] as Float, 0f, "erf must be odd at x=${xs[i]}")
        }
        val sat = cpuOps.erf(fTensor(Shape(2), floatArrayOf(6f, -6f)))
        assertEquals(1f, sat.data[0] as Float)
        assertEquals(-1f, sat.data[1] as Float)
    }

    @Test
    fun gelu_exact_matches_host_formula() {
        val xs = FloatArray(121) { -6f + it * 0.1f }
        val out = cpuOps.gelu(fTensor(Shape(xs.size), xs), approximate = false)
        for (i in xs.indices) {
            assertEquals(geluExactRef(xs[i]), out.data[i] as Float, "exact gelu at x=${xs[i]}")
        }
    }

    @Test
    fun gelu_default_is_unchanged_tanh() {
        val xs = floatArrayOf(-3f, -2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 3f)
        val t = fTensor(Shape(xs.size), xs)
        val byDefault = cpuOps.gelu(t)
        val explicit = cpuOps.gelu(t, approximate = true)
        for (i in xs.indices) {
            assertEquals(byDefault.data[i] as Float, explicit.data[i] as Float, "default must equal approximate=true")
            assertAlmostEquals(geluTanhRef(xs[i]), byDefault.data[i] as Float, 1e-6f, "tanh gelu at x=${xs[i]}")
        }
    }

    @Test
    fun gelu_exact_vs_tanh_max_difference() {
        val xs = FloatArray(1201) { -6f + it * 0.01f }
        val t = fTensor(Shape(xs.size), xs)
        val exact = cpuOps.gelu(t, approximate = false)
        val tanh = cpuOps.gelu(t, approximate = true)
        var maxDiff = 0f
        var argMax = 0f
        for (i in xs.indices) {
            val d = kotlin.math.abs((exact.data[i] as Float) - (tanh.data[i] as Float))
            if (d > maxDiff) { maxDiff = d; argMax = xs[i] }
        }
        // The tanh approximation deviates from the exact form by ≈ 4.7e-4 near |x| ≈ 2.7.
        assertTrue(maxDiff > 4.5e-4f && maxDiff < 5.0e-4f, "max |exact - tanh| = $maxDiff at x=$argMax")
        assertTrue(kotlin.math.abs(argMax) in 2.5f..2.9f, "arg max at x=$argMax")
    }

    @Test
    fun erf_and_exact_gelu_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(2), intArrayOf(1, 2))
        @Suppress("UNCHECKED_CAST")
        val asInt = input as sk.ainet.lang.tensor.Tensor<Int32, Int>
        var threw = 0
        try { cpuOps.erf(asInt) } catch (e: IllegalArgumentException) { threw++; assertTrue(e.message?.contains("Unsupported dtype for erf") == true) }
        try { cpuOps.gelu(asInt, approximate = false) } catch (e: IllegalArgumentException) { threw++; assertTrue(e.message?.contains("Unsupported dtype for gelu: class sk.ainet.lang.types.Int32") == true, e.message) }
        assertEquals(2, threw)
    }
}
