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
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * [NativeFp32MatmulWtKernel] — the JVM binding of `skainet_fp32_matmul_wt` — against a plain
 * row-dot reference, on both of its faces. The row-pool threshold (512 output rows) is crossed by
 * one shape so the threaded partition is covered; per-row arithmetic does not depend on it.
 */
class NativeFp32MatmulWtKernelTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeFp32MatmulWtKernel.isAvailable(), "bundled libskainet_kernels must resolve skainet_fp32_matmul_wt")
    }

    private fun reference(x: FloatArray, xOff: Int, w: FloatArray, wOff: Int, m: Int, n: Int, k: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (o in 0 until n) {
            var acc = 0.0
            for (p in 0 until k) acc += x[xOff + i * k + p].toDouble() * w[wOff + o * k + p]
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
    fun heap_face_matches_the_row_dot_reference_on_every_shape() {
        val rng = Random(3)
        // (m, n, k): a single token; a prefill against a threaded row count with a 4-lane tail; one row of weight.
        for ((m, n, k) in listOf(Triple(1, 7, 5), Triple(3, 600, 37), Triple(5, 1, 64))) {
            val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
            val out = FloatArray(m * n)
            NativeFp32MatmulWtKernel.matmulWt(x, 0, w, 0, out, 0, m, n, k)
            assertClose(reference(x, 0, w, 0, m, n, k), out, "wt $m×$k · ($n×$k)ᵀ")
        }
    }

    @Test
    fun offsets_are_honoured_and_the_rest_of_the_arrays_is_left_alone() {
        val rng = Random(5)
        val m = 2; val n = 9; val k = 12
        val x = FloatArray(3 + m * k) { rng.nextFloat() }
        val w = FloatArray(2 + n * k) { rng.nextFloat() }
        val out = FloatArray(1 + m * n + 4) { 7f }
        NativeFp32MatmulWtKernel.matmulWt(x, 3, w, 2, out, 1, m, n, k)
        assertClose(reference(x, 3, w, 2, m, n, k), out.copyOfRange(1, 1 + m * n), "offset wt")
        assertEquals(7f, out[0])
        for (i in 1 + m * n until out.size) assertEquals(7f, out[i], "tail[$i] untouched")
    }

    @Test
    fun k_zero_zeroes_the_block_and_empty_shapes_are_no_ops() {
        val out = FloatArray(6) { 1f }
        NativeFp32MatmulWtKernel.matmulWt(FloatArray(0), 0, FloatArray(0), 0, out, 0, 2, 3, 0)
        assertContentEquals(FloatArray(6), out)
        val untouched = FloatArray(4) { 2f }
        NativeFp32MatmulWtKernel.matmulWt(FloatArray(0), 0, FloatArray(8), 0, untouched, 0, 0, 4, 2)
        NativeFp32MatmulWtKernel.matmulWt(FloatArray(4), 0, FloatArray(0), 0, untouched, 0, 2, 0, 2)
        assertContentEquals(FloatArray(4) { 2f }, untouched)
    }

    @Test
    fun segment_face_is_bit_identical_to_the_heap_face() {
        val rng = Random(11)
        val m = 4; val n = 530; val k = 96
        val x = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val w = FloatArray(n * k) { rng.nextFloat() - 0.5f }
        val viaHeap = FloatArray(m * n)
        NativeFp32MatmulWtKernel.matmulWt(x, 0, w, 0, viaHeap, 0, m, n, k)
        Arena.ofConfined().use { arena ->
            // The weight sits 16 bytes into a larger segment: the offset is honoured and no copy is made.
            val seg = arena.allocate(16L + n.toLong() * k * java.lang.Float.BYTES, 64L)
            MemorySegment.copy(w, 0, seg, ValueLayout.JAVA_FLOAT, 16L, n * k)
            val viaSegment = FloatArray(m * n)
            NativeFp32MatmulWtKernel.matmulWt(x, 0, seg, 16L, viaSegment, 0, m, n, k)
            assertContentEquals(viaHeap, viaSegment)
        }
        assertClose(reference(x, 0, w, 0, m, n, k), viaHeap, "threaded wt")
    }

    @Test
    fun the_provider_exposes_both_faces() {
        assertTrue(NativeKernelProvider.isAvailable())
        assertNotNull(NativeKernelProvider.matmulFp32Wt())
        assertNotNull(NativeKernelProvider.matmulFp32WtMemSeg())
        assertTrue(NativeKernelProvider.supports("matmul_wt", listOf("Float32", "Float32")))
        assertTrue(!ScalarKernelProvider.supports("matmul_wt", listOf("Float32", "Float32")), "the scalar floor has no direct kernel")
    }
}
