package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.kernel.NativeFp32MatmulWtKernel
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * The dense `x · Wᵀ` product through the native direct kernel under the context schedule
 * (SKEEP-005): the activation rows are split across tasks, one native call each, and the result
 * is bit-identical to the sequential schedule and to a plain row-dot reference. Shapes cross the
 * kernel's own row-pool threshold (512 output columns) and stay under it, and include the
 * single-row decode shape that has nothing to split.
 */
class DenseWeightTransposedScheduleTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(NativeFp32MatmulWtKernel.isAvailable(), "bundled libskainet_kernels must resolve skainet_fp32_matmul_wt")
        // The direct path lives on the Vector API tier; without it the common ops serve and this test would check nothing.
        assertTrue(DirectCpuExecutionContext().ops::class.simpleName == "DefaultCpuOpsJvm", "the test JVM must run with --add-modules jdk.incubator.vector")
    }

    private fun fill(size: Int, seed: Int) = FloatArray(size) { i -> (((i * 31L + seed * 17L) % 23L) - 11L) / 11f }

    private fun product(ctx: ExecutionContext, m: Int, n: Int, k: Int): FloatArray {
        val x: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(m, k), FP32::class, fill(m * k, 1))
        val w: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(n, k), FP32::class, fill(n * k, 2))
        val out = ctx.ops.matmulWeightTransposed(x, w)
        assertContentEquals(intArrayOf(m, n), out.shape.dimensions)
        return out.data.copyToFloatArray()
    }

    private fun reference(m: Int, n: Int, k: Int): FloatArray {
        val x = fill(m * k, 1)
        val w = fill(n * k, 2)
        val out = FloatArray(m * n)
        for (i in 0 until m) for (o in 0 until n) {
            var acc = 0.0
            for (p in 0 until k) acc += x[i * k + p].toDouble() * w[o * k + p]
            out[i * n + o] = acc.toFloat()
        }
        return out
    }

    @Test
    fun scheduledRowsMatchSequentialBitForBit() {
        val sequential = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val parallel = DirectCpuExecutionContext(schedule = CoroutineSchedule(parallelism = 4))
        for ((m, n, k) in listOf(Triple(63, 1024, 1024), Triple(183, 640, 1024), Triple(5, 256, 64), Triple(2, 1024, 32), Triple(1, 1024, 1024))) {
            repeat(2) {
                val a = product(sequential, m, n, k)
                val b = product(parallel, m, n, k)
                assertContentEquals(a, b, "m=$m n=$n k=$k: the schedule must not change a single bit")
                val ref = reference(m, n, k)
                val tol = 1e-5f * k
                for (i in ref.indices) assertTrue(abs(ref[i] - a[i]) <= tol * maxOf(1f, abs(ref[i])), "m=$m n=$n k=$k element $i: ${a[i]} vs reference ${ref[i]}")
            }
        }
    }

    /**
     * Opt-in timing of the split: `-Dskainet.runBench=true`. Encoder-sized products (a 63- and a
     * 183-token sequence against ModernBERT-large's 1024-wide projections and its 2624-wide
     * feed-forward) on the sequential schedule and on every core.
     */
    @Test
    fun scheduledRowsBench() {
        if (System.getProperty("skainet.runBench") != "true") {
            println("DenseWeightTransposedScheduleTest.scheduledRowsBench skipped — pass -Dskainet.runBench=true to enable.")
            return
        }
        val hardware = CoroutineSchedule.hardware()
        val contexts = listOf("sequential" to DirectCpuExecutionContext(schedule = Schedule.Sequential), hardware.name to DirectCpuExecutionContext(schedule = hardware))
        for ((m, n, k) in listOf(Triple(63, 1024, 1024), Triple(63, 2624, 1024), Triple(63, 1024, 2624), Triple(183, 1024, 1024), Triple(1, 1024, 1024))) {
            for ((label, ctx) in contexts) {
                val x: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(m, k), FP32::class, fill(m * k, 1))
                val w: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(n, k), FP32::class, fill(n * k, 2))
                repeat(3) { ctx.ops.matmulWeightTransposed(x, w) }
                val reps = 10
                val t0 = System.nanoTime()
                repeat(reps) { ctx.ops.matmulWeightTransposed(x, w) }
                val perCall = (System.nanoTime() - t0) / 1e6 / reps
                val gflops = 2.0 * m * n * k / (perCall * 1e6)
                println("[wt-schedule] m=$m n=$n k=$k $label: ${"%.3f".format(perCall)} ms/call  ${"%.1f".format(gflops)} GFLOP/s")
            }
        }
    }
}
