package sk.ainet.exec

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * #1342 acceptance: the GELU form chosen on the eager (DIRECT) path is recorded as the node
 * parameter `approximate`, and replaying the graph (OPTIMIZED) reproduces the eager output.
 */
class GeluExactRecordingTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun tensor(shape: Shape, seed: Int): VoidOpsTensor<FP32, Float> {
        val values = FloatArray(shape.volume) { i -> (((i * 37 + seed * 11) % 29) - 14) / 7f }
        return VoidOpsTensor(dataFactory.fromFloatArray(shape, FP32::class, values), FP32::class)
    }

    private fun floats(t: Tensor<*, *>): FloatArray = t.data.copyToFloatArray()

    @Test
    fun exact_gelu_is_recorded_with_its_flag_and_replays_bit_identical() {
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        val x = tensor(Shape(4, 32), 1)
        val w = tensor(Shape(32, 24), 2)
        val b = tensor(Shape(4, 24), 3)
        val w2 = tensor(Shape(24, 8), 4)

        ctx.startRecording()
        val ops = ctx.ops
        val h = ops.add(ops.matmul(x, w), b)
        val g = ops.gelu(h, approximate = false)
        val eager = ops.matmul(g, w2)
        val tape = ctx.stopRecording() as DefaultExecutionTape

        // The trace carries the flag …
        val geluTrace = tape.traces.single { it.opType == "gelu" }
        assertEquals(false, geluTrace.attributes["approximate"])

        // … and so does the graph node built from it.
        val graph = tape.toComputeGraph(synthesizeExternalInputs = true, embedConstants = false)
        val geluNode = graph.nodes.single { it.operation.name == "gelu" }
        assertEquals(false, geluNode.operation.parameters["approximate"])

        // Replay through the graph executor on the same kernels: external inputs keyed by trace id.
        val session = ctx.session
        val inputs = mapOf(
            session.refOf(x).id to x, session.refOf(w).id to w,
            session.refOf(b).id to b, session.refOf(w2).id to w2,
        )
        val outputs = ComputeGraphExecutor(graph, DefaultCpuOps(dataFactory)).execute(inputs)
        val lastMatmul = graph.nodes.filter { it.operation.name == "matmul" }.maxBy { it.id.substringAfter('n').substringBefore('_').toInt() }
        val replayed = floats(outputs.getValue(lastMatmul.id))
        val expected = floats(eager)

        assertEquals(expected.size, replayed.size)
        var maxDiff = 0f
        for (i in expected.indices) maxDiff = maxOf(maxDiff, kotlin.math.abs(expected[i] - replayed[i]))
        assertTrue(maxDiff <= 1e-6f, "DIRECT vs OPTIMIZED max |Δ| = $maxDiff")

        // Sanity: the exact form really was used — the tanh form differs on this block.
        val tanhOut = floats(DefaultCpuOps(dataFactory).gelu(h, approximate = true))
        val exactOut = floats(g)
        var formDiff = 0f
        for (i in exactOut.indices) formDiff = maxOf(formDiff, kotlin.math.abs(exactOut[i] - tanhOut[i]))
        assertTrue(formDiff > 1e-5f, "exact and tanh GELU should differ on this block, max |Δ| = $formDiff")
    }

    @Test
    fun default_gelu_records_approximate_true() {
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        val x = tensor(Shape(2, 3), 5)
        ctx.startRecording()
        ctx.ops.gelu(x)
        val tape = ctx.stopRecording() as DefaultExecutionTape
        assertEquals(true, tape.traces.single { it.opType == "gelu" }.attributes["approximate"])
    }
}
