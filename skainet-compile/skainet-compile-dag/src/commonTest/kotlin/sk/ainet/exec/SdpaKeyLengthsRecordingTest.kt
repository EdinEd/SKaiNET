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
import sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * #1343 acceptance: the mask rules of an eager (DIRECT) SDPA call are recorded as node metadata
 * (`maskKind`, `window`, `rightContext`) with `keyLengths` as the last operand, and replaying the
 * graph (OPTIMIZED) reproduces the eager output.
 */
class SdpaKeyLengthsRecordingTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun tensor(shape: Shape, seed: Int): VoidOpsTensor<FP32, Float> {
        val values = FloatArray(shape.volume) { i -> (((i * 37 + seed * 11) % 29) - 14) / 7f }
        return VoidOpsTensor(dataFactory.fromFloatArray(shape, FP32::class, values), FP32::class)
    }

    @Suppress("UNCHECKED_CAST")
    private fun lengths(vararg values: Int): Tensor<DType, *> =
        VoidOpsTensor(dataFactory.fromIntArray<Int32, Int>(Shape(values.size), Int32::class, values), Int32::class) as Tensor<DType, *>

    private fun floats(t: Tensor<*, *>): FloatArray = t.data.copyToFloatArray()

    @Test
    fun key_lengths_are_recorded_as_metadata_and_replay_bit_identical() {
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        val q = tensor(Shape(2, 4, 6, 8), 1)
        val k = tensor(Shape(2, 2, 6, 8), 2)
        val v = tensor(Shape(2, 2, 6, 8), 3)
        val w = tensor(Shape(8, 5), 4)
        val len = lengths(6, 3)

        ctx.startRecording()
        val ops = ctx.ops
        val attn = ops.scaledDotProductAttention(q, k, v, keyLengths = len, window = 4, rightContext = 1)
        val eager = ops.matmul(attn, w)
        val tape = ctx.stopRecording() as DefaultExecutionTape

        // The trace carries the rules …
        val trace = tape.traces.single { it.opType == "scaledDotProductAttention" }
        assertEquals("combined", trace.attributes["maskKind"])
        assertEquals(4, trace.attributes["window"])
        assertEquals(1, trace.attributes["rightContext"])
        assertEquals(false, trace.attributes["causal"])
        assertEquals(4, trace.inputs.size, "q, k, v, keyLengths")

        // … and so does the graph node, with keyLengths as its last operand.
        val graph = tape.toComputeGraph(synthesizeExternalInputs = true, embedConstants = false)
        val node = graph.nodes.single { it.operation.name == "scaledDotProductAttention" }
        assertEquals("combined", node.operation.parameters["maskKind"])
        val layout = ScaledDotProductAttentionOperation.operandLayout(node.inputs.size, node.operation.parameters)
        assertEquals(ScaledDotProductAttentionOperation.OperandLayout(maskIndex = null, lengthsIndex = 3), layout)

        // Replay through the graph executor on the same kernels: external inputs keyed by trace id.
        val session = ctx.session
        @Suppress("UNCHECKED_CAST")
        val inputs: Map<String, Tensor<DType, Any>> = mapOf(
            session.refOf(q).id to q, session.refOf(k).id to k, session.refOf(v).id to v,
            session.refOf(w).id to w, session.refOf(len).id to len,
        ) as Map<String, Tensor<DType, Any>>
        val outputs = ComputeGraphExecutor(graph, DefaultCpuOps(dataFactory)).execute<DType, Any>(inputs)
        val matmul = graph.nodes.single { it.operation.name == "matmul" }
        val replayed = floats(outputs.getValue(matmul.id))
        val expected = floats(eager)
        assertEquals(expected.size, replayed.size)
        var maxDiff = 0f
        for (i in expected.indices) maxDiff = maxOf(maxDiff, kotlin.math.abs(expected[i] - replayed[i]))
        assertTrue(maxDiff <= 1e-6f, "DIRECT vs OPTIMIZED max |Δ| = $maxDiff")

        // Sanity: the rules really applied — the unmasked call differs on this block.
        val plain = floats(DefaultCpuOps(dataFactory).scaledDotProductAttention(q, k, v))
        var ruleDiff = 0f
        for (i in plain.indices) ruleDiff = maxOf(ruleDiff, kotlin.math.abs(plain[i] - floats(attn)[i]))
        assertTrue(ruleDiff > 1e-5f, "masked and unmasked attention should differ, max |Δ| = $ruleDiff")
    }

    @Test
    fun dense_mask_and_lengths_keep_their_operand_slots() {
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        val q = tensor(Shape(1, 2, 3, 4), 1); val k = tensor(Shape(1, 2, 3, 4), 2); val v = tensor(Shape(1, 2, 3, 4), 3)
        val mask = tensor(Shape(1, 1, 3, 3), 9)
        ctx.startRecording()
        ctx.ops.scaledDotProductAttention(q, k, v, mask = mask, causal = true, keyLengths = lengths(2))
        ctx.ops.scaledDotProductAttention(q, k, v, mask = mask, causal = true)
        ctx.ops.scaledDotProductAttention(q, k, v, causal = true)
        val tape = ctx.stopRecording() as DefaultExecutionTape
        val traces = tape.traces.filter { it.opType == "scaledDotProductAttention" }
        assertEquals(listOf("combined", "causal", "causal"), traces.map { it.attributes["maskKind"] })
        assertEquals(listOf(5, 4, 3), traces.map { it.inputs.size })
        val layouts = traces.map { ScaledDotProductAttentionOperation.operandLayout(it.inputs.size, it.attributes) }
        assertEquals(ScaledDotProductAttentionOperation.OperandLayout(3, 4), layouts[0])
        assertEquals(ScaledDotProductAttentionOperation.OperandLayout(3, null), layouts[1])
        assertEquals(ScaledDotProductAttentionOperation.OperandLayout(null, null), layouts[2])
    }
}
