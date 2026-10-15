package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.hlo.validation.MlirParser
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

/**
 * #1343: the SDPA mask rules (`causal`, `window`/`rightContext`, `keyLengths`) lower to an
 * `iota`/`compare`/`select` keep mask built from the node metadata — no `[b, 1, Sq, Sk]`
 * constant or graph input — and the causal diagonal carries the `Sk − Sq` offset the CPU kernel uses.
 */
class SdpaMaskRulesHloTest {

    private fun graph(
        q: List<Int>, kv: List<Int>, params: Map<String, Any>, mask: List<Int>? = null, lengths: Int? = null,
    ): DefaultComputeGraph {
        val g = DefaultComputeGraph()
        fun input(id: String, shape: List<Int>, dtype: String = "FP32") =
            GraphNode(id, InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec(id, shape, dtype))).also { g.addNode(it) }
        val qs = input("q", q); val ks = input("k", kv); val vs = input("v", kv)
        val ms = mask?.let { input("m", it) }
        val ls = lengths?.let { input("len", listOf(it), "Int32") }
        val operands = listOfNotNull(qs, ks, vs, ms, ls)
        val out = TensorSpec("o", q, "FP32")
        val sdpa = GraphNode("sdpa", ScaledDotProductAttentionOperation(params), operands.map { it.outputs[0] }, listOf(out))
        g.addNode(sdpa)
        operands.forEachIndexed { i, n -> g.addEdge(GraphEdge("e$i", n, sdpa, 0, i, n.outputs[0])) }
        return g
    }

    private fun convert(g: DefaultComputeGraph, name: String = "t") = StableHloConverterFactory.createBasic().convert(g, name).content

    private fun params(causal: Boolean = false, window: Int = 0, rightContext: Int = 0, lengths: Boolean = false) =
        ScaledDotProductAttentionOperation.parameters(0f, causal, window, rightContext, lengths)

    @Test
    fun key_lengths_lower_to_iota_compare_select_from_the_lengths_operand() {
        val mlir = convert(graph(listOf(3, 4, 8, 16), listOf(3, 4, 8, 16), params(lengths = true), lengths = 3))
        assertTrue(mlir.contains("%arg3: tensor<3xi32>"), "lengths are a plain [batch] i32 input:\n$mlir")
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<3xi32>) -> tensor<3x4x8x8xi32>"), "lengths broadcast over the scores:\n$mlir")
        assertTrue(mlir.contains("stablehlo.iota dim = 3 : tensor<3x4x8x8xi32>"), "key iota:\n$mlir")
        assertTrue(mlir.contains("stablehlo.compare LT, %"), "ki < length:\n$mlir")
        assertTrue(mlir.contains("stablehlo.select %") && mlir.contains("dense<-1.000000e+30> : tensor<3x4x8x8xf32>"), "masked fill:\n$mlir")
        assertFalse(mlir.contains("%arg3: tensor<3x1x8x8xf32>"), "no dense mask input for padding:\n$mlir")
        assertTrue(MlirValidator().validate(mlir).isEmpty(), MlirValidator().validate(mlir).toString())
        assertEquals("t", MlirParser().parse(mlir).getOrThrow().functionName)
    }

    @Test
    fun window_band_and_causal_offset_follow_the_shared_rules() {
        // Chunked prefill: Sq = 2 over Sk = 6 → absQ = 4 + qi; window 3 → lower = absQ - 2.
        val mlir = convert(graph(listOf(1, 2, 2, 8), listOf(1, 2, 6, 8), params(causal = true, window = 3)))
        assertTrue(mlir.contains("stablehlo.constant dense<4> : tensor<1x2x2x6xi32>"), "causal offset Sk - Sq = 4:\n$mlir")
        assertTrue(mlir.contains("stablehlo.compare GE, %"), "causal diagonal:\n$mlir")
        assertTrue(mlir.contains("stablehlo.constant dense<-2> : tensor<1x2x2x6xi32>"), "window lower edge 1 - window:\n$mlir")
        assertTrue(mlir.contains("stablehlo.and %"), "rules intersect:\n$mlir")
        assertFalse(mlir.contains("compare LE"), "under causal the band has no separate right edge:\n$mlir")

        // Bidirectional band: window 4, rightContext 2, no causal → both edges, no diagonal.
        val band = convert(graph(listOf(1, 2, 6, 8), listOf(1, 2, 6, 8), params(window = 4, rightContext = 2)))
        assertTrue(band.contains("dense<-3> : tensor<1x2x6x6xi32>") && band.contains("dense<2> : tensor<1x2x6x6xi32>"), "band edges:\n$band")
        assertTrue(band.contains("compare GE") && band.contains("compare LE"), "both edges:\n$band")
        assertFalse(band.contains("dense<0> : tensor<1x2x6x6xi32>"), "no zero offset constant when Sq == Sk:\n$band")
    }

    @Test
    fun causal_prefill_keeps_the_previous_emission() {
        // Sq == Sk: offset 0 is skipped, so the lines are what the converter emitted before #1343.
        val mlir = convert(graph(listOf(1, 1, 2, 4), listOf(1, 1, 2, 4), mapOf("scale" to 0.0f, "causal" to true)))
        assertTrue(mlir.contains("stablehlo.iota dim = 2 : tensor<1x1x2x2xi32>"))
        assertTrue(mlir.contains("stablehlo.iota dim = 3 : tensor<1x1x2x2xi32>"))
        assertTrue(Regex("stablehlo\\.compare GE, %v\\d+, %v\\d+ : \\(tensor<1x1x2x2xi32>, tensor<1x1x2x2xi32>\\) -> tensor<1x1x2x2xi1>").containsMatchIn(mlir), mlir)
        assertFalse(mlir.contains("stablehlo.and"), "a single rule needs no and:\n$mlir")
        assertFalse(mlir.contains("stablehlo.constant dense<0> : tensor<1x1x2x2xi32>"), "no offset add:\n$mlir")
    }

    @Test
    fun dense_mask_and_rules_combine_like_the_cpu_kernel() {
        val mlir = convert(graph(listOf(2, 2, 4, 8), listOf(2, 2, 4, 8), params(causal = true, lengths = true), mask = listOf(2, 1, 4, 4), lengths = 2))
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2, 3] : (tensor<2x1x4x4xf32>) -> tensor<2x2x4x4xf32>"), "dense mask added:\n$mlir")
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<2xi32>) -> tensor<2x2x4x4xi32>"), "lengths after the mask:\n$mlir")
        assertTrue(mlir.contains("compare GE") && mlir.contains("compare LT"), "causal and padding both apply:\n$mlir")
    }

    @Test
    fun rules_on_a_dynamic_key_length_are_refused_but_a_dense_mask_is_not() {
        val dyn = listOf(1, 2, TypeMapper.DYNAMIC_DIM, 8)
        val refused = runCatching { convert(graph(listOf(1, 2, 1, 8), dyn, params(lengths = true), lengths = 1)) }
        assertTrue(refused.isFailure || !refused.getOrThrow().contains("stablehlo.iota"), "structured rules need static scores")
        val dense = convert(graph(listOf(1, 2, 1, 8), dyn, params(), mask = listOf(1, 1, 1, TypeMapper.DYNAMIC_DIM)))
        assertTrue(dense.contains("stablehlo.dynamic_broadcast_in_dim"), "dense mask on the dynamic path:\n$dense")
    }
}
