package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.hlo.validation.MlirParser
import sk.ainet.compile.target.FusedOpAllowList
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.ErfOperation
import sk.ainet.lang.tensor.ops.GeluOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

/**
 * #1342 part 2: `stablehlo.composite` emission behind [sk.ainet.compile.target.OpGranularityPolicy]
 * — the first consumer of the policy. A target that keeps `gelu` fused gets the composite plus a
 * private decomposition function; every other target gets the decomposition inline, and both
 * come from the same builder.
 */
class CompositeEmissionTest {

    private val shape = listOf(2, 3)
    private val type = "tensor<2x3xf32>"

    private fun chain(vararg ops: Operation): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        var prev = GraphNode("input", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("input", shape, "FP32")))
        graph.addNode(prev)
        ops.forEachIndexed { i, op ->
            val node = GraphNode("n$i", op, listOf(TensorSpec("in$i", shape, "FP32")), listOf(TensorSpec("out$i", shape, "FP32")))
            graph.addNode(node)
            graph.addEdge(GraphEdge("e$i", prev, node, 0, 0, prev.outputs[0]))
            prev = node
        }
        return graph
    }

    private fun exactGelu() = GeluOperation<DType, Any>(mapOf("approximate" to false))
    private fun fused(vararg ops: String) = FusedOpAllowList("test-npu", ops.toSet())

    @Test
    fun fused_target_gets_composite_and_private_decomposition_after_main() {
        val module = StableHloConverterFactory.createBasic(granularity = fused("gelu"))
            .convert(chain(exactGelu()), "test_fused")
        val c = module.content

        assertTrue(c.contains("stablehlo.composite \"skainet.gelu\" %arg0 {composite_attributes = {approximate = false}, decomposition = @skainet_gelu_approximate_false_2x3xf32, version = 1 : i32} : (tensor<2x3xf32>) -> tensor<2x3xf32>"), c)
        assertTrue(c.contains("func.func private @skainet_gelu_approximate_false_2x3xf32(%arg0: $type) -> $type {"), c)
        assertTrue(c.contains("chlo.erf"), "the private body must hold the exact decomposition")
        assertTrue(c.indexOf("func.func @test_fused") < c.indexOf("func.func private"), "entry function must stay first")
        assertTrue(c.trimEnd().endsWith("}"), c)

        // The text tooling still treats @test_fused as the entry function and the module validates.
        assertEquals("test_fused", MlirParser().parse(c).getOrThrow().functionName)
        assertTrue(MlirValidator().validate(c).isEmpty(), MlirValidator().validate(c).toString())
    }

    @Test
    fun portable_target_gets_the_decomposition_inline() {
        val module = StableHloConverterFactory.createBasic().convert(chain(exactGelu()), "test_inline")
        assertFalse(module.content.contains("stablehlo.composite"))
        assertFalse(module.content.contains("func.func private"))
        assertTrue(module.content.contains("chlo.erf"))
    }

    @Test
    fun policy_that_decomposes_gelu_emits_inline_even_when_set() {
        val module = StableHloConverterFactory.createBasic(granularity = fused("layernorm"))
            .convert(chain(exactGelu()), "test_other")
        assertFalse(module.content.contains("stablehlo.composite"))
    }

    @Test
    fun decomposition_function_is_emitted_once_per_shape_and_attributes() {
        val module = StableHloConverterFactory.createBasic(granularity = fused("gelu"))
            .convert(chain(exactGelu(), exactGelu(), GeluOperation<DType, Any>(mapOf("approximate" to true))), "test_dedup")
        val c = module.content
        assertEquals(3, Regex("stablehlo\\.composite \"skainet\\.gelu\"").findAll(c).count())
        assertEquals(1, Regex("func\\.func private @skainet_gelu_approximate_false_2x3xf32\\(").findAll(c).count())
        assertEquals(1, Regex("func\\.func private @skainet_gelu_approximate_true_2x3xf32\\(").findAll(c).count())
        assertTrue(c.contains("composite_attributes = {approximate = true}"))
        assertTrue(MlirValidator().validate(c).isEmpty(), "SSA names must stay unique across main and helpers")
    }

    @Test
    fun erf_has_its_own_composite() {
        val module = StableHloConverterFactory.createBasic(granularity = fused("erf"))
            .convert(chain(ErfOperation<DType, Any>()), "test_erf")
        val c = module.content
        assertTrue(c.contains("stablehlo.composite \"skainet.erf\" %arg0 {decomposition = @skainet_erf_2x3xf32, version = 1 : i32} : ($type) -> $type"), c)
        assertTrue(c.contains("func.func private @skainet_erf_2x3xf32(%arg0: $type) -> $type {"), c)
        assertTrue(c.contains("chlo.erf %arg0 : $type -> $type"), c)
        assertTrue(MlirValidator().validate(c).isEmpty())
    }

    @Test
    fun composite_body_equals_the_inline_decomposition_modulo_ssa_names() {
        fun normalize(lines: List<String>) = lines.map { it.trim().replace(Regex("%v\\d+"), "%v").replace("%arg0", "%x") }
        val inline = StableHloConverterFactory.createBasic().convert(chain(exactGelu()), "a").content
        val fusedModule = StableHloConverterFactory.createBasic(granularity = fused("gelu")).convert(chain(exactGelu()), "b").content

        val inlineOps = inline.lines().filter { it.trim().startsWith("%v") }
        val bodyStart = fusedModule.indexOf("func.func private")
        val body = fusedModule.substring(bodyStart).lines().filter { it.trim().startsWith("%v") }
        assertEquals(normalize(inlineOps), normalize(body), "one builder must produce both forms")
    }

    // ---- #1343 part 2: the n-ary seam, `skainet.sdpa` ----

    private fun sdpaGraph(lengths: Boolean, mask: Boolean): DefaultComputeGraph {
        val g = DefaultComputeGraph()
        fun input(id: String, shape: List<Int>, dtype: String = "FP32") =
            GraphNode(id, InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec(id, shape, dtype))).also { g.addNode(it) }
        val q = input("q", listOf(2, 4, 8, 16)); val k = input("k", listOf(2, 2, 8, 16)); val v = input("v", listOf(2, 2, 8, 16))
        val m = if (mask) input("m", listOf(2, 1, 8, 8)) else null
        val l = if (lengths) input("len", listOf(2), "Int32") else null
        val operands = listOfNotNull(q, k, v, m, l)
        val params = sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation.parameters(0f, causal = true, window = 4, rightContext = 0, hasLengths = lengths)
        val sdpa = GraphNode("sdpa", sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation(params), operands.map { it.outputs[0] }, listOf(TensorSpec("o", listOf(2, 4, 8, 16), "FP32")))
        g.addNode(sdpa)
        operands.forEachIndexed { i, n -> g.addEdge(GraphEdge("e$i", n, sdpa, 0, i, n.outputs[0])) }
        return g
    }

    @Test
    fun sdpa_composite_carries_every_operand_and_the_mask_rules_as_attributes() {
        val module = StableHloConverterFactory.createBasic(granularity = fused("scaleddotproductattention")).convert(sdpaGraph(lengths = true, mask = false), "sdpa_fused")
        val c = module.content
        val line = c.lines().single { "stablehlo.composite \"skainet.sdpa\"" in it }.trim()
        assertTrue(line.contains("stablehlo.composite \"skainet.sdpa\" %arg0, %arg1, %arg2, %arg3 {"), line)
        assertTrue(line.contains("composite_attributes = {scale = 0.25 : f32, causal = true, window = 4 : i32, rightContext = 0 : i32, maskKind = \"combined\"}"), line)
        assertTrue(line.endsWith(": (tensor<2x4x8x16xf32>, tensor<2x2x8x16xf32>, tensor<2x2x8x16xf32>, tensor<2xi32>) -> tensor<2x4x8x16xf32>"), line)
        val decl = Regex("decomposition = @(\\S+), version = 1 : i32").find(line)!!.groupValues[1]
        assertTrue(c.contains("func.func private @$decl(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x2x8x16xf32>, %arg2: tensor<2x2x8x16xf32>, %arg3: tensor<2xi32>) -> tensor<2x4x8x16xf32> {"), c)
        assertTrue(c.indexOf("func.func @sdpa_fused") < c.indexOf("func.func private"), "entry function must stay first")
        assertTrue(c.substring(c.indexOf("func.func private")).contains("stablehlo.broadcast_in_dim %arg3, dims = [0]"), "the body lowers the lengths operand")
        assertEquals("sdpa_fused", MlirParser().parse(c).getOrThrow().functionName)
        assertTrue(MlirValidator().validate(c).isEmpty(), MlirValidator().validate(c).toString())

        // Five operands when a dense mask is present as well.
        val withMask = StableHloConverterFactory.createBasic(granularity = fused("scaleddotproductattention")).convert(sdpaGraph(lengths = true, mask = true), "sdpa_mask").content
        assertTrue(withMask.contains("stablehlo.composite \"skainet.sdpa\" %arg0, %arg1, %arg2, %arg3, %arg4 {"), withMask)
        assertTrue(withMask.contains("%arg3: tensor<2x1x8x8xf32>, %arg4: tensor<2xi32>)"), withMask)
    }

    @Test
    fun sdpa_composite_body_equals_the_inline_decomposition_modulo_ssa_names() {
        fun normalize(lines: List<String>) = lines.map { it.trim().replace(Regex("%v\\d+"), "%v") }
        val inline = StableHloConverterFactory.createBasic().convert(sdpaGraph(lengths = true, mask = true), "a").content
        val fusedModule = StableHloConverterFactory.createBasic(granularity = fused("scaleddotproductattention")).convert(sdpaGraph(lengths = true, mask = true), "b").content
        assertFalse(inline.contains("stablehlo.composite"))
        val inlineOps = inline.lines().filter { it.trim().startsWith("%v") }
        val body = fusedModule.substring(fusedModule.indexOf("func.func private")).lines().filter { it.trim().startsWith("%v") }
        assertEquals(normalize(inlineOps), normalize(body), "one builder must produce both forms")
        // A policy that fuses only gelu leaves SDPA inline.
        assertFalse(StableHloConverterFactory.createBasic(granularity = fused("gelu")).convert(sdpaGraph(lengths = false, mask = false), "c").content.contains("stablehlo.composite"))
    }

    @Test
    fun text_optimizer_passes_tolerate_a_module_with_a_helper_function() {
        val converter = StableHloConverterFactory.createBasic(granularity = fused("gelu"))
        val optimized = converter.convertWithOptimization(chain(exactGelu()), StableHloOptimizer.createDefault())
        assertTrue(optimized.content.contains("stablehlo.composite \"skainet.gelu\""))
        assertTrue(optimized.content.contains("func.func private @skainet_gelu_approximate_false_2x3xf32"))
        assertTrue(optimized.content.contains("chlo.erf"), "DCE must not drop the helper body")
    }
}
