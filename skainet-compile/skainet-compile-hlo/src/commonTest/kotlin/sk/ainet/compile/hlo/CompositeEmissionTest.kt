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

    @Test
    fun text_optimizer_passes_tolerate_a_module_with_a_helper_function() {
        val converter = StableHloConverterFactory.createBasic(granularity = fused("gelu"))
        val optimized = converter.convertWithOptimization(chain(exactGelu()), StableHloOptimizer.createDefault())
        assertTrue(optimized.content.contains("stablehlo.composite \"skainet.gelu\""))
        assertTrue(optimized.content.contains("func.func private @skainet_gelu_approximate_false_2x3xf32"))
        assertTrue(optimized.content.contains("chlo.erf"), "DCE must not drop the helper body")
    }
}
