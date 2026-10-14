package sk.ainet.compile.opt

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.opt.passes.GELU_TANH_MAX_ABS_ERROR
import sk.ainet.compile.opt.passes.GeluApproximationLegalizationPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GeluOperation
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

class GeluApproximationLegalizationPassTest {

    @AfterTest
    fun reset() = TargetOptimizers.clear()

    private fun spec(name: String = "t") = TensorSpec(name = name, shape = listOf(4), dtype = "Float32")

    /** input -> gelu(approximate) -> consumer */
    private fun graphWith(approximate: Boolean): DefaultComputeGraph {
        val g = DefaultComputeGraph()
        val input = g.addNode(GraphNode("x", GenericOperation("input"), emptyList(), listOf(spec("x"))))
        val gelu = g.addNode(
            GraphNode("g", GeluOperation<DType, Any>(mapOf("approximate" to approximate)), listOf(spec("x")), listOf(spec("g")))
        )
        val relu = g.addNode(GraphNode("r", GenericOperation("relu"), listOf(spec("g")), listOf(spec("r"))))
        g.addEdge(GraphEdge("e0", input, gelu, tensorSpec = spec()))
        g.addEdge(GraphEdge("e1", gelu, relu, tensorSpec = spec()))
        return g
    }

    @Test
    fun rewrites_exact_gelu_to_tanh_and_records_the_substitution() {
        val result = GeluApproximationLegalizationPass("no erf kernel").apply(graphWith(approximate = false))
        assertTrue(result.changed)
        val node = result.graph.nodes.first { it.id == "g" }
        assertTrue(node.operation is GeluOperation<*, *>)
        assertEquals(true, node.operation.parameters["approximate"])
        assertEquals("gelu(approximate=false)", node.operation.parameters[GeluApproximationLegalizationPass.LEGALIZED_FROM])
        assertEquals(GELU_TANH_MAX_ABS_ERROR, node.operation.parameters[GeluApproximationLegalizationPass.MAX_ABS_ERROR])
        assertEquals(1, result.diagnostics.size)
        assertTrue(result.diagnostics[0].startsWith("GELU_APPROX_LEGALIZED node=g:"), result.diagnostics[0])
        assertTrue(result.diagnostics[0].contains("[no erf kernel]"))

        // Edges survive the node swap.
        assertEquals(listOf("x"), result.graph.edges.filter { it.destination.id == "g" }.map { it.source.id })
        assertEquals(listOf("r"), result.graph.edges.filter { it.source.id == "g" }.map { it.destination.id })
    }

    @Test
    fun leaves_tanh_gelu_untouched_and_is_idempotent() {
        val untouched = GeluApproximationLegalizationPass().apply(graphWith(approximate = true))
        assertFalse(untouched.changed)
        assertTrue(untouched.diagnostics.isEmpty())

        val first = GeluApproximationLegalizationPass().apply(graphWith(approximate = false))
        val second = GeluApproximationLegalizationPass().apply(first.graph)
        assertFalse(second.changed, "a legalized node must not be rewritten again")
    }

    @Test
    fun a_target_opts_in_through_the_registry() {
        TargetOptimizers.registerDagPasses("test-no-erf") { listOf(GeluApproximationLegalizationPass("test-no-erf has no erf")) }

        val result = dagPipelineFor("test-no-erf").optimize(graphWith(approximate = false))
        val node = result.graph.nodes.first { it.id == "g" }
        assertEquals(true, node.operation.parameters["approximate"])
        assertTrue(result.passResults.flatMap { it.diagnostics }.any { it.startsWith("GELU_APPROX_LEGALIZED") })

        // A target that did not register keeps the exact form.
        val portable = dagPipelineFor("llvm-cpu").optimize(graphWith(approximate = false))
        assertEquals(false, portable.graph.nodes.first { it.id == "g" }.operation.parameters["approximate"])
    }
}
