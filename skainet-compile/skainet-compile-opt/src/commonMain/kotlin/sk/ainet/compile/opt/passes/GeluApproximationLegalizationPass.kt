package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode

/**
 * Maximum absolute difference between the exact and the tanh-approximated GELU, measured on
 * the CPU kernel over `[-6, 6]` (4.73e-4 at |x| ≈ 2.70), rounded up.
 */
public const val GELU_TANH_MAX_ABS_ERROR: Double = 4.8e-4

/**
 * Target legalization for targets without any `erf` lowering (#1342): rewrites
 * `gelu(approximate = false)` nodes to `gelu(approximate = true)` and **records** the
 * substitution on the node (`legalizedFrom`, `maxAbsError`) and in the pass diagnostics,
 * so a numerics change is a visible decision instead of a silent default.
 *
 * `erf` nodes are left alone on purpose: there is no cheap substitute, and a loud failure in
 * the emitter beats a quiet approximation.
 *
 * A target opts in by registering the pass:
 * ```kotlin
 * TargetOptimizers.registerDagPasses("my-npu") { listOf(GeluApproximationLegalizationPass("my-npu has no erf kernel")) }
 * ```
 */
public class GeluApproximationLegalizationPass(
    private val reason: String = "target has no erf lowering",
) : GraphOptimizationPass {

    override val name: String = "gelu-approximation-legalization"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        var changed = false

        val candidates = graph.nodes.filter { node ->
            node.operation.name.equals("gelu", ignoreCase = true) &&
                node.operation.parameters[APPROXIMATE] == false
        }

        for (node in candidates) {
            val legalized = GraphNode(
                id = node.id,
                operation = node.operation.clone(
                    node.operation.parameters + mapOf(
                        APPROXIMATE to true,
                        LEGALIZED_FROM to "gelu(approximate=false)",
                        MAX_ABS_ERROR to GELU_TANH_MAX_ABS_ERROR,
                    )
                ),
                inputs = node.inputs,
                outputs = node.outputs,
                metadata = node.metadata,
            )

            val incoming = graph.edges.filter { it.destination.id == node.id }
            val outgoing = graph.edges.filter { it.source.id == node.id }
            graph.removeNode(node)
            graph.addNode(legalized)
            for (edge in incoming) {
                graph.addEdge(GraphEdge(edge.id, edge.source, legalized, edge.sourceOutputIndex, edge.destinationInputIndex, edge.tensorSpec))
            }
            for (edge in outgoing) {
                graph.addEdge(GraphEdge(edge.id, legalized, edge.destination, edge.sourceOutputIndex, edge.destinationInputIndex, edge.tensorSpec))
            }

            diagnostics += "GELU_APPROX_LEGALIZED node=${node.id}: gelu(approximate=false) -> gelu(approximate=true) [$reason] maxAbsError=$GELU_TANH_MAX_ABS_ERROR"
            changed = true
        }

        return GraphOptimizationResult(graph, changed = changed, diagnostics = diagnostics)
    }

    public companion object {
        public const val APPROXIMATE: String = "approximate"
        /** Node parameter naming the original op the node was legalized from. */
        public const val LEGALIZED_FROM: String = "legalizedFrom"
        /** Node parameter carrying the worst-case absolute error introduced by the rewrite. */
        public const val MAX_ABS_ERROR: String = "maxAbsError"
    }
}
