package sk.ainet.exec.autograd

import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.tensor.ops.DifferentiableTensorOpsRules
import java.io.File
import kotlin.test.Test

/**
 * Dumps the autodiff backward coverage to `build/generated/coverage/backward-coverage.json`
 * for the operator coverage matrix's Backward column (`generateDocs`).
 *
 * `implemented` = dispatched backward rule; `stubbed` = dispatched but listed in
 * `stubbedBackwardOpNames` (returns null gradients); `missing` = `@Diff` op with no dispatch arm.
 */
class BackwardCoverageDumpTest {

    @Test
    fun dump_backward_coverage() {
        val tape = DefaultGradientTape()
        val dispatched = tape.dispatchedOpNames
        val stubbed = tape.stubbedBackwardOpNames
        val ops = (DifferentiableTensorOpsRules.ruleNames + dispatched).toSortedSet()
        val json = ops.joinToString(",\n", prefix = "{\n", postfix = "\n}\n") { op ->
            val status = when {
                op in stubbed -> "stubbed"
                op in dispatched -> "implemented"
                else -> "missing"
            }
            "  \"$op\": \"$status\""
        }
        val outDir = File("build/generated/coverage").apply { mkdirs() }
        File(outDir, "backward-coverage.json").writeText(json)
    }
}
