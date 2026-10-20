package sk.ainet.compile.hlo

import java.io.File
import kotlin.test.Test

/**
 * Dumps the op names the extended StableHLO converter can lower to
 * `build/generated/coverage/hlo-coverage.json`, for the operator coverage matrix's
 * StableHLO column (`generateDocs`). An op missing from this list throws
 * "No converter found" at export time.
 */
class HloCoverageDumpTest {

    @Test
    fun dump_stablehlo_coverage() {
        val ops = StableHloConverterFactory.createExtended().supportedOperationNames.toSortedSet()
        val json = ops.joinToString(",\n", prefix = "[\n", postfix = "\n]\n") { "  \"$it\"" }
        val outDir = File("build/generated/coverage").apply { mkdirs() }
        File(outDir, "hlo-coverage.json").writeText(json)
    }
}
