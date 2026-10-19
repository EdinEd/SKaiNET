import kotlinx.serialization.json.Json
import models.KernelSupportModule
import org.gradle.api.DefaultTask
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction

/**
 * Renders `kernel-support.json` (registry-introspected by `KernelSupportMatrixTest`) into an
 * Antora AsciiDoc matrix — the kernel-side counterpart of [GenerateDocumentationTask]'s
 * `ops-status-matrix.adoc`. Rows are weight formats; columns are KMP platforms; each cell is
 * the best provider that serves `FP32 × format` on that platform.
 *
 * Deliberately omits a generation timestamp so the committed `.adoc` only changes when the
 * actual provider coverage changes (keeps the docs-CI staleness diff meaningful).
 */
@CacheableTask
abstract class GenerateKernelMatrixTask : DefaultTask() {

    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val inputFile: RegularFileProperty

    @get:OutputFile
    abstract val outputFile: RegularFileProperty

    @TaskAction
    fun generate() {
        val json = Json { ignoreUnknownKeys = true }
        val module = json.decodeFromString<KernelSupportModule>(inputFile.get().asFile.readText())
        val out = outputFile.get().asFile
        out.parentFile?.mkdirs()

        out.writeText(buildString {
            appendLine("= Kernel × platform support matrix")
            appendLine(":description: Which compute-kernel provider serves each weight format on each KMP target.")
            appendLine("")
            appendLine(
                "Generated from `kernel-support.json` (version `${module.version}`) by " +
                    "`KernelSupportMatrixTest` — registry introspection of the registered " +
                    "`KernelProvider` implementations. Do not edit by hand; run " +
                    "`./gradlew generateKernelMatrix` to refresh.",
            )
            appendLine("")
            appendLine(
                "Each cell is the best (highest-priority) provider that serves " +
                    "`${module.inputDtype} × format` `${module.op}` on that platform: " +
                    "*native-ffm* / *native-jni* / *native-cinterop* (100) → *panama-vector* / " +
                    "*wasm-simd128* (50) → *scalar* (0). An empty cell " +
                    "(`—`) means no provider carries a kernel there (the format is dequant-to-FP32 only). " +
                    "The JS/WASM column assumes `WasmSimd.install()` has run; without the sidecar it is the scalar floor.",
            )
            appendLine("")
            if (module.formats.isEmpty() || module.platforms.isEmpty()) {
                appendLine("NOTE: No kernel-support data found in the source JSON.")
                return@buildString
            }

            val colSpec = (listOf("1") + List(module.platforms.size) { "1" }).joinToString(",")
            appendLine("[cols=\"$colSpec\", options=\"header\"]")
            appendLine("|===")
            append("| Weight format ")
            module.platforms.forEach { append("| $it ") }
            appendLine("")
            appendLine("")
            module.formats.forEach { fmt ->
                append("| `${fmt.name}` ")
                module.platforms.forEach { p -> append("| ${fmt.byPlatform[p] ?: "—"} ") }
                appendLine("")
            }
            appendLine("|===")
            appendLine("")
            if (module.mapped.isNotEmpty()) {
                appendLine("== Mapped serving (row-major, off-heap)")
                appendLine("")
                appendLine(
                    "Kernels that read the weight in canonical row-major GGUF file order straight " +
                        "from off-heap bytes (mmap'd pages or a direct buffer) — no heap staging, no " +
                        "prepack copy (#1189). This is what lets a model larger than the managed-heap " +
                        "cap decode on Android. Dense `Float32` tensors are also served from the " +
                        "mapping (as element views, not a matmul kernel) — see " +
                        "`StorageCapabilities.mappedServableEncodings` for the authoritative set the " +
                        "memory plan budgets against the page cache. An empty cell means the format " +
                        "heap-stages under `WeightResidency.MAPPED` on that platform.",
                )
                appendLine("")
                appendLine("[cols=\"$colSpec\", options=\"header\"]")
                appendLine("|===")
                append("| Weight format ")
                module.platforms.forEach { append("| $it ") }
                appendLine("")
                appendLine("")
                module.mapped.forEach { fmt ->
                    append("| `${fmt.name}` ")
                    module.platforms.forEach { p -> append("| ${fmt.byPlatform[p] ?: "—"} ") }
                    appendLine("")
                }
                appendLine("|===")
                appendLine("")
            }
            if (module.resident.isNotEmpty()) {
                appendLine("== Resident serving (sidecar memory)")
                appendLine("")
                appendLine(
                    "The browser has no files to map, so the wasm tier keeps weights *resident* in the " +
                        "SIMD128 sidecar's linear memory instead: each weight is uploaded once on its first " +
                        "matmul and every later call reads it in place (`WasmKernels.weightHandle`). This is " +
                        "the JS/WASM counterpart of mapped serving above — no per-call copy of the weight, " +
                        "no prepack — bounded by the wasm32 4 GB linear memory rather than the page cache. " +
                        "The other platforms serve from the heap, off-heap or the mapping and have no cell here.",
                )
                appendLine("")
                appendLine("[cols=\"$colSpec\", options=\"header\"]")
                appendLine("|===")
                append("| Weight format ")
                module.platforms.forEach { append("| $it ") }
                appendLine("")
                appendLine("")
                module.resident.forEach { fmt ->
                    append("| `${fmt.name}` ")
                    module.platforms.forEach { p -> append("| ${fmt.byPlatform[p] ?: "—"} ") }
                    appendLine("")
                }
                appendLine("|===")
                appendLine("")
            }
            if (module.packs.isNotEmpty()) {
                appendLine("== Kernel packs (view dispatch)")
                appendLine("")
                appendLine(
                    "Kernels that are not reached through the `matmul` accessors above: the ternary formats " +
                        "are served by `ViewKernelPack`s keyed in `KernelDispatch` (`BitNet-b1.58` with exact " +
                        "FP32 activations, the fused multi-plane `BitNet-planes` lm_head, `TQ2_0` with int8 " +
                        "activations), and the dense `x · Wᵀ` of a `Linear` layer by the `matmulFp32Wt` " +
                        "accessor, which multiplies a row-major `[out, in]` weight in place instead of " +
                        "transposing and caching a copy. The scalar reference kernels serve every cell that " +
                        "is empty here (`—`), at reference speed; the common row-dot serves `x · Wᵀ` there.",
                )
                appendLine("")
                appendLine("[cols=\"$colSpec\", options=\"header\"]")
                appendLine("|===")
                append("| Kernel ")
                module.platforms.forEach { append("| $it ") }
                appendLine("")
                appendLine("")
                module.packs.forEach { fmt ->
                    append("| `${fmt.name}` ")
                    module.platforms.forEach { p -> append("| ${fmt.byPlatform[p] ?: "—"} ") }
                    appendLine("")
                }
                appendLine("|===")
                appendLine("")
            }
            appendLine(
                "See also the eager backends & kernels mindmap " +
                    "(xref:explanation/eager-execution.adoc[]) for the narrative overview and gaps.",
            )
        })
    }
}
