import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.*
import java.io.File

/**
 * Fails the build when production Kotlin code throws `NotImplementedError` / calls `TODO(`
 * without the gap being declared, so every missing op or layer shows up in the operator
 * coverage matrix and has a tracking issue.
 *
 * A stub is declared when either
 *  - it sits inside a declaration annotated `@NotImplemented` / `@InProgress` (the KSP docs
 *    processor renders these as ❌ / 🚧), or
 *  - its file is listed in [allowlistFile] as `path|count|issue or reason`. Non-op gaps
 *    (loaders, platform actuals, DSL) use this; `count` is the exact number of unmarked stubs
 *    expected in the file, so adding one more fails until the entry is updated.
 *
 * Comment lines are ignored, so `// TODO:` notes are not stubs.
 */
@CacheableTask
abstract class CheckUnmarkedStubsTask : DefaultTask() {

    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sources: ConfigurableFileCollection

    /** Repository root that report and allowlist paths are relative to. */
    @get:Internal
    abstract val rootDirectory: DirectoryProperty

    @get:InputFile
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val allowlistFile: RegularFileProperty

    @get:OutputFile
    abstract val reportFile: RegularFileProperty

    private val stubCall = Regex("""\bNotImplementedError\s*\(|\bTODO\s*\(""")
    private val mainSourceDir = Regex("""/src/[A-Za-z0-9]*[Mm]ain/""")
    private val marker = Regex("""^\s*@(NotImplemented|InProgress)\b""")

    @TaskAction
    fun check() {
        val root = rootDirectory.get().asFile
        val allowlist = parseAllowlist(allowlistFile.get().asFile)
        val unmarked = mutableMapOf<String, MutableList<Int>>()

        sources.files
            .filter { it.extension == "kt" }
            .forEach { file ->
                val rel = file.relativeTo(root).invariantSeparatorsPath
                if (!mainSourceDir.containsMatchIn("/$rel")) return@forEach
                val lines = file.readLines()
                val covered = markedRanges(lines)
                lines.forEachIndexed { index, line ->
                    val trimmed = line.trimStart()
                    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) return@forEachIndexed
                    // Mentions inside string literals (messages, regexes) are not calls.
                    if (!stubCall.containsMatchIn(stripStrings(line))) return@forEachIndexed
                    if (covered.any { index in it }) return@forEachIndexed
                    unmarked.getOrPut(rel) { mutableListOf() } += index + 1
                }
            }

        val problems = mutableListOf<String>()
        unmarked.toSortedMap().forEach { (path, hits) ->
            val expected = allowlist[path]
            when {
                expected == null -> hits.forEach { problems += "$path:$it — unmarked stub" }
                expected != hits.size -> problems += "$path — ${hits.size} unmarked stubs, allowlist expects $expected (lines ${hits.joinToString()})"
            }
        }
        allowlist.keys.filter { it !in unmarked }.sorted().forEach { path ->
            problems += "$path — allowlisted but has no unmarked stubs any more; remove the entry"
        }

        reportFile.get().asFile.apply { parentFile.mkdirs() }.writeText(
            unmarked.toSortedMap().entries.joinToString("\n", postfix = "\n") { (path, hits) -> "$path ${hits.joinToString(",")}" },
        )

        if (problems.isNotEmpty()) {
            throw GradleException(
                buildString {
                    appendLine("Unmarked NotImplementedError / TODO( stubs in production code:")
                    problems.forEach { appendLine("  $it") }
                    appendLine()
                    appendLine("Mark a missing op or layer with @NotImplemented(reason = …, issue = …) so it shows in")
                    appendLine("docs/modules/ROOT/pages/reference/ops-status-matrix.adoc, or add/update the file's entry in")
                    appendLine("${allowlistFile.get().asFile.relativeTo(root).invariantSeparatorsPath} (path|count|issue).")
                },
            )
        }
        logger.lifecycle("✅ No unmarked stubs (${unmarked.values.sumOf { it.size }} allowlisted in ${unmarked.size} files)")
    }

    /** Blank out the contents of `"…"` literals on one line (escape-aware). */
    private fun stripStrings(line: String): String = buildString {
        var inString = false
        var i = 0
        while (i < line.length) {
            val ch = line[i]
            when {
                inString && ch == '\\' -> i++
                ch == '"' -> { inString = !inString; append(ch) }
                !inString -> append(ch)
            }
            i++
        }
    }

    private fun parseAllowlist(file: File): Map<String, Int> =
        file.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .associate { line ->
                val parts = line.split("|").map { it.trim() }
                if (parts.size < 3 || parts[2].isEmpty()) {
                    throw GradleException("Malformed allowlist line (want path|count|issue or reason): $line")
                }
                parts[0] to (parts[1].toIntOrNull() ?: throw GradleException("Bad count in allowlist line: $line"))
            }

    /**
     * Line ranges (0-based, inclusive) of declarations carrying a status marker: from the
     * annotation to the matching closing brace, or to the `= throw` / `= TODO(` line of an
     * expression-bodied function.
     */
    private fun markedRanges(lines: List<String>): List<IntRange> {
        val ranges = mutableListOf<IntRange>()
        lines.forEachIndexed { start, line ->
            if (!marker.containsMatchIn(line)) return@forEachIndexed
            var depth = 0
            var opened = false
            var end = lines.lastIndex
            loop@ for (i in start until lines.size) {
                val text = lines[i]
                if (!opened && (text.contains("= throw") || text.contains("= TODO("))) { end = i; break@loop }
                for (ch in text) {
                    if (ch == '{') { depth++; opened = true }
                    if (ch == '}') depth--
                    if (opened && depth == 0) { end = i; break@loop }
                }
            }
            ranges += start..end
        }
        return ranges
    }
}
