import models.DocumentationFormat
import org.gradle.api.Project
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property

open class DocumentationExtension(project: Project) {
    val inputFile: RegularFileProperty = project.objects.fileProperty()
    val outputDirectory: DirectoryProperty = project.objects.directoryProperty()
    val templateDirectory: DirectoryProperty = project.objects.directoryProperty()
    val format: Property<DocumentationFormat> = project.objects.property(DocumentationFormat::class.java)
        .convention(DocumentationFormat.ASCIIDOC)
    val includeBackendStatus: Property<Boolean> = project.objects.property(Boolean::class.java)
        .convention(true)
    val generateIndex: Property<Boolean> = project.objects.property(Boolean::class.java)
        .convention(true)

    /**
     * operators.json files emitted by running the docs KSP processor in backend modules
     * (e.g. skainet-backend-cpu). Only their `statusByBackend` maps and notes are merged into
     * [inputFile]; missing files are skipped.
     */
    val backendStatusFiles: ConfigurableFileCollection = project.objects.fileCollection()

    /** hlo-coverage.json (op names the StableHLO converter lowers), from HloCoverageDumpTest. */
    val hloCoverageFile: RegularFileProperty = project.objects.fileProperty()

    /** backward-coverage.json (op name → implemented / stubbed / missing), from BackwardCoverageDumpTest. */
    val backwardCoverageFile: RegularFileProperty = project.objects.fileProperty()

    /** Base URL issue references like `#123` resolve against in rendered matrix cells. */
    val issueBaseUrl: Property<String> = project.objects.property(String::class.java)
        .convention("https://github.com/SKaiNET-developers/SKaiNET/issues/")

    /** kernel-support.json emitted by KernelSupportMatrixTest (registry introspection). */
    val kernelInputFile: RegularFileProperty = project.objects.fileProperty()

    /** Rendered kernel × platform matrix Antora page. */
    val kernelOutputFile: RegularFileProperty = project.objects.fileProperty()
}