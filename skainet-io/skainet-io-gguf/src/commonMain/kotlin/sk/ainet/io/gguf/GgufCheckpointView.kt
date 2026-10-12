package sk.ainet.io.gguf

import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFormat

/**
 * A [CheckpointView] over a GGUF file's metadata (SKEEP-006, FE-I-05). GGUF carries everything
 * inline — `general.architecture`, `tokenizer.ggml.*`, `tokenizer.chat_template` — so there are
 * no side files; [sideFile] answers `null` unless extra sidecars (`frontend.json`, a
 * `tokenizer_config.json` next to the file) are supplied through [sideFiles].
 */
public class GgufCheckpointView(
    override val metadata: Map<String, Any?>,
    private val sideFiles: Map<String, ByteArray> = emptyMap(),
) : CheckpointView {
    public constructor(reader: StreamingGGUFReader, sideFiles: Map<String, ByteArray> = emptyMap()) : this(reader.fields, sideFiles)

    override val format: ModelFormat get() = ModelFormat.GGUF
    override fun sideFile(name: String): ByteArray? = sideFiles[name]
    override fun listSideFiles(): Set<String> = sideFiles.keys

    /** `general.architecture`, if present. */
    public val architecture: String? get() = metadata["general.architecture"] as? String
}

/** This reader's metadata as a [CheckpointView]. */
public fun StreamingGGUFReader.checkpointView(sideFiles: Map<String, ByteArray> = emptyMap()): GgufCheckpointView =
    GgufCheckpointView(this, sideFiles)
