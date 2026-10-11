package sk.ainet.model

/**
 * Read-only window on a checkpoint plus its side files; the only thing identity and front-end
 * resolution ever see. One implementation per container format lives with that format's reader
 * (`GgufCheckpointView` in `skainet-io-gguf`, `HfDirectoryCheckpointView` in
 * `skainet-io-safetensors`); [InMemoryCheckpointView] serves tests and in-process callers.
 */
public interface CheckpointView {
    public val format: ModelFormat

    /**
     * Container metadata: the GGUF field map, the flattened `config.json` of a Hugging Face
     * directory, or the `metadata_props` of an ONNX model. Values keep the reader's runtime types
     * (strings, numbers, booleans, lists, nested maps).
     */
    public val metadata: Map<String, Any?>

    /**
     * Bytes of a side file by its conventional name — `tokenizer.json`, `tokenizer_config.json`,
     * `special_tokens_map.json`, `tekken.json`, `vocab.txt`, `preprocessor_config.json`,
     * `frontend.json` — or `null` when the checkpoint has no such file.
     */
    public fun sideFile(name: String): ByteArray?

    /** Names every [sideFile] call would answer non-null for. */
    public fun listSideFiles(): Set<String>

    /** Convenience: [sideFile] decoded as UTF-8. */
    public fun sideFileText(name: String): String? = sideFile(name)?.decodeToString()

    public companion object {
        public const val TOKENIZER_JSON: String = "tokenizer.json"
        public const val TOKENIZER_CONFIG_JSON: String = "tokenizer_config.json"
        public const val SPECIAL_TOKENS_MAP_JSON: String = "special_tokens_map.json"
        public const val TEKKEN_JSON: String = "tekken.json"
        public const val VOCAB_TXT: String = "vocab.txt"
        public const val PREPROCESSOR_CONFIG_JSON: String = "preprocessor_config.json"
        public const val FRONTEND_JSON: String = "frontend.json"

        /** The side-file names a directory-backed view probes for by default. */
        public val KNOWN_SIDE_FILES: Set<String> = setOf(
            TOKENIZER_JSON, TOKENIZER_CONFIG_JSON, SPECIAL_TOKENS_MAP_JSON, TEKKEN_JSON, VOCAB_TXT,
            PREPROCESSOR_CONFIG_JSON, FRONTEND_JSON,
        )
    }
}

/** A [CheckpointView] over values already in memory. */
public class InMemoryCheckpointView(
    override val format: ModelFormat,
    override val metadata: Map<String, Any?> = emptyMap(),
    private val sideFiles: Map<String, ByteArray> = emptyMap(),
) : CheckpointView {
    override fun sideFile(name: String): ByteArray? = sideFiles[name]
    override fun listSideFiles(): Set<String> = sideFiles.keys

    public companion object {
        /** Build from UTF-8 text side files. */
        public fun ofText(
            format: ModelFormat,
            metadata: Map<String, Any?> = emptyMap(),
            sideFiles: Map<String, String> = emptyMap(),
        ): InMemoryCheckpointView =
            InMemoryCheckpointView(format, metadata, sideFiles.mapValues { it.value.encodeToByteArray() })
    }
}
