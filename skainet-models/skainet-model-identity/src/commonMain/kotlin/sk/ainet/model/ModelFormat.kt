package sk.ainet.model

/**
 * Supported model container formats.
 *
 * Moved from `sk.ainet.io.model.ModelFormat` (SKEEP-006); `skainet-io-core` keeps a deprecated
 * type alias for one release cycle.
 */
public enum class ModelFormat(public val extension: String, public val displayName: String) {
    GGUF("gguf", "GGUF"),
    ONNX("onnx", "ONNX"),
    SAFETENSORS("safetensors", "SafeTensors");

    public companion object {
        /** Find a format by file extension (case-insensitive). */
        public fun fromExtension(extension: String): ModelFormat? =
            entries.find { it.extension.equals(extension, ignoreCase = true) }

        /** Find a format from a file path. */
        public fun fromFilePath(filePath: String): ModelFormat? {
            val ext = filePath.substringAfterLast('.', "")
            return fromExtension(ext)
        }
    }
}
