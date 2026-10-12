package sk.ainet.io.safetensors

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import sk.ainet.io.openRandomAccessSource
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFormat

/**
 * A [CheckpointView] over a Hugging Face model directory (SKEEP-006, FE-I-05): `config.json`
 * flattened into [metadata] (so `model_type`, `architectures`, `vocab_size`, … are plain keys)
 * and the sidecar files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
 * `tekken.json`, `vocab.txt`, `preprocessor_config.json`, `frontend.json`) as side files.
 *
 * The common constructor takes the file contents, so a browser or an in-memory caller can build
 * one; [fromDirectory] reads a directory on targets with a file system.
 */
public class HfDirectoryCheckpointView(
    configJson: String?,
    private val sideFiles: Map<String, ByteArray>,
    /** Where the files came from, for messages; `null` for in-memory views. */
    public val directory: String? = null,
) : CheckpointView {

    override val format: ModelFormat get() = ModelFormat.SAFETENSORS

    override val metadata: Map<String, Any?> = configJson?.let { flatten(JSON.parseToJsonElement(it)) } ?: emptyMap()

    override fun sideFile(name: String): ByteArray? = sideFiles[name]
    override fun listSideFiles(): Set<String> = sideFiles.keys

    /** `config.json#model_type`, if present. */
    public val modelType: String? get() = metadata["model_type"] as? String

    public companion object {
        internal val JSON: Json = Json { ignoreUnknownKeys = true; isLenient = true }

        public const val CONFIG_JSON: String = "config.json"

        /**
         * Read `config.json` and every known sidecar from [directory]. Returns a view with empty
         * metadata and no side files when the directory has none of them (the caller's resolver
         * then throws a message naming the directory).
         */
        public fun fromDirectory(directory: String, sideFileNames: Set<String> = CheckpointView.KNOWN_SIDE_FILES): HfDirectoryCheckpointView {
            val dir = directory.trimEnd('/', '\\')
            val config = readTextFile("$dir/$CONFIG_JSON")
            val files = LinkedHashMap<String, ByteArray>()
            for (name in sideFileNames) readBytes("$dir/$name")?.let { files[name] = it }
            return HfDirectoryCheckpointView(config, files, dir)
        }

        private fun readBytes(path: String): ByteArray? {
            val source = openRandomAccessSource(path) ?: return null
            return source.use { s ->
                val size = s.size
                if (size > Int.MAX_VALUE) return null
                s.readAt(0, size.toInt())
            }
        }

        /** `config.json` as a flat map: primitives keep their type, objects nest as maps, arrays as lists. */
        internal fun flatten(element: JsonElement): Map<String, Any?> =
            (element as? JsonObject)?.mapValues { (_, v) -> toKotlin(v) } ?: emptyMap()

        private fun toKotlin(e: JsonElement): Any? = when (e) {
            is JsonNull -> null
            is JsonPrimitive -> when {
                e.isString -> e.content
                e.booleanOrNull != null -> e.booleanOrNull
                e.longOrNull != null -> e.longOrNull
                e.doubleOrNull != null -> e.doubleOrNull
                else -> e.content
            }
            is JsonArray -> e.map { toKotlin(it) }
            is JsonObject -> e.mapValues { (_, v) -> toKotlin(v) }
        }
    }
}
