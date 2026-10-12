package sk.ainet.frontend

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * The serialisable twin of a resolved front end (FE-C-04): sufficient to re-create it from the
 * same side files, what a `frontend.json` sidecar carries, what a cartridge manifest can embed,
 * and what [FrontEndResolver.describe] returns.
 *
 * JSON form: `{"modality":"text","algorithm":"bpe","artifact":"tokenizer.json"}`; the `modality`
 * key is the class discriminator so the audio and image cases keep their `type` field.
 */
@Serializable
public sealed class FrontEndSpec {
    public abstract val modality: Modality

    /**
     * @property algorithm `bpe`, `metaspace-bpe`, `unigram`, `wordpiece`, `tekken`, …
     * @property artifact the side file the tokenizer was built from (`tokenizer.json`,
     *   `tekken.json`, `vocab.txt`) or `gguf` for container metadata; `null` for in-memory
     *   construction
     */
    @Serializable
    @SerialName("text")
    public data class Text(
        val algorithm: String,
        val artifact: String? = null,
        val params: JsonObject = JsonObject(emptyMap()),
    ) : FrontEndSpec() {
        override val modality: Modality get() = Modality.TEXT
    }

    /** @property type `log-mel` or `module` */
    @Serializable
    @SerialName("audio")
    public data class Audio(
        val type: String,
        val params: JsonObject = JsonObject(emptyMap()),
        val artifact: String? = null,
    ) : FrontEndSpec() {
        override val modality: Modality get() = Modality.AUDIO
    }

    /** @property type `letterbox` or `hf-image-processor` */
    @Serializable
    @SerialName("image")
    public data class Image(
        val type: String,
        val params: JsonObject = JsonObject(emptyMap()),
    ) : FrontEndSpec() {
        override val modality: Modality get() = Modality.IMAGE
    }

    /** A multimodal composition: its parts plus the name of the family's placeholder policy. */
    @Serializable
    @SerialName("composite")
    public data class Composite(
        val parts: List<FrontEndSpec>,
        val policy: String,
    ) : FrontEndSpec() {
        override val modality: Modality get() = Modality.COMPOSITE
    }

    public fun toJson(): String = JSON.encodeToString(serializer(), this)

    public companion object {
        internal val JSON: Json = Json { ignoreUnknownKeys = true; encodeDefaults = true; classDiscriminator = "modality" }

        public fun fromJson(json: String): FrontEndSpec = JSON.decodeFromString(serializer(), json)

        /** Parse a `frontend.json` sidecar: either one spec or a JSON array of specs. */
        public fun listFromJson(json: String): List<FrontEndSpec> {
            val trimmed = json.trimStart()
            return if (trimmed.startsWith("[")) {
                JSON.decodeFromString(kotlinx.serialization.builtins.ListSerializer(serializer()), json)
            } else {
                listOf(fromJson(json))
            }
        }
    }
}

/** Input modalities a front end can serve. */
public enum class Modality { TEXT, AUDIO, IMAGE, COMPOSITE }
