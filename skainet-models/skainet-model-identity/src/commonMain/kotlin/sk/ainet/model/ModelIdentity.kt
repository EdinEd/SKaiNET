package sk.ainet.model

/**
 * What a checkpoint *is*, independent of the container it came in (SKEEP-006 §Identity).
 *
 * @property format the container format the identity was read from
 * @property architecture the raw architecture string the container declares: GGUF
 *   `general.architecture`, Hugging Face `config.json#model_type`, an ONNX producer tag
 * @property family the normalised family the registry resolved the architecture to
 *   (`llama`, `qwen`, `gemma`, `bert`, …); equals the lower-cased architecture when no
 *   descriptor claims it
 * @property variant the architecture when it is more specific than the family (`gemma4`,
 *   `qwen3`), otherwise `null`
 */
public data class ModelIdentity(
    val format: ModelFormat,
    val architecture: String,
    val family: String,
    val variant: String? = null,
) {
    /** `family` or, when present, `family/variant` — for logs and exception messages. */
    public val label: String get() = if (variant == null) family else "$family/$variant"
}
