package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * GGUF-specific extension: resolve a `general.architecture` string to a [ModelArchitecture].
 */
private val ggufIdMap: Map<String, ModelArchitecture> = mapOf(
    "llama" to ModelArchitecture.LLAMA,
    "gemma4" to ModelArchitecture.GEMMA,
    "gemma3n" to ModelArchitecture.GEMMA,
    "gemma3" to ModelArchitecture.GEMMA,
    "gemma" to ModelArchitecture.GEMMA,
    "bert" to ModelArchitecture.BERT,
    "qwen2" to ModelArchitecture.QWEN,
)

/**
 * Resolve a GGUF `general.architecture` value to a [ModelArchitecture].
 *
 * @param arch The architecture string from GGUF metadata (may be null)
 * @return The matching architecture, or [UNKNOWN][ModelArchitecture.UNKNOWN] if not recognized
 */
@Deprecated(
    "The enum cannot grow with new families; identify checkpoints through sk.ainet.model.ModelFamilyRegistry (SKEEP-006). Removed in 0.60.",
    ReplaceWith("ModelFamilyRegistry.builtIn.identify(view)", "sk.ainet.model.ModelFamilyRegistry"),
)
public fun ModelArchitecture.Companion.fromGguf(arch: String?): ModelArchitecture =
    arch?.let { a -> ggufIdMap.entries.firstOrNull { it.key.equals(a, ignoreCase = true) }?.value }
        ?: ModelArchitecture.UNKNOWN
