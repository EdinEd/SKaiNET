package sk.ainet.model

import sk.ainet.model.families.BertFamilyDescriptor
import sk.ainet.model.families.GemmaFamilyDescriptor
import sk.ainet.model.families.LlamaFamilyDescriptor
import sk.ainet.model.families.QwenFamilyDescriptor

/**
 * Immutable set of [ModelFamilyDescriptor]s (SKEEP-006 §Identity).
 *
 * Compose with [plus]; there is deliberately no global mutable `register()` and no service
 * loading. SKaiNET-transformers builds `ModelFamilyRegistry.builtIn + llamaFamily + …` once and
 * hands it to loaders and resolvers.
 */
public class ModelFamilyRegistry private constructor(
    private val descriptors: List<ModelFamilyDescriptor>,
) {
    init {
        val seen = HashMap<String, String>()
        for (d in descriptors) {
            for (a in d.architectures) {
                val key = a.lowercase()
                val prev = seen.put(key, d.family)
                require(prev == null || prev == d.family) {
                    "architecture '$a' is claimed by both family '$prev' and family '${d.family}'"
                }
            }
        }
    }

    /** Families in registration order. */
    public val families: List<String> get() = descriptors.map { it.family }

    /** The descriptors in registration order. */
    public val all: List<ModelFamilyDescriptor> get() = descriptors

    /** A registry with [d] appended; a descriptor for an already registered family replaces it. */
    public operator fun plus(d: ModelFamilyDescriptor): ModelFamilyRegistry =
        ModelFamilyRegistry(descriptors.filter { it.family != d.family } + d)

    /** A registry with every descriptor of [others] appended (later entries win per family). */
    public operator fun plus(others: Iterable<ModelFamilyDescriptor>): ModelFamilyRegistry =
        others.fold(this) { acc, d -> acc + d }

    /** The descriptor claiming [architecture], or `null`. */
    public fun descriptorForArchitecture(architecture: String): ModelFamilyDescriptor? =
        descriptors.firstOrNull { d -> d.architectures.any { it.equals(architecture, ignoreCase = true) } }

    /** The descriptor named [family], or `null`. */
    public fun descriptorForFamily(family: String): ModelFamilyDescriptor? =
        descriptors.firstOrNull { it.family.equals(family, ignoreCase = true) }

    /** The descriptor of [identity]'s family, or `null` when no registered family claims it. */
    public fun descriptorFor(identity: ModelIdentity): ModelFamilyDescriptor? =
        descriptorForFamily(identity.family)

    /**
     * Resolve the identity of [view]: the raw architecture from the container metadata
     * (GGUF `general.architecture`, Hugging Face `model_type`, or a plain `architecture` key),
     * the family of the descriptor that claims it (or the lower-cased architecture when none does)
     * and the variant when the architecture is more specific than the family.
     */
    public fun identify(view: CheckpointView): ModelIdentity {
        val architecture = architectureOf(view.metadata) ?: UNKNOWN_ARCHITECTURE
        val descriptor = descriptorForArchitecture(architecture)
        val family = descriptor?.family ?: architecture.lowercase()
        val variant = architecture.lowercase().takeIf { it != family.lowercase() }
        return ModelIdentity(view.format, architecture, family, variant)
    }

    /** The weight [NameMap] for [identity], asked of its family descriptor. */
    public fun nameMapFor(identity: ModelIdentity): NameMap? =
        descriptorFor(identity)?.nameMap(identity.format)

    public companion object {
        public const val UNKNOWN_ARCHITECTURE: String = "unknown"

        /** Metadata keys consulted for the raw architecture, in order. */
        public val ARCHITECTURE_KEYS: List<String> = listOf("general.architecture", "model_type", "architecture")

        /** The raw architecture string of a metadata map, or `null`. */
        public fun architectureOf(metadata: Map<String, Any?>): String? {
            for (key in ARCHITECTURE_KEYS) {
                val v = metadata[key] as? String ?: continue
                if (v.isNotBlank()) return v
            }
            // Hugging Face config.json: "architectures": ["LlamaForCausalLM"] — only as a last resort.
            val archs = metadata["architectures"] as? List<*>
            val first = archs?.firstOrNull() as? String
            return first?.takeIf { it.isNotBlank() }
        }

        /** An empty registry; every view identifies to its lower-cased architecture. */
        public val empty: ModelFamilyRegistry = ModelFamilyRegistry(emptyList())

        /**
         * The families the engine ships name maps or tokenizer dispatch for. Downstream
         * repositories compose their own descriptors on top (`builtIn + …`).
         */
        public val builtIn: ModelFamilyRegistry = ModelFamilyRegistry(
            listOf(LlamaFamilyDescriptor, QwenFamilyDescriptor, GemmaFamilyDescriptor, BertFamilyDescriptor),
        )

        /** A registry from explicit descriptors. */
        public fun of(vararg descriptors: ModelFamilyDescriptor): ModelFamilyRegistry =
            ModelFamilyRegistry(emptyList()) + descriptors.asList()
    }
}
