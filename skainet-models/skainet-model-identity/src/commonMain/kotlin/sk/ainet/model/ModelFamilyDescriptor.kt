package sk.ainet.model

/**
 * What the engine knows about one model family (SKEEP-006 §Identity): which raw architecture
 * strings belong to it and how its checkpoint tensors map onto SKaiNET ids per format.
 *
 * Family *policy* for input front ends (special tokens, stop set, prompt renderer, placeholder
 * expansion) is declared by additionally implementing `sk.ainet.frontend.FamilyFrontEnds` from
 * `skainet-frontend-core`; this interface stays free of front-end types so identity can sit below
 * every reader module.
 */
public interface ModelFamilyDescriptor {
    /** Normalised family name: `llama`, `qwen`, `gemma`, `bert`, … */
    public val family: String

    /** Raw architecture strings this family claims (matched case-insensitively). */
    public val architectures: Set<String>

    /** The weight [NameMap] for [format], or `null` when the family has none for it. */
    public fun nameMap(format: ModelFormat): NameMap? = null
}
