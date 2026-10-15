package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * Loader descriptor for Gemma-family models (Gemma, Gemma 3, Gemma 3n).
 *
 * Register with [ModelLoaderRegistry] to enable auto-detection of Gemma GGUF files.
 */
@Suppress("DEPRECATION", "DEPRECATION_ERROR")
@Deprecated("Register families with sk.ainet.model.ModelFamilyRegistry (SKEEP-006). An error since 0.60, removed in 0.61.", level = DeprecationLevel.ERROR)
public object Gemma3nLoaderDescriptor : ModelLoaderDescriptor {
    override val architecture: ModelArchitecture = ModelArchitecture.GEMMA
    override val displayName: String = "Gemma 3n"
}
