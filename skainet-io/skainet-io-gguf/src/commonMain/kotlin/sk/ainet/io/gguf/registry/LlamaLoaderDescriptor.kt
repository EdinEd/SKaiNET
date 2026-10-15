package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * Loader descriptor for LLaMA-family models (LLaMA, Mistral, Qwen, etc.).
 *
 * Register with [ModelLoaderRegistry] to enable auto-detection of LLaMA GGUF files.
 */
@Suppress("DEPRECATION", "DEPRECATION_ERROR")
@Deprecated("Register families with sk.ainet.model.ModelFamilyRegistry (SKEEP-006). An error since 0.60, removed in 0.61.", level = DeprecationLevel.ERROR)
public object LlamaLoaderDescriptor : ModelLoaderDescriptor {
    override val architecture: ModelArchitecture = ModelArchitecture.LLAMA
    override val displayName: String = "LLaMA"
}
