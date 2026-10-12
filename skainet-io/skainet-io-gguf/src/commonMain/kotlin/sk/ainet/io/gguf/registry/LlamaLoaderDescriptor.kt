package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * Loader descriptor for LLaMA-family models (LLaMA, Mistral, Qwen, etc.).
 *
 * Register with [ModelLoaderRegistry] to enable auto-detection of LLaMA GGUF files.
 */
@Suppress("DEPRECATION")
@Deprecated("Register families with sk.ainet.model.ModelFamilyRegistry (SKEEP-006). Removed in 0.60.")
public object LlamaLoaderDescriptor : ModelLoaderDescriptor {
    override val architecture: ModelArchitecture = ModelArchitecture.LLAMA
    override val displayName: String = "LLaMA"
}
