package sk.ainet.model.families

import sk.ainet.model.ModelFamilyDescriptor
import sk.ainet.model.ModelFormat
import sk.ainet.model.NameMap
import sk.ainet.model.TransformerNameMaps

/** Llama-shaped checkpoints (Llama 2/3/4, Mistral, SmolLM, TinyLlama). */
public object LlamaFamilyDescriptor : ModelFamilyDescriptor {
    override val family: String = "llama"
    override val architectures: Set<String> = setOf("llama", "llama4", "mistral", "smollm", "tinyllama")
    override fun nameMap(format: ModelFormat): NameMap? = when (format) {
        ModelFormat.GGUF -> TransformerNameMaps.Gguf.llama
        ModelFormat.SAFETENSORS -> TransformerNameMaps.Hf.llama
        ModelFormat.ONNX -> null
    }
}

/** Qwen 2 / 2.5 / 3 checkpoints. */
public object QwenFamilyDescriptor : ModelFamilyDescriptor {
    override val family: String = "qwen"
    override val architectures: Set<String> = setOf("qwen2", "qwen2.5", "qwen3")
    override fun nameMap(format: ModelFormat): NameMap? = when (format) {
        ModelFormat.GGUF -> TransformerNameMaps.Gguf.qwen2
        ModelFormat.SAFETENSORS -> TransformerNameMaps.Hf.qwen2
        ModelFormat.ONNX -> null
    }
}

/** Gemma checkpoints (Gemma, Gemma 2, Gemma 3, Gemma 3n, Gemma 4). */
public object GemmaFamilyDescriptor : ModelFamilyDescriptor {
    override val family: String = "gemma"
    override val architectures: Set<String> = setOf("gemma", "gemma2", "gemma3", "gemma3n", "gemma3_text", "gemma4")
    override fun nameMap(format: ModelFormat): NameMap? = when (format) {
        ModelFormat.GGUF -> TransformerNameMaps.Gguf.gemma3
        ModelFormat.SAFETENSORS -> TransformerNameMaps.Hf.gemma3
        ModelFormat.ONNX -> null
    }
}

/** BERT-shaped encoders (BERT, ModernBERT and friends); WordPiece or byte-level BPE tokenizers. */
public object BertFamilyDescriptor : ModelFamilyDescriptor {
    override val family: String = "bert"
    override val architectures: Set<String> = setOf("bert", "modernbert", "distilbert", "roberta")
}
