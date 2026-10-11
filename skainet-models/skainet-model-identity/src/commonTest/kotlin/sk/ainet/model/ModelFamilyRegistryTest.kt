package sk.ainet.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame

class ModelFamilyRegistryTest {

    private fun gguf(arch: String) = InMemoryCheckpointView(ModelFormat.GGUF, mapOf("general.architecture" to arch))
    private fun hf(modelType: String) = InMemoryCheckpointView(ModelFormat.SAFETENSORS, mapOf("model_type" to modelType))

    @Test
    fun `gguf architecture resolves to the built-in family and variant`() {
        val id = ModelFamilyRegistry.builtIn.identify(gguf("qwen3"))
        assertEquals(ModelIdentity(ModelFormat.GGUF, "qwen3", "qwen", "qwen3"), id)
        assertEquals("qwen/qwen3", id.label)
    }

    @Test
    fun `architecture equal to the family has no variant`() {
        val id = ModelFamilyRegistry.builtIn.identify(gguf("llama"))
        assertEquals("llama", id.family)
        assertNull(id.variant)
    }

    @Test
    fun `hugging face model_type is matched case-insensitively`() {
        val id = ModelFamilyRegistry.builtIn.identify(hf("Gemma3"))
        assertEquals("gemma", id.family)
        assertEquals("gemma3", id.variant)
        assertSame(TransformerNameMaps.Hf.gemma3, ModelFamilyRegistry.builtIn.nameMapFor(id))
    }

    @Test
    fun `unknown architecture falls back to itself as family with no descriptor`() {
        val id = ModelFamilyRegistry.builtIn.identify(gguf("whisper"))
        assertEquals("whisper", id.family)
        assertNull(id.variant)
        assertNull(ModelFamilyRegistry.builtIn.descriptorFor(id))
        assertNull(ModelFamilyRegistry.builtIn.nameMapFor(id))
    }

    @Test
    fun `missing architecture identifies as unknown`() {
        val id = ModelFamilyRegistry.builtIn.identify(InMemoryCheckpointView(ModelFormat.ONNX))
        assertEquals(ModelFamilyRegistry.UNKNOWN_ARCHITECTURE, id.architecture)
    }

    @Test
    fun `plus composes immutably and later descriptors win per family`() {
        val whisper = object : ModelFamilyDescriptor {
            override val family = "whisper"
            override val architectures = setOf("whisper")
        }
        val composed = ModelFamilyRegistry.builtIn + whisper
        assertNotNull(composed.descriptorFor(composed.identify(gguf("whisper"))))
        assertNull(ModelFamilyRegistry.builtIn.descriptorForFamily("whisper"), "builtIn must be untouched")

        val llama2 = object : ModelFamilyDescriptor {
            override val family = "llama"
            override val architectures = setOf("llama", "my-llama")
        }
        val replaced = composed + llama2
        assertSame(llama2, replaced.descriptorForArchitecture("my-llama"))
        assertEquals(composed.families.size, replaced.families.size)
    }

    @Test
    fun `two families claiming one architecture is rejected`() {
        val a = object : ModelFamilyDescriptor { override val family = "a"; override val architectures = setOf("x") }
        val b = object : ModelFamilyDescriptor { override val family = "b"; override val architectures = setOf("X") }
        assertFailsWith<IllegalArgumentException> { ModelFamilyRegistry.of(a, b) }
    }

    @Test
    fun `built-in name maps follow the 0_57 string switch`() {
        val r = ModelFamilyRegistry.builtIn
        assertSame(TransformerNameMaps.Gguf.llama, r.nameMapFor(r.identify(gguf("mistral"))))
        assertSame(TransformerNameMaps.Gguf.qwen2, r.nameMapFor(r.identify(gguf("qwen2.5"))))
        assertSame(TransformerNameMaps.Gguf.gemma3, r.nameMapFor(r.identify(gguf("gemma2"))))
        @Suppress("DEPRECATION")
        assertSame(TransformerNameMaps.Gguf.llama, TransformerNameMaps.Gguf.forArchitecture("apertus"))
    }
}
