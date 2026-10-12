package sk.ainet.frontend

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontEndSpecTest {

    @Test
    fun `text spec round trips through json`() {
        val spec = FrontEndSpec.Text("bpe", "tokenizer.json", JsonObject(mapOf("byte_level" to JsonPrimitive(true))))
        val json = spec.toJson()
        assertTrue("\"modality\":\"text\"" in json, json)
        assertEquals(spec, FrontEndSpec.fromJson(json))
    }

    @Test
    fun `frontend_json sidecar may hold one spec or a list`() {
        val one = FrontEndSpec.listFromJson("""{"modality":"image","type":"letterbox","params":{"size":640}}""")
        assertEquals(1, one.size)
        assertEquals("letterbox", (one[0] as FrontEndSpec.Image).type)
        val list = FrontEndSpec.listFromJson("""[{"modality":"text","algorithm":"wordpiece","artifact":"vocab.txt"},{"modality":"audio","type":"log-mel"}]""")
        assertEquals(listOf(Modality.TEXT, Modality.AUDIO), list.map { it.modality })
    }

    @Test
    fun `composite spec lists its parts`() {
        val c = FrontEndSpec.Composite(listOf(FrontEndSpec.Text("unigram"), FrontEndSpec.Image("hf-image-processor")), "gemma3")
        val back = FrontEndSpec.fromJson(c.toJson()) as FrontEndSpec.Composite
        assertEquals(2, back.parts.size)
        assertEquals("gemma3", back.policy)
    }
}
