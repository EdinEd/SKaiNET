package sk.ainet.io.safetensors

import kotlinx.coroutines.runBlocking
import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.ByteArrayRandomAccessSource
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * SKEEP-007 §T2: the SafeTensors loader honours `WeightForm`, including `RequantizeTo(Q8_0)`.
 *
 * Every Hugging Face checkpoint is dense F32/F16/BF16; until now only a GGUF could be served packed.
 * These tests pin what the request applies to (rank-2 float tensors whose rows block), what it leaves
 * alone (vectors), feed-order relayout, per-tensor overrides, and the up-front refusal of a request
 * no quantizer can honour.
 */
class SafeTensorsRequantizeTest {
    private val ctx = DefaultDataExecutionContext()

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 400f
        }
    }

    private val w = pseudo(4 * 64, 1)      // [4, 64]: two Q8_0 blocks per row
    private val e = pseudo(8 * 96, 2)      // [8, 96]: three blocks per row, F16 on disk
    private val bias = pseudo(4, 3)        // [4]: a vector, never requantized
    private val odd = pseudo(3 * 40, 4)    // [3, 40]: rows do not block → stays dense

    private fun f16(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun file(): ByteArray {
        val buffer = Buffer()
        SafeTensorsWriter.write(buffer) {
            tensorF32("w", intArrayOf(4, 64), w)
            tensor("e", "F16", intArrayOf(8, 96)) { f16(e) }
            tensorF32("bias", intArrayOf(4), bias)
            tensorF32("odd", intArrayOf(3, 40), odd)
        }
        return buffer.readByteArray()
    }

    private fun load(form: WeightForm?, formFor: ((String) -> WeightForm?)? = null): Map<String, Tensor<FP32, Float>> {
        val bytes = file()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            SafeTensorsParametersLoader(
                sourceProvider = { ByteArrayRandomAccessSource(bytes) },
                weightForm = form,
                weightFormFor = formFor,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tolerance: Float, what: String) {
        assertEquals(expected.size, actual.size, "$what size")
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tolerance, "$what[$i]: ${actual[i]} vs ${expected[i]}")
        }
    }

    @Test
    fun `a uniform RequantizeTo Q8_0 packs the matrices and leaves vectors and non-blocking rows dense`() {
        val form = WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0), shape = WeightShapeOrientation.OUT_IN)
        val loaded = load(form)
        val packedW = assertIs<Q8_0BlockTensorData>(loaded.getValue("w").data)
        assertEquals(BlockOrder.ROW_MAJOR, packedW.blockOrder)
        assertClose(w, packedW.toFloatArray(), tolerance = 2.5f / 127f + 1e-3f, what = "w")
        val packedE = assertIs<Q8_0BlockTensorData>(loaded.getValue("e").data)
        assertClose(e, packedE.toFloatArray(), tolerance = 2.5f / 127f + 4e-3f, what = "e (F16 source)")
        assertIs<FloatArrayTensorData<*>>(loaded.getValue("bias").data, "a vector is not a weight matrix")
        assertIs<FloatArrayTensorData<*>>(loaded.getValue("odd").data, "rows of 40 do not block by 32")
    }

    @Test
    fun `KERNEL_FEED relays the fresh blocks into input-block-major order with the same values`() {
        val form = WeightForm(
            encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0),
            order = WeightByteOrder.KERNEL_FEED,
            shape = WeightShapeOrientation.OUT_IN,
        )
        val fed = assertIs<Q8_0BlockTensorData>(load(form).getValue("w").data)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, fed.blockOrder)
        val canonical = assertIs<Q8_0BlockTensorData>(
            load(WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0), shape = WeightShapeOrientation.OUT_IN)).getValue("w").data,
        )
        assertClose(canonical.toFloatArray(), fed.toFloatArray(), tolerance = 0f, what = "feed order decodes to the same values")
    }

    @Test
    fun `weightFormFor overrides the uniform form per tensor`() {
        val loaded = load(
            form = null,
            formFor = { name -> if (name == "w") WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0)) else null },
        )
        assertIs<Q8_0BlockTensorData>(loaded.getValue("w").data)
        assertIs<FloatArrayTensorData<*>>(loaded.getValue("e").data, "no form → the F16 policy decides (widen)")
    }

    @Test
    fun `requests no quantizer can honour are refused at construction`() {
        val bytes = file()
        val noQuantizer = assertFailsWith<IllegalArgumentException> {
            SafeTensorsParametersLoader(
                sourceProvider = { ByteArrayRandomAccessSource(bytes) },
                weightForm = WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q4_K)),
            )
        }
        assertTrue(noQuantizer.message!!.contains("quantizer"), noQuantizer.message!!)
        val narrow = assertFailsWith<IllegalArgumentException> {
            SafeTensorsParametersLoader(
                sourceProvider = { ByteArrayRandomAccessSource(bytes) },
                weightForm = WeightForm(encoding = EncodingRequest.DequantizeTo(FP16)),
            )
        }
        assertTrue(narrow.message!!.contains("FP32"), narrow.message!!)
    }
}
