package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** SKEEP-007 §T2: the Q8_0 encoder is ggml-faithful and round-trips through the engine's decoder. */
class Q8_0QuantizerTest {

    private fun pseudo(n: Int, seed: Int, scale: Float = 1f): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 1000f * scale
        }
    }

    @Test
    fun `one block matches the ggml reference layout`() {
        // amax = 2.54 → d = 0.02, codes = round(x / d)
        val values = FloatArray(32) { i -> (i - 16) * 0.1f }.also { it[5] = 2.54f }
        val bytes = Q8_0Quantizer.quantizeToBytes(values)
        assertEquals(34, bytes.size)
        val halfBits = (bytes[0].toInt() and 0xFF) or ((bytes[1].toInt() and 0xFF) shl 8)
        val d = Fp16Codec.decode(halfBits)
        assertTrue(abs(d - 2.54f / 127f) < 1e-5f, "scale $d")
        assertEquals(127, bytes[2 + 5].toInt())
        assertEquals((-1.6f / (2.54f / 127f)).let { kotlin.math.round(it).toInt() }, bytes[2 + 0].toInt())
    }

    @Test
    fun `round trip through Q8_0BlockTensorData stays within one code step`() {
        val rows = 6; val cols = 96
        val values = pseudo(rows * cols, 17, scale = 3f)
        val data = Q8_0Quantizer.quantize(values, Shape(rows, cols))
        assertEquals(TensorEncoding.Q8_0, data.encoding)
        val back = data.toFloatArray()
        var maxErr = 0f
        var blockAmax = 0f
        for (b in 0 until rows * cols / 32) {
            blockAmax = 0f
            for (i in 0 until 32) blockAmax = maxOf(blockAmax, abs(values[b * 32 + i]))
            val step = blockAmax / 127f
            for (i in 0 until 32) {
                val idx = b * 32 + i
                val err = abs(back[idx] - values[idx])
                maxErr = maxOf(maxErr, err)
                assertTrue(err <= step * 0.5f + 1e-3f * blockAmax, "element $idx: |${back[idx]} - ${values[idx]}| = $err > half step $step")
            }
        }
        assertTrue(maxErr > 0f, "quantization is lossy; a zero error means the decoder was not exercised")
    }

    @Test
    fun `an all-zero block has a zero scale and zero codes`() {
        val bytes = Q8_0Quantizer.quantizeToBytes(FloatArray(32))
        assertTrue(bytes.all { it == 0.toByte() })
    }

    @Test
    fun `shapes that do not block along the last dimension are refused`() {
        assertFailsWith<IllegalArgumentException> { Q8_0Quantizer.quantizeToBytes(FloatArray(33)) }
        assertFailsWith<IllegalArgumentException> { Q8_0Quantizer.quantize(FloatArray(64), Shape(4, 16)) }
        assertFailsWith<IllegalArgumentException> { Q8_0Quantizer.quantize(FloatArray(64), Shape(2, 64)) }
    }

    @Test
    fun `registry routes Q8_0 and Q4_0 and refuses unknown encodings`() {
        val values = pseudo(64, 3)
        assertTrue(Requantizers.quantize(TensorEncoding.Q8_0, values, Shape(2, 32)) is Q8_0BlockTensorData)
        assertTrue(Requantizers.quantize(TensorEncoding.Q4_0, values, Shape(2, 32)) is Q4_0BlockTensorData)
        assertTrue(Requantizers.supports(TensorEncoding.BITNET_PLANES))
        assertTrue(!Requantizers.supports(TensorEncoding.Q4_K))
        assertFailsWith<IllegalArgumentException> { Requantizers.quantize(TensorEncoding.Q4_K, values, Shape(2, 32)) }
    }
}
