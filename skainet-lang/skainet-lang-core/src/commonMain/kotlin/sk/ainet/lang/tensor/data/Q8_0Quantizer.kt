package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * FP32 → Q8_0 quantizer — the encode side of [Q8_0TensorData], in pure `commonMain` (SKEEP-007 §T2).
 *
 * Q8_0 is the first format quantize-at-load offers: eight-bit codes keep a classifier's decisions
 * where a four-bit format may not, the kernels exist on every tier, and the layout is the simplest
 * of the ggml family. Algorithm per 32-element block, matching ggml `quantize_row_q8_0_ref`:
 *
 *  1. `amax = max |x|` over the block.
 *  2. `d = amax / 127`, stored as the block's FP16 scale (round to nearest even).
 *  3. `code = round(x / d)`, clamped to `[-127, 127]`, one signed byte per element.
 *
 * Output is row-major blocks: 34 bytes per block (2 scale + 32 codes), blocks in element order, so
 * a `[rows, cols]` tensor's blocks run along `cols` — the orientation [Q8_0BlockTensorData] decodes
 * and the packed kernels read.
 */
public object Q8_0Quantizer {
    private const val BLOCK_SIZE = Q8_0TensorData.BLOCK_SIZE
    private const val BYTES_PER_BLOCK = Q8_0TensorData.BYTES_PER_BLOCK

    /** Quantize [values] (length a multiple of 32) into packed Q8_0 bytes — `34 * (values.size / 32)` bytes. */
    public fun quantizeToBytes(values: FloatArray): ByteArray {
        require(values.size % BLOCK_SIZE == 0) {
            "Q8_0 quantization requires a length that is a multiple of $BLOCK_SIZE; got ${values.size}"
        }
        val blocks = values.size / BLOCK_SIZE
        val out = ByteArray(blocks * BYTES_PER_BLOCK)
        for (b in 0 until blocks) {
            val base = b * BLOCK_SIZE
            var amax = 0f
            for (i in 0 until BLOCK_SIZE) {
                val a = abs(values[base + i])
                if (a > amax) amax = a
            }
            val d = amax / 127f
            val id = if (d != 0f) 1f / d else 0f
            val outBase = b * BYTES_PER_BLOCK
            val half = Fp16Codec.encode(d)
            out[outBase] = (half and 0xFF).toByte()
            out[outBase + 1] = ((half ushr 8) and 0xFF).toByte()
            for (i in 0 until BLOCK_SIZE) {
                val q = (values[base + i] * id).roundToInt()
                out[outBase + 2 + i] = (if (q < -127) -127 else if (q > 127) 127 else q).toByte()
            }
        }
        return out
    }

    /**
     * Quantize [values] into a [Q8_0BlockTensorData] with logical [shape] (`shape.volume` must equal
     * `values.size`; the last dimension must be a multiple of 32 so blocks never straddle rows).
     */
    public fun quantize(values: FloatArray, shape: Shape): Q8_0BlockTensorData {
        require(shape.volume == values.size) { "shape volume ${shape.volume} must equal values length ${values.size}" }
        require(shape.rank == 0 || shape[shape.rank - 1] % BLOCK_SIZE == 0) {
            "Q8_0 blocks run along the last dimension, which must be a multiple of $BLOCK_SIZE; got $shape"
        }
        return Q8_0BlockTensorData(shape, quantizeToBytes(values))
    }
}
