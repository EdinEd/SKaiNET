package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The quantizers a loader can run at load time, keyed by target encoding (SKEEP-007 §T2).
 *
 * `EncodingRequest.RequantizeTo(encoding)` has been part of [sk.ainet.lang.memory.plan.WeightForm]
 * since #1109 and the planner prices it, but only the GGUF loader executed it and only for
 * `BITNET_PLANES` (#1150). This registry is what every loader consults instead of hard-coding one
 * format: a request whose encoding has no entry here is refused up front, exactly as before, and
 * adding a quantizer is one line.
 *
 * Every quantizer produces canonical **row-major** blocks for a tensor whose blocks run along its
 * last dimension; feed-order relayout, if the kernel wants it, is a separate step on the result.
 */
public object Requantizers {

    /** `true` when a dense FP32 tensor can be quantized to [encoding] at load. */
    public fun supports(encoding: TensorEncoding): Boolean = when (encoding) {
        TensorEncoding.Q8_0, TensorEncoding.Q4_0, TensorEncoding.BITNET_PLANES -> true
        else -> false
    }

    /** The encodings [supports] answers `true` for, for messages and docs. */
    public val supported: List<TensorEncoding> = listOf(TensorEncoding.Q8_0, TensorEncoding.Q4_0, TensorEncoding.BITNET_PLANES)

    /**
     * Quantize dense [values] with logical [shape] into packed tensor data of [encoding].
     *
     * @throws IllegalArgumentException for an encoding without a quantizer, or a shape the format
     *   cannot block (last dimension not a multiple of the block size; planes need rank 2).
     */
    public fun quantize(encoding: TensorEncoding, values: FloatArray, shape: Shape): TensorData<*, *> = when (encoding) {
        TensorEncoding.Q8_0 -> Q8_0Quantizer.quantize(values, shape)
        TensorEncoding.Q4_0 -> {
            require(shape.rank == 0 || shape[shape.rank - 1] % 32 == 0) {
                "Q4_0 blocks run along the last dimension, which must be a multiple of 32; got $shape"
            }
            Q4_0Quantizer.quantize(values, shape)
        }
        TensorEncoding.BITNET_PLANES -> {
            require(shape.rank == 2) { "BITNET_PLANES needs a rank-2 weight, got $shape" }
            BitNetPlanesTensorData.fromFloats(shape, values)
        }
        else -> throw IllegalArgumentException(
            "no quantizer for ${encoding.name}; quantize-at-load supports ${supported.joinToString { it.name }}",
        )
    }
}
