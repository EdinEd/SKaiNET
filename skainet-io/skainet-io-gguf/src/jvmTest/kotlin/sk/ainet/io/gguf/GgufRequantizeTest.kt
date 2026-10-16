package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * SKEEP-007 §T2: the GGUF loader's `RequantizeTo` now reaches every encoding in `Requantizers`, not
 * only `BITNET_PLANES` (#1150). A dense F32 weight in a GGUF becomes Q8_0 at load, in canonical or
 * kernel feed order, within one quantization step of its dense values.
 */
class GgufRequantizeTest {
    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("dense2d", GGMLQuantizationType.F32, elements = 2048).copy(dims = listOf(64L, 32L)),
        SyntheticGguf.tensor("vector", GGMLQuantizationType.F32, elements = 64),
    )

    private fun loadVia(f: File, form: WeightForm?, formFor: ((String) -> WeightForm?)? = null): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
                weightFormFor = formFor,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    private val q8 = WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0), shape = WeightShapeOrientation.OUT_IN)

    @Test
    fun `a dense F32 matrix is requantized to Q8_0 within one code step`() {
        val f = file()
        try {
            val dense = loadVia(f, null).getValue("dense2d").data.copyToFloatArray()
            val packed = assertIs<Q8_0BlockTensorData>(
                loadVia(f, null, formFor = { if (it == "dense2d") q8 else null }).getValue("dense2d").data,
            )
            assertEquals(BlockOrder.ROW_MAJOR, packed.blockOrder)
            val back = packed.toFloatArray()
            assertEquals(dense.size, back.size)
            val amax = dense.maxOf { abs(it) }
            for (i in dense.indices) {
                assertTrue(abs(dense[i] - back[i]) <= amax / 127f + 1e-4f, "element $i: ${back[i]} vs ${dense[i]}")
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `KERNEL_FEED on a requantized weight yields input-block-major blocks with the same values`() {
        val f = file()
        try {
            val fed = assertIs<Q8_0BlockTensorData>(
                loadVia(f, null, formFor = { if (it == "dense2d") q8.copy(order = WeightByteOrder.KERNEL_FEED) else null })
                    .getValue("dense2d").data,
            )
            assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, fed.blockOrder)
            val canonical = assertIs<Q8_0BlockTensorData>(
                loadVia(f, null, formFor = { if (it == "dense2d") q8 else null }).getValue("dense2d").data,
            )
            assertEquals(canonical.toFloatArray().toList(), fed.toFloatArray().toList())
        } finally {
            f.delete()
        }
    }
}
