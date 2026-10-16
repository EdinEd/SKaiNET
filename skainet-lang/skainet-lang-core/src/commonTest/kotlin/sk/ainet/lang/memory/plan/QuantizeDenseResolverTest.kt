package sk.ainet.lang.memory.plan

import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertEquals

/** SKEEP-007 §T2: a profile may ask for dense sources to be quantized at load, when a kernel can feed the result. */
class QuantizeDenseResolverTest {
    private fun capableOf(vararg supported: TensorEncoding): KernelCapabilities =
        object : KernelCapabilities {
            override fun canFeedMatmul(encoding: TensorEncoding?): Boolean = encoding == null || encoding in supported
        }

    private val browser = PlannerProfile.DESKTOP.copy(name = "browser", quantizeDenseTo = TensorEncoding.Q8_0)

    @Test
    fun `a dense source is requantized when the profile asks and a kernel can feed the encoding`() {
        for (stored in listOf<TensorEncoding?>(null, TensorEncoding.Dense(4))) {
            val form = WeightFormResolver.resolve(stored, browser, capableOf(TensorEncoding.Q8_0), canProduceKernelFeedOrder = true)
            assertEquals(EncodingRequest.RequantizeTo(TensorEncoding.Q8_0), form.encoding, "stored=$stored")
            assertEquals(WeightShapeOrientation.OUT_IN, form.shape, "per-row scales pin the orientation")
            assertEquals(WeightByteOrder.KERNEL_FEED, form.order, "the packed kernels want feed order")
            assertEquals(WeightResidency.HEAP, form.residency)
        }
    }

    @Test
    fun `without a kernel for the encoding the dense bytes are kept`() {
        val form = WeightFormResolver.resolve(null, browser, capableOf(TensorEncoding.Q4_K), canProduceKernelFeedOrder = true)
        assertEquals(EncodingRequest.KeepAsStored, form.encoding)
        assertEquals(WeightByteOrder.AS_STORED, form.order)
    }

    @Test
    fun `the default profile never requantizes and a packed source is untouched by the option`() {
        val dense = WeightFormResolver.resolve(null, PlannerProfile.DESKTOP, capableOf(TensorEncoding.Q8_0), canProduceKernelFeedOrder = true)
        assertEquals(EncodingRequest.KeepAsStored, dense.encoding)
        val packed = WeightFormResolver.resolve(TensorEncoding.Q4_K, browser, capableOf(TensorEncoding.Q4_K), canProduceKernelFeedOrder = true)
        assertEquals(EncodingRequest.KeepAsStored, packed.encoding, "quantizeDenseTo applies to dense sources only")
    }
}
