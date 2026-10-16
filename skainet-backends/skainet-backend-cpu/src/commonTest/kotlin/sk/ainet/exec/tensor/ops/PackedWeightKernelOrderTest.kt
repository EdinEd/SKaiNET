package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.FP32
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * SKEEP-007 §T0.1: a canonical (row-major) packed weight must never be served by the decoding
 * reference kernel when a scalar packed kernel for its format exists.
 *
 * Before the dispatch order change, every non-JVM target sent the canonical GGUF weight through
 * `KernelDispatch` where only feed-order scalar kernels were registered, so the per-element
 * reference decode ran on every call. Now the weight is relaid once into feed order and the scalar
 * kernel runs; the reference kernel is recorded in the trace if it ever runs, and this test reads it.
 */
class PackedWeightKernelOrderTest {
    private val ctx = DirectCpuExecutionContext()
    private val sink = RecordingTraceSink()
    private var previousSink = KernelDispatch.defaultSink

    @BeforeTest
    fun attachSink() {
        previousSink = KernelDispatch.defaultSink
        KernelDispatch.defaultSink = sink
    }

    @AfterTest
    fun detachSink() {
        KernelDispatch.defaultSink = previousSink
    }

    private val outDim = 6
    private val inDim = 128 // four Q8_0 blocks per row

    @Suppress("UNCHECKED_CAST")
    private fun canonicalWeight(): Tensor<FP32, Float> {
        val blocks = outDim * (inDim / 32)
        val bytes = ByteArray(blocks * 34)
        var seed = 5
        for (b in 0 until blocks) {
            val base = b * 34
            bytes[base] = 0x00; bytes[base + 1] = 0x3C // fp16 scale 1.0
            for (i in 0 until 32) {
                seed = seed * 1103515245 + 12345
                bytes[base + 2 + i] = ((seed ushr 16) % 23 - 11).toByte()
            }
        }
        return ctx.fromData(Q8_0BlockTensorData(Shape(outDim, inDim), bytes) as TensorData<FP32, Float>, FP32::class)
    }

    @Test
    fun `canonical Q8_0 weight is served without the reference kernel and matches the relayout path`() {
        val w = canonicalWeight()
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3, inDim), FP32::class, FloatArray(3 * inDim) { ((it % 11) - 5) * 0.25f })

        val viaPrimitive = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
        // Second call exercises the per-weight cache.
        val again = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
        val viaRelayout = ctx.ops.matmul(x, ctx.ops.relayoutPackedWeightForKernels(w)).data.copyToFloatArray()

        assertContentEquals(viaRelayout, viaPrimitive)
        assertContentEquals(viaPrimitive, again)
        val referenceRuns = sink.eventsOf<TraceEvent.KernelRun>().filter { it.kernel.contains("reference", ignoreCase = true) }
        assertTrue(referenceRuns.isEmpty(), "reference kernel ran: $referenceRuns")
    }
}
