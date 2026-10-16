package sk.ainet.exec.wasm

import sk.ainet.backend.api.kernel.KernelKey
import sk.ainet.backend.api.kernel.LayoutClass
import sk.ainet.backend.api.kernel.OperandKey
import sk.ainet.backend.api.kernel.ReferenceMatmulKernel
import sk.ainet.backend.api.kernel.ViewKernel
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * `x · Wᵀ` over a canonical row-major packed weight in the SIMD128 sidecar (SKEEP-007 §T1): the
 * browser's `FfmRowMajorMatmulKernel`. Same key, same contract — dense contiguous FP32 activation ×
 * `Format(FP32, encoding)` weight in `BLOCKED_ROW_MAJOR` order, heap storage on both sides.
 *
 * The weight is uploaded into sidecar memory on first use ([WasmKernels.weightHandle]) and read
 * from there on every later call; the activation rows cross per call and the kernel runs once per
 * row, exactly as the FFM kernel invokes its native entry.
 */
public class WasmRowMajorMatmulKernel internal constructor(
    private val encoding: TensorEncoding,
    private val fnName: String,
) : ViewKernel {
    override val key: KernelKey = keyFor(encoding)
    override val name: String = "wasm-${WasmKernels.mode}-${encoding.name.lowercase()}-rm"

    override fun run(inputs: List<TensorView>, out: TensorView): Unit = run(inputs, out, NoopTraceSink)

    override fun run(inputs: List<TensorView>, out: TensorView, sink: TraceSink) {
        require(inputs.size == 2) { "matmul takes two operands" }
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions disagree: [$rows, $k] × [$n, ${w.shape[1]}]" }
        if (!WasmKernels.isReady) return fallback(inputs, out, sink, "sidecar not initialized")
        check(w.layout.blockOrder == BlockOrder.ROW_MAJOR) { "$name reads canonical row-major weights; this one is ${w.layout.blockOrder}" }
        val aHeap = a.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "activation storage ${a.storage::class.simpleName}")
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "output storage ${out.storage::class.simpleName}")
        val wHeap = w.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "weight storage ${w.storage::class.simpleName}")
        val activation = aHeap.floats ?: return fallback(inputs, out, sink, "activation is not a FloatArray")
        val output = oHeap.floats ?: return fallback(inputs, out, sink, "output is not a FloatArray")
        val weightBytes = wHeap.bytes ?: return fallback(inputs, out, sink, "weight is not a ByteArray")
        if (!a.isContiguous) return fallback(inputs, out, sink, "strided activation")

        val weightOffset = wHeap.arrayOffset + (w.layout.offsetElements * w.layout.elementBytes).toInt()
        val weightLength = (wHeap.sizeBytes - (w.layout.offsetElements * w.layout.elementBytes)).toInt()
        val weightPtr = WasmKernels.weightHandle(weightBytes, weightOffset, weightLength)

        val inPtr = WasmKernels.alloc(rows * k * 4)
        val outPtr = WasmKernels.alloc(rows * n * 4)
        try {
            WasmKernels.writeFloats(inPtr, activation, aHeap.arrayOffset + a.layout.offsetElements.toInt(), rows * k)
            for (r in 0 until rows) {
                WasmKernels.packedMatmul(fnName, inPtr + r * k * 4, weightPtr, k, n, outPtr + r * n * 4)
            }
            WasmKernels.readFloats(outPtr, output, oHeap.arrayOffset + out.layout.offsetElements.toInt(), rows * n)
        } finally {
            WasmKernels.free(outPtr)
            WasmKernels.free(inPtr)
        }
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView, sink: TraceSink, reason: String) {
        if (sink.isEnabled) {
            sink.emit(
                TraceEvent.KernelRun(op = key.op, kernel = "reference-fallback from $name: $reason", inputs = inputs.map { it.id }, output = out.id),
            )
        }
        ReferenceMatmulKernel(key).run(inputs, out)
    }

    public companion object {
        /** The key this kernel serves: dense contiguous FP32 activation × row-major packed weight. */
        public fun keyFor(encoding: TensorEncoding): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, encoding), LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )
    }
}
