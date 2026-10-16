package sk.ainet.exec.wasm

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.tensor.ops.DefaultCpuOpsBase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * The browser tier's `TensorOps` (SKEEP-007 §T1): [DefaultCpuOpsBase] with the dense FP32 matmuls
 * routed through the SIMD128 sidecar. Packed weights need no override — they reach the pack through
 * `KernelDispatch` like on every other tier. Everything else is the common scalar implementation,
 * and so is everything here while the sidecar is not ready.
 */
public class WasmCpuOps(
    dataFactory: TensorDataFactory,
    schedule: Schedule,
) : DefaultCpuOpsBase(dataFactory, schedule) {

    override fun withSchedule(schedule: Schedule): TensorOps =
        if (schedule === this.schedule) this else WasmCpuOps(dataFactory, schedule)

    /** The sidecar computes `x · Wᵀ` from the weight's rows itself; no Kotlin-side row-dot and no transposed copy. */
    override fun preferDirectWeightTransposed(): Boolean = false

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && weight.shape.rank == 2 && x.rank >= 1 && x.dtype == FP32::class && weight.dtype == FP32::class) {
            val w = weight.data as? FloatArrayTensorData<*>
            val a = x.data as? FloatArrayTensorData<*>
            if (w != null && a != null) {
                val n = weight.shape[0]
                val k = weight.shape[1]
                if (x.shape.dimensions.last() == k) {
                    val m = if (k == 0) 0 else x.shape.volume / k
                    val out = FloatArray(m * n)
                    if (m > 0 && n > 0) {
                        val inPtr = WasmKernels.alloc(m * k * 4)
                        val outPtr = WasmKernels.alloc(m * n * 4)
                        try {
                            WasmKernels.writeFloats(inPtr, a.buffer, 0, m * k)
                            // Resident like a packed weight: a parameter is immutable once loaded, and
                            // re-uploading it per call made single-token decode run at copy speed.
                            val wPtr = WasmKernels.denseWeightHandle(w.buffer)
                            WasmKernels.fp32MatmulWt(inPtr, wPtr, outPtr, m, n, k)
                            WasmKernels.readFloats(outPtr, out, 0, m * n)
                        } finally {
                            WasmKernels.free(outPtr)
                            WasmKernels.free(inPtr)
                        }
                    }
                    val outShape = if (x.rank == 1) Shape(n) else Shape(*(x.shape.dimensions.copyOf(x.rank - 1) + n))
                    val outData = dataFactory.adoptFloatArray<T, Float>(outShape, x.dtype, out) as TensorData<T, V>
                    return newTensor(outData, x.dtype, x, weight)
                }
            }
        }
        return super.matmulWeightTransposed(x, weight)
    }
}
