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
 * (`x · Wᵀ` and `A · B`) routed through the SIMD128 sidecar. Packed weights need no override — they reach the pack through
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

    /**
     * Dense `A(m,k) · B(k,n)` on heap FP32 operands runs the sidecar GEMM. The common implementation
     * serves this shape with its own `[k, n]` loop before it consults the registry, so without this
     * override the SIMD128 kernel was reachable only through `KernelDispatch` (batched or view
     * operands) and a plain 2-D matmul stayed at scalar speed. Both operands are staged per call:
     * `B` here is not necessarily a parameter (attention scores, projections of activations), so
     * nothing is kept resident. A packed weight is not `FloatArrayTensorData` and keeps its path.
     */
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && a.rank == 2 && b.rank == 2 && a.dtype == FP32::class && b.dtype == FP32::class) {
            val ad = a.data as? FloatArrayTensorData<*>
            val bd = b.data as? FloatArrayTensorData<*>
            if (ad != null && bd != null && untransposedWeight(b) == null) {
                val m = a.shape[0]
                val k = a.shape[1]
                val n = b.shape[1]
                require(k == b.shape[0]) { "Matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}" }
                val out = FloatArray(m * n)
                if (m > 0 && n > 0) WasmFp32MatmulKernel.matmul(ad.buffer, 0, k, bd.buffer, 0, n, out, 0, n, m, n, k)
                val outData = dataFactory.adoptFloatArray<T, Float>(Shape(m, n), a.dtype, out) as TensorData<T, V>
                return newTensor(outData, a.dtype, a, b)
            }
        }
        return super.matmul(a, b)
    }

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
