package sk.ainet.exec.wasm

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.tensor.ops.DefaultCpuOpsBase
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * The browser tier's `TensorOps` (SKEEP-007 §T1 and the resident-activation step): [DefaultCpuOpsBase]
 * with the dense matmuls, the elementwise arithmetic, the activations and the reductions routed
 * through the SIMD128 sidecar, and their outputs kept **resident** in the sidecar's linear memory
 * ([WasmLinearTensorData]) so a chain of them never copies an activation back to the Kotlin heap
 * between kernels. Packed weights need no override — they reach the pack through `KernelDispatch`
 * like on every other tier. Everything else is the common scalar implementation, which reads a
 * resident tensor through its heap mirror (one bulk download, the first time), and so is
 * everything here while the sidecar is not ready.
 *
 * What goes to the sidecar, and when:
 *
 * - `matmul`, `matmulWeightTransposed` (dense FP32): always; the output is resident.
 * - `add`/`subtract`/`multiply`/`divide`: when at least one operand is already resident. Two heap
 *   operands stay on the common path — uploading both costs more than the scalar loop saves, and
 *   a chain starts resident at its first matmul anyway.
 * - `sigmoid`/`silu`/`gelu`/`erf`/`tanh`: always — the transcendental loop is several times the
 *   upload. `relu`, `sum`, `mean`: when the operand is resident.
 *
 * An output is resident only while [WasmSimd.residentBudgetBytes] allows; above it the kernel still
 * runs in the sidecar and the result is read back to the heap. Scoped contexts (`ForwardScope`)
 * allocate the resident output from the scope, so `reset()` frees the sidecar bytes; ambient outputs
 * are freed when their tensor is collected (see [WasmLinearStorage]).
 */
public class WasmCpuOps(
    dataFactory: TensorDataFactory,
    schedule: Schedule,
) : DefaultCpuOpsBase(dataFactory, schedule) {

    override fun withSchedule(schedule: Schedule): TensorOps =
        if (schedule === this.schedule) this else WasmCpuOps(dataFactory, schedule)

    /** The sidecar computes `x · Wᵀ` from the weight's rows itself; no Kotlin-side row-dot and no transposed copy. */
    override fun preferDirectWeightTransposed(): Boolean = false

    // ── operands and outputs ──────────────────────────────────────────────────────────────────────

    /** A sidecar address for an operand: a resident tensor's own, or a staged copy freed after the call. */
    private class Operand(val ptr: Int, val temp: Boolean)

    private fun isResident(t: Tensor<*, *>): Boolean = (t.data as? WasmLinearTensorData<*>)?.isResident == true

    /** Dense FP32 the sidecar can take: resident, a heap array, or a slab window. */
    private fun isSidecarFloat(t: Tensor<*, *>): Boolean =
        t.dtype == FP32::class && (t.data is WasmLinearTensorData<*> || t.data is FloatArrayTensorData<*> || t.data is StorageFloatTensorData<*>)

    private fun operand(t: Tensor<*, *>): Operand? = when (val d = t.data) {
        is WasmLinearTensorData<*> -> if (d.isResident) Operand(d.pointer(), false) else null
        is FloatArrayTensorData<*> -> stage(d.buffer, 0, t.shape.volume)
        is StorageFloatTensorData<*> -> {
            val s = d.storage
            s.checkAlive()
            stage(s.floats!!, s.arrayOffset, t.shape.volume)
        }
        else -> null
    }

    private fun stage(src: FloatArray, offset: Int, count: Int): Operand {
        val ptr = WasmKernels.alloc(maxOf(count, 1) * 4)
        if (count > 0) WasmKernels.writeFloats(ptr, src, offset, count)
        return Operand(ptr, true)
    }

    private fun release(o: Operand?) {
        if (o != null && o.temp) WasmKernels.free(o.ptr)
    }

    /** A resident output allocation from the active scope (ambient: GC-managed), or `null` when the scope cannot place it in the sidecar. */
    private fun residentStorage(bytes: Long): WasmLinearStorage? {
        val scope = dispatchScope()
        return if (scope === Scope.Ambient) WasmLinearStorage.allocate(bytes, ScopeKind.AMBIENT, zero = false)
        else scope.allocate(bytes, MemoryDomain.HOST_OFFHEAP) as? WasmLinearStorage
    }

    /**
     * Run [fill] into an output buffer of [shape]: resident when the budget allows, otherwise a
     * temporary that is read back to the heap and adopted through the factory.
     */
    @Suppress("UNCHECKED_CAST")
    private inline fun <T : DType, V> output(
        shape: Shape,
        dtype: KClass<T>,
        inputs: Array<out Tensor<T, V>>,
        fill: (outPtr: Int) -> Unit,
    ): Tensor<T, V> {
        val n = shape.volume
        val bytes = n.toLong() * 4
        if (WasmSimd.residentBudgetBytes - WasmKernels.residentActivationBytes() >= bytes) {
            residentStorage(bytes)?.let { storage ->
                fill(storage.pointer())
                return newTensor(WasmLinearTensorData<T>(shape, storage) as TensorData<T, V>, dtype, *inputs)
            }
        }
        val tmp = WasmKernels.alloc(maxOf(n, 1) * 4)
        try {
            fill(tmp)
            val out = FloatArray(n)
            if (n > 0) WasmKernels.readFloats(tmp, out, 0, n)
            return floatResult(shape, dtype, out, *inputs)
        } finally {
            WasmKernels.free(tmp)
        }
    }

    /**
     * A resident copy of a dense FP32 [tensor] — the explicit "upload once" for an input that many
     * sidecar ops will read (a bench's operands, a prompt's embeddings), so none of them stages it
     * per call. The tensor itself if it is already resident; the common data unchanged when the
     * sidecar is not ready or the budget is spent.
     */
    public fun <T : DType, V> toResident(tensor: Tensor<T, V>): Tensor<T, V> {
        if (!WasmKernels.isReady || !isSidecarFloat(tensor) || isResident(tensor)) return tensor
        val n = tensor.shape.volume
        val o = operand(tensor) ?: return tensor
        try {
            val src = o.ptr
            val bytes = n.toLong() * 4
            if (WasmSimd.residentBudgetBytes - WasmKernels.residentActivationBytes() < bytes) return tensor
            val storage = residentStorage(bytes) ?: return tensor
            WasmKernels.copy(src, storage.pointer(), n * 4)
            @Suppress("UNCHECKED_CAST")
            return newTensor(WasmLinearTensorData<T>(tensor.shape, storage) as TensorData<T, V>, tensor.dtype, tensor)
        } finally {
            release(o)
        }
    }

    // ── matmuls ───────────────────────────────────────────────────────────────────────────────────

    /**
     * Dense `A(m,k) · B(k,n)` on FP32 operands runs the sidecar GEMM. The common implementation
     * serves this shape with its own `[k, n]` loop before it consults the registry, so without this
     * override the SIMD128 kernel was reachable only through `KernelDispatch`. Resident operands are
     * read in place; heap operands are staged per call (`B` is not necessarily a parameter). A packed
     * weight is not dense FP32 data and keeps its path.
     */
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && a.rank == 2 && b.rank == 2 && isSidecarFloat(a) && isSidecarFloat(b) && untransposedWeight(b) == null) {
            val m = a.shape[0]
            val k = a.shape[1]
            val n = b.shape[1]
            require(k == b.shape[0]) { "Matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}" }
            if (m > 0 && n > 0) {
                var ao: Operand? = null
                var bo: Operand? = null
                try {
                    ao = operand(a) ?: return super.matmul(a, b)
                    bo = operand(b) ?: return super.matmul(a, b)
                    val aPtr = ao.ptr
                    val bPtr = bo.ptr
                    return output(Shape(m, n), a.dtype, arrayOf(a, b)) { out -> WasmKernels.fp32Matmul(aPtr, k, bPtr, n, out, n, m, n, k) }
                } finally {
                    release(bo)
                    release(ao)
                }
            }
        }
        return super.matmul(a, b)
    }

    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && weight.shape.rank == 2 && x.rank >= 1 && isSidecarFloat(x) && isSidecarFloat(weight)) {
            val n = weight.shape[0]
            val k = weight.shape[1]
            if (x.shape.dimensions.last() == k) {
                val m = if (k == 0) 0 else x.shape.volume / k
                if (m > 0 && n > 0) {
                    var xo: Operand? = null
                    var wo: Operand? = null
                    try {
                        xo = operand(x) ?: return super.matmulWeightTransposed(x, weight)
                        // A heap weight is a parameter: resident after its first call (re-uploading it
                        // per call made single-token decode run at copy speed). Resident and slab
                        // weights are read where they are.
                        wo = when (val wd = weight.data) {
                            is FloatArrayTensorData<*> -> Operand(WasmKernels.denseWeightHandle(wd.buffer), false)
                            else -> operand(weight) ?: return super.matmulWeightTransposed(x, weight)
                        }
                        val xPtr = xo.ptr
                        val wPtr = wo.ptr
                        val outShape = if (x.rank == 1) Shape(n) else Shape(*(x.shape.dimensions.copyOf(x.rank - 1) + n))
                        return output(outShape, x.dtype, arrayOf(x, weight)) { out -> WasmKernels.fp32MatmulWt(xPtr, wPtr, out, m, n, k) }
                    } finally {
                        release(wo)
                        release(xo)
                    }
                }
            }
        }
        return super.matmulWeightTransposed(x, weight)
    }

    // ── elementwise ───────────────────────────────────────────────────────────────────────────────

    /** The operand's index mode against the broadcast output shape, or `null` for a broadcast the kernel does not take. */
    private fun mode(s: Shape, out: Shape): Int? = when {
        s == out -> WasmKernels.F32_FULL
        s.volume == 1 -> WasmKernels.F32_SCALAR
        s.rank >= 1 && out.rank >= 1 && s[s.rank - 1] == out[out.rank - 1] && s.dimensions.dropLast(1).all { it == 1 } -> WasmKernels.F32_BIAS
        else -> null
    }

    private fun <T : DType, V> binary(op: Int, a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!WasmKernels.isReady || a.dtype != b.dtype || !isSidecarFloat(a) || !isSidecarFloat(b)) return null
        if (!isResident(a) && !isResident(b)) return null
        val outShape = try {
            broadcastShapes(a.shape, b.shape)
        } catch (e: IllegalArgumentException) {
            return null
        }
        val n = outShape.volume
        if (n == 0) return null
        val aMode = mode(a.shape, outShape) ?: return null
        val bMode = mode(b.shape, outShape) ?: return null
        val inner = outShape[outShape.rank - 1]
        var ao: Operand? = null
        var bo: Operand? = null
        try {
            ao = operand(a) ?: return null
            bo = operand(b) ?: return null
            val aPtr = ao.ptr
            val bPtr = bo.ptr
            return output(outShape, a.dtype, arrayOf(a, b)) { out -> WasmKernels.f32Binary(op, aPtr, aMode, bPtr, bMode, out, n, inner) }
        } finally {
            release(bo)
            release(ao)
        }
    }

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_ADD, a, b) ?: super.add(a, b)

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_SUB, a, b) ?: super.subtract(a, b)

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_MUL, a, b) ?: super.multiply(a, b)

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_DIV, a, b) ?: super.divide(a, b)

    // ── activations ───────────────────────────────────────────────────────────────────────────────

    private fun <T : DType, V> unary(op: Int, t: Tensor<T, V>, heapToo: Boolean): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(t)) return null
        if (!heapToo && !isResident(t)) return null
        val n = t.shape.volume
        if (n == 0) return null
        val o = operand(t) ?: return null
        try {
            val inPtr = o.ptr
            return output(t.shape, t.dtype, arrayOf(t)) { out -> WasmKernels.f32Unary(op, inPtr, out, n) }
        } finally {
            release(o)
        }
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_RELU, tensor, heapToo = false) ?: super.relu(tensor)

    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_SIGMOID, tensor, heapToo = true) ?: super.sigmoid(tensor)

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_SILU, tensor, heapToo = true) ?: super.silu(tensor)

    override fun <T : DType, V> gelu(tensor: Tensor<T, V>, approximate: Boolean): Tensor<T, V> =
        unary(if (approximate) WasmKernels.F32_GELU_TANH else WasmKernels.F32_GELU_EXACT, tensor, heapToo = true)
            ?: super.gelu(tensor, approximate)

    override fun <T : DType, V> erf(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_ERF, tensor, heapToo = true) ?: super.erf(tensor)

    override fun <T : DType, V> tanh(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_TANH, tensor, heapToo = true) ?: super.tanh(tensor)

    // ── reductions ────────────────────────────────────────────────────────────────────────────────

    private fun <T : DType, V> reduce(t: Tensor<T, V>, dim: Int?, mean: Boolean): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(t) || !isResident(t)) return null
        val rank = t.rank
        val volume = t.shape.volume
        if (volume == 0) return null
        val o = operand(t) ?: return null
        try {
            val inPtr = o.ptr
            if (dim == null) {
                return output(Shape(), t.dtype, arrayOf(t)) { out -> WasmKernels.f32ReduceAll(inPtr, volume, mean, out) }
            }
            val nd = if (dim < 0) dim + rank else dim
            if (nd !in 0 until rank) return null
            val dims = t.shape.dimensions
            var outer = 1
            for (i in 0 until nd) outer *= dims[i]
            val red = dims[nd]
            var inner = 1
            for (i in nd + 1 until rank) inner *= dims[i]
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) { if (i == nd) continue; outDims[oi++] = dims[i] }
            return output(Shape(outDims), t.dtype, arrayOf(t)) { out -> WasmKernels.f32ReduceAxis(inPtr, outer, red, inner, mean, out) }
        } finally {
            release(o)
        }
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> =
        reduce(tensor, dim, mean = false) ?: super.sum(tensor, dim)

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> =
        reduce(tensor, dim, mean = true) ?: super.mean(tensor, dim)
}
