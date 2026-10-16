package sk.ainet.exec.wasm

import kotlinx.coroutines.test.runTest
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * SKEEP-007 §T1: the SIMD128 sidecar agrees with the common scalar lane on every path it takes
 * over — row-major packed matmuls (Q8_0, Q4_0, Q4_K via the pack), dense `x · Wᵀ`, and the dense
 * GEMM — and the wiring makes a freshly built context use [WasmCpuOps].
 *
 * Runs on Node (`wasmJsNodeTest`): the sidecar is loaded from the module's own resources. Lane
 * sums differ from the sequential scalar accumulation in the last bits, so comparisons are
 * relative-tolerance, as the JVM Panama and native parity tests are.
 */
class WasmSimdParityTest {

    private fun pseudo(n: Int, seed: Int, scale: Float = 1f): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 1000f * scale
        }
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String, rel: Float = 1e-5f) {
        assertEquals(expected.size, actual.size, "$what size")
        var maxAbs = 0f
        for (v in expected) maxAbs = maxOf(maxAbs, abs(v))
        val tol = maxAbs * rel + 1e-6f
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${actual[i]} vs ${expected[i]} (tol $tol)")
        }
    }

    @AfterTest
    fun tearDown() {
        WasmSimd.uninstall()
    }

    @Suppress("UNCHECKED_CAST")
    private fun q8(ctx: DirectCpuExecutionContext, n: Int, k: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromData(Q8_0Quantizer.quantize(pseudo(n * k, seed, 2f), Shape(n, k)) as TensorData<FP32, Float>, FP32::class)

    @Suppress("UNCHECKED_CAST")
    private fun q4(ctx: DirectCpuExecutionContext, n: Int, k: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromData(Q4_0Quantizer.quantize(pseudo(n * k, seed, 2f), Shape(n, k)) as TensorData<FP32, Float>, FP32::class)

    @Test
    fun `install swaps the ops and the sidecar reports a SIMD mode`() = runTest {
        val before = DirectCpuExecutionContext()
        assertIs<DefaultCpuOps>(before.ops)
        WasmSimd.install()
        assertTrue(WasmSimd.isInstalled)
        assertTrue(WasmKernels.mode.startsWith("simd128-"), WasmKernels.mode)
        val after = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(after.ops)
    }

    @Test
    fun `row-major Q8_0 and Q4_0 weights match the scalar lane and run on the pack`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 5; val n = 70; val k = 96
        val x = pseudo(m * k, 3)
        val wq8 = q8(scalar, n, k, 7)
        val wq4 = q4(scalar, n, k, 9)
        val xs = scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val ref8 = scalar.ops.matmulWeightTransposed(xs, wq8).data.copyToFloatArray()
        val ref4 = scalar.ops.matmulWeightTransposed(xs, wq4).data.copyToFloatArray()

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        val xf = fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        assertIs<Q8_0BlockTensorData>(wq8.data)
        val got8 = fast.ops.matmulWeightTransposed(xf, wq8).data.copyToFloatArray()
        val got4 = fast.ops.matmulWeightTransposed(xf, wq4).data.copyToFloatArray()
        assertClose(ref8, got8, "q8_0")
        assertClose(ref4, got4, "q4_0")
        // Second call hits the resident weight; same answer, bytes stay resident.
        assertClose(ref8, fast.ops.matmulWeightTransposed(xf, wq8).data.copyToFloatArray(), "q8_0 again")
        assertTrue(WasmKernels.residentWeightBytes() > 0)
        val names = KernelDispatch.kernels().map { it.name }
        assertTrue(names.any { it.startsWith("wasm-simd128") }, "pack registered: $names")
    }

    @Test
    fun `dense weight-transposed and dense GEMM match the scalar lane`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 6; val n = 37; val k = 45
        val x = pseudo(m * k, 11)
        val w = pseudo(n * k, 13)
        val b = pseudo(k * n, 17)
        val xs = scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val ws = scalar.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bs = scalar.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b)
        val refWt = scalar.ops.matmulWeightTransposed(xs, ws).data.copyToFloatArray()
        val refMm = scalar.ops.matmul(xs, bs).data.copyToFloatArray()

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        val xf = fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val wf = fast.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bf = fast.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b)
        assertClose(refWt, fast.ops.matmulWeightTransposed(xf, wf).data.copyToFloatArray(), "x·Wᵀ")
        val rank1 = fast.fromFloatArray<FP32, Float>(Shape(k), FP32::class, x.copyOfRange(0, k))
        assertClose(refWt.copyOfRange(0, n), fast.ops.matmulWeightTransposed(rank1, wf).data.copyToFloatArray(), "rank-1 x·Wᵀ")
        // The dense GEMM goes through KernelDispatch → Fp32ViewMatmulKernel(WasmFp32MatmulKernel) for
        // the registry path; the common [k, n] loop still serves the heap fast path, so exercise the
        // provider kernel directly as well.
        assertClose(refMm, fast.ops.matmul(xf, bf).data.copyToFloatArray(), "matmul")
        val direct = FloatArray(m * n)
        WasmFp32MatmulKernel.matmul(x, 0, k, b, 0, n, direct, 0, n, m, n, k)
        assertClose(refMm, direct, "provider gemm")
    }
}
