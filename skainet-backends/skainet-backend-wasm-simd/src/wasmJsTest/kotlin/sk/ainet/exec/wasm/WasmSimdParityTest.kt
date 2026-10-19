package sk.ainet.exec.wasm

import kotlinx.coroutines.test.runTest
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelStrictness
import sk.ainet.backend.api.kernel.NoSuchKernelException
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarFp16MatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_1MatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ6_KMatmulKernel
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
private typealias PackedMatmul = (FloatArray, ByteArray, Int, Int, FloatArray) -> Unit

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
        assertEquals("none", WasmKernels.features().mode)
        WasmSimd.install()
        assertTrue(WasmSimd.isInstalled)
        assertTrue(WasmKernels.mode.startsWith("simd128-"), WasmKernels.mode)
        val after = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(after.ops)
        val f = WasmSimd.features()
        assertEquals(WasmKernels.mode, f.mode)
        assertTrue(f.simd128)
        assertEquals(f.mode == "simd128-mt", f.threads)
        assertTrue(f.hardwareConcurrency >= 1, "cores: ${f.hardwareConcurrency}")
    }

    @Test
    fun `BF16 and FP16 weights run on the sidecar and match the scalar kernels`() = runTest {
        WasmSimd.install()
        val provider = checkNotNull(KernelRegistry.bestAvailable())
        assertEquals("wasm-simd128", provider.name)
        val m = 7; val n = 33; val k = 40
        val a = pseudo(m * k, 21)
        val wf = pseudo(k * n, 23)
        val bf16 = ByteArray(k * n * 2)
        val fp16 = ByteArray(k * n * 2)
        for (i in 0 until k * n) {
            val b = Bf16Codec.encode(wf[i]); bf16[i * 2] = (b and 0xFF).toByte(); bf16[i * 2 + 1] = ((b ushr 8) and 0xFF).toByte()
            val h = Fp16Codec.encode(wf[i]); fp16[i * 2] = (h and 0xFF).toByte(); fp16[i * 2 + 1] = ((h ushr 8) and 0xFF).toByte()
        }
        val refB = FloatArray(m * n); val refH = FloatArray(m * n)
        ScalarBf16MatmulKernel.matmul(a, 0, k, bf16, 0, n * 2, refB, 0, n, m, n, k)
        ScalarFp16MatmulKernel.matmul(a, 0, k, fp16, 0, n * 2, refH, 0, n, m, n, k)
        val gotB = FloatArray(m * n); val gotH = FloatArray(m * n)
        checkNotNull(provider.matmulBf16()).matmul(a, 0, k, bf16, 0, n * 2, gotB, 0, n, m, n, k)
        checkNotNull(provider.matmulFp16()).matmul(a, 0, k, fp16, 0, n * 2, gotH, 0, n, m, n, k)
        assertClose(refB, gotB, "bf16")
        assertClose(refH, gotH, "fp16")
        // A contiguous weight is resident after the first call; a strided window is staged per call.
        assertTrue(WasmKernels.residentWeightBytes() >= 2L * k * n * 2)
        val sub = FloatArray(m * 8)
        val refSub = FloatArray(m * 8)
        ScalarBf16MatmulKernel.matmul(a, 0, k, bf16, 4, n * 2, refSub, 0, 8, m, 8, k)
        checkNotNull(provider.matmulBf16()).matmul(a, 0, k, bf16, 4, n * 2, sub, 0, 8, m, 8, k)
        assertClose(refSub, sub, "bf16 window")
    }

    @Test
    fun `strict mode is honoured on wasm`() = runTest {
        KernelStrictness.overrideEnabled = true
        try {
            assertTrue(KernelStrictness.isEnabled())
            val ctx = DirectCpuExecutionContext()
            // Packed weight, no pack installed and the reference fallback forbidden: fail fast.
            val w = q8(ctx, 64, 96, 5)
            val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 96), FP32::class, pseudo(2 * 96, 1))
            KernelDispatch.clearForTesting()
            // A canonical row-major Q8_0 view has no exact scalar kernel (those are keyed input-block-major)
            // and prepack is off by default here, so this is the reference-fallback site.
            val out = sk.ainet.lang.memory.TensorView.dense(sk.ainet.lang.memory.Storage.Heap.floats(2 * 64), Shape(2, 64), sk.ainet.lang.types.FP32)
            assertFailsWith<NoSuchKernelException> {
                KernelDispatch.matmul(x.data.view!!, w.data.view!!, out, sk.ainet.lang.memory.Scope.Ambient)
            }
        } finally {
            KernelStrictness.overrideEnabled = null
        }
        assertEquals(false, KernelStrictness.isEnabled())
    }

    /**
     * Random packed bytes with the FP16 scale slots forced to 1.0 (0x3C00) so magnitudes stay
     * finite — the fixture the JVM native parity tests use. [exact] formats accumulate in FP32 on
     * both sides and agree to lane-sum tolerance; the K-quant int8 kernels (Q4_K, Q6_K) quantize the
     * activation to Q8 first, so they are gated on an aggregate relative RMS error like the JVM tests.
     */
    private fun packedParity(
        what: String, blockSize: Int, bytesPerBlock: Int, scaleSlots: IntArray, inputDim: Int, outputDim: Int,
        exact: Boolean, scalar: PackedMatmul, wasm: PackedMatmul,
    ) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val rng = Random(what.hashCode())
        val packed = ByteArray(numBlocks * bytesPerBlock).also { rng.nextBytes(it) }
        for (block in 0 until numBlocks) {
            for (slot in scaleSlots) {
                packed[block * bytesPerBlock + slot] = 0x00
                packed[block * bytesPerBlock + slot + 1] = 0x3C
            }
        }
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val ref = FloatArray(outputDim)
        val got = FloatArray(outputDim)
        scalar(input, packed, inputDim, outputDim, ref)
        wasm(input, packed, inputDim, outputDim, got)
        if (exact) {
            assertClose(ref, got, what, rel = 1e-4f)
        } else {
            var err = 0.0; var sig = 0.0
            for (o in 0 until outputDim) { err += (ref[o] - got[o]).toDouble().let { it * it }; sig += ref[o].toDouble().let { it * it } }
            val relRms = sqrt(err / outputDim) / (sqrt(sig / outputDim) + 1e-9)
            assertTrue(relRms < 0.03, "$what: relRms=$relRms over $outputDim rows")
        }
    }

    @Test
    fun `K-quant and Q5 weights run on the sidecar and match the scalar kernels`() = runTest {
        WasmSimd.install()
        val p = checkNotNull(KernelRegistry.bestAvailable())
        assertEquals("wasm-simd128", p.name)
        // 37 rows: an odd count, so a threaded row split (none below 512) is not what is under test;
        // two or three blocks per row cover the block-outer accumulation and the per-row carry.
        packedParity("q4_k", 256, 144, intArrayOf(0, 2), 512, 37, exact = false,
            { i, w, id, od, o -> ScalarQ4_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ4K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q6_k", 256, 210, intArrayOf(208), 512, 37, exact = false,
            { i, w, id, od, o -> ScalarQ6_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ6K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_k", 256, 176, intArrayOf(0, 2), 512, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_0", 32, 22, intArrayOf(0), 96, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_0MatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5_0()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_1", 32, 24, intArrayOf(0, 2), 96, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_1MatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5_1()).matmul(i, 0, w, 0, id, od, o, 0) })
    }

    @Test
    fun `dense GEMM through the ops layer runs on the sidecar for every tile shape`() = runTest {
        val scalar = DirectCpuExecutionContext()
        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(fast.ops)
        // Full 4×8 tiles, a 1-row remainder, a 6-column remainder and a k past one 256-block.
        for ((m, n, k) in listOf(Triple(9, 70, 300), Triple(1, 8, 5), Triple(4, 3, 1), Triple(5, 16, 256))) {
            val x = pseudo(m * k, m + n)
            val b = pseudo(k * n, k)
            val ref = scalar.ops.matmul(
                scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x),
                scalar.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b),
            ).data.copyToFloatArray()
            val got = fast.ops.matmul(
                fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x),
                fast.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b),
            ).data.copyToFloatArray()
            assertClose(ref, got, "gemm $m×$k·$k×$n")
        }
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
        // The dense GEMM runs the sidecar through the WasmCpuOps override (heap operands) and through
        // KernelDispatch → Fp32ViewMatmulKernel(WasmFp32MatmulKernel) for view operands; exercise the
        // provider kernel directly as well.
        assertClose(refMm, fast.ops.matmul(xf, bf).data.copyToFloatArray(), "matmul")
        val direct = FloatArray(m * n)
        WasmFp32MatmulKernel.matmul(x, 0, k, b, 0, n, direct, 0, n, m, n, k)
        assertClose(refMm, direct, "provider gemm")
    }
}
