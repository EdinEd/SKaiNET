package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32

/**
 * Multi-row activations through the native (FFM) packed kernels.
 *
 * `DefaultCpuOpsJvm` serves a `[rows, in] × packed [out, in]ᵀ` product by calling the format's SPI
 * kernel once per row with `outputOffset = row * out` into one shared output array. The Q8_0,
 * Q4_0, Q5_0 and Q5_1 bindings staged a zero-filled segment of `outputOffset + out` floats and
 * copied the whole reach back from index 0 — so every call wiped the rows computed before it and a
 * multi-row product came back as zeros except for its last row. A single-row product (decode) was
 * right, which is why the GGUF decode paths never saw it; an encoder's token rows (Laya, BERT) did.
 *
 * Two contracts, with the native provider alone and with the default registry:
 * - `ops.matmulWeightTransposed(x, w)` on a canonical row-major packed weight matches the
 *   block-sequential dequant ground truth for 1, 7 and 50 rows, every format;
 * - a direct SPI call at a non-zero `outputOffset` writes exactly its slice and nothing else.
 */
class NativeQ8CanonicalWeightTransposedReproTest {

    @BeforeTest
    fun nativeAvailable() {
        assertTrue(NativeKernelProvider.isAvailable(), "the native provider must be available for this test")
    }

    @AfterTest
    fun reset() = KernelRegistry.clearForTesting()

    private class Format(val name: String, val blockElems: Int, val bytesPerBlock: Int, val build: (Shape, ByteArray) -> TensorData<FP32, Float>, val block: (Random, ByteArray, Int) -> Unit)

    private fun le16(b: ByteArray, o: Int, h: Int) {
        b[o] = (h and 0xFF).toByte(); b[o + 1] = ((h ushr 8) and 0xFF).toByte()
    }

    @Suppress("UNCHECKED_CAST")
    private val formats: List<Format> = listOf(
        Format("q8_0", 32, 34, { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x2C00 + rng.nextInt(0x80)); for (k in 2 until 34) b[base + k] = rng.nextInt(256).toByte() },
        Format("q4_0", 32, 18, { s, b -> Q4_0BlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x2C00 + rng.nextInt(0x80)); for (k in 2 until 18) b[base + k] = rng.nextInt(256).toByte() },
        Format("q5_0", 32, 22, { s, b -> Q5_0BlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x2C00 + rng.nextInt(0x80)); for (k in 2 until 22) b[base + k] = rng.nextInt(256).toByte() },
        Format("q5_1", 32, 24, { s, b -> Q5_1BlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x2C00 + rng.nextInt(0x80)); le16(b, base + 2, 0x2400 + rng.nextInt(0x80)); for (k in 4 until 24) b[base + k] = rng.nextInt(256).toByte() },
        Format("q4_k", 256, 144, { s, b -> Q4_KBlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x3400 + rng.nextInt(0x80)); le16(b, base + 2, 0x2C00 + rng.nextInt(0x80)); for (k in 4 until 144) b[base + k] = rng.nextInt(256).toByte() },
        Format("q5_k", 256, 176, { s, b -> Q5_KBlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> le16(b, base, 0x3400 + rng.nextInt(0x80)); le16(b, base + 2, 0x2C00 + rng.nextInt(0x80)); for (k in 4 until 176) b[base + k] = rng.nextInt(256).toByte() },
        Format("q6_k", 256, 210, { s, b -> Q6_KBlockTensorData(s, b) as TensorData<FP32, Float> }) { rng, b, base -> for (k in 0 until 208) b[base + k] = rng.nextInt(256).toByte(); le16(b, base + 208, 0x3400 + rng.nextInt(0x80)) },
    )

    private fun canonicalBytes(f: Format, outputDim: Int, inputDim: Int, seed: Int): ByteArray {
        val blocks = inputDim / f.blockElems
        val bytes = ByteArray(outputDim * blocks * f.bytesPerBlock)
        for (o in 0 until outputDim) for (bI in 0 until blocks) f.block(Random(seed * 92821 + o * 977 + bI), bytes, (o * blocks + bI) * f.bytesPerBlock)
        return bytes
    }

    private fun checkWeightTransposed(label: String, f: Format, rows: Int, inputDim: Int, outputDim: Int, seed: Int) {
        val bytes = canonicalBytes(f, outputDim, inputDim, seed)
        val ctx = DirectCpuExecutionContext()
        val wData = f.build(Shape(outputDim, inputDim), bytes)
        val wFlat = (wData as PackedBlockStorage).toFloatArray()
        val w = ctx.fromData(wData, FP32::class)
        val rng = Random(seed + 7)
        val xf = FloatArray(rows * inputDim) { rng.nextFloat() - 0.5f }
        val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, inputDim), FP32::class, xf)
        val truth = FloatArray(rows * outputDim)
        for (r in 0 until rows) for (o in 0 until outputDim) {
            var s = 0f
            for (i in 0 until inputDim) s += xf[r * inputDim + i] * wFlat[o * inputDim + i]
            truth[r * outputDim + o] = s
        }
        val scale = maxOf(truth.maxOf { abs(it) }, 1e-6f)
        repeat(2) { pass -> // the second pass reads the relayout cache
            val y = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
            assertEquals(truth.size, y.size, "$label ${f.name}: output size")
            val worst = y.indices.maxOf { abs(y[it] - truth[it]) }
            val zeroRows = (0 until rows).count { r -> (0 until outputDim).all { y[r * outputDim + it] == 0f } }
            println("[$label ${f.name}] rows=$rows in=$inputDim out=$outputDim pass=$pass zeroRows=$zeroRows worst=${"%.3g".format(worst)} scale=${"%.3g".format(scale)}")
            assertEquals(0, zeroRows, "$label ${f.name} rows=$rows pass=$pass: $zeroRows all-zero output rows")
            assertTrue(worst <= 1e-2f * scale, "$label ${f.name} rows=$rows pass=$pass: worst |diff| $worst exceeds 1% of scale $scale")
        }
    }

    private fun allRows(label: String) {
        for (f in formats) {
            val inputDim = if (f.blockElems == 256) 512 else 768
            checkWeightTransposed(label, f, rows = 1, inputDim = inputDim, outputDim = 96, seed = 1)
            checkWeightTransposed(label, f, rows = 7, inputDim = inputDim, outputDim = 96, seed = 2)
            checkWeightTransposed(label, f, rows = 50, inputDim = inputDim, outputDim = 64, seed = 3)
        }
    }

    @Test
    fun multi_row_products_match_ground_truth_with_the_native_provider_alone() {
        KernelRegistry.clearForTesting()
        KernelRegistry.register(NativeKernelProvider)
        allRows("native-only")
    }

    @Test
    fun multi_row_products_match_ground_truth_with_the_default_registry() {
        KernelRegistry.clearForTesting()
        allRows("default")
    }

    @Test
    fun a_call_at_an_output_offset_writes_only_its_own_slice() {
        val inputDim = 64
        val outputDim = 48
        val rng = Random(5)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val q8 = canonicalBytes(formats[0], outputDim, inputDim, 9)
        val q4 = canonicalBytes(formats[1], outputDim, inputDim, 9)
        val q50 = canonicalBytes(formats[2], outputDim, inputDim, 9)
        val q51 = canonicalBytes(formats[3], outputDim, inputDim, 9)
        val calls: List<Pair<String, (FloatArray, Int) -> Unit>> = listOf(
            "q8_0" to { out, off -> NativeQ8_0MatmulKernel.matmul(input, 0, q8, 0, inputDim, outputDim, out, off) },
            "q4_0" to { out, off -> NativeQ4_0MatmulKernel.matmul(input, 0, q4, 0, inputDim, outputDim, out, off) },
            "q5_0" to { out, off -> NativeQ5_0MatmulKernel.matmul(input, 0, q50, 0, inputDim, outputDim, out, off) },
            "q5_1" to { out, off -> NativeQ5_1MatmulKernel.matmul(input, 0, q51, 0, inputDim, outputDim, out, off) },
        )
        for ((name, call) in calls) {
            val reference = FloatArray(outputDim)
            call(reference, 0)
            assertTrue(reference.any { it != 0f }, "$name: reference output is all zero")
            val sentinel = 123.5f
            val out = FloatArray(3 * outputDim) { sentinel }
            call(out, outputDim) // the middle row
            for (i in 0 until outputDim) {
                assertEquals(sentinel, out[i], "$name: row before the call must be untouched at $i")
                assertEquals(sentinel, out[2 * outputDim + i], "$name: row after the call must be untouched at $i")
                assertEquals(reference[i], out[outputDim + i], "$name: the call's own row at $i")
            }
        }
    }
}
