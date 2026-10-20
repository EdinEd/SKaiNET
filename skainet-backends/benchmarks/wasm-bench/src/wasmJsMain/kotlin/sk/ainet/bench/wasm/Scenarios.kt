package sk.ainet.bench.wasm

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelKey
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.wasm.WasmCpuOps
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.types.FP32
import kotlin.math.roundToInt
import kotlin.time.TimeSource

/**
 * The `engine-*` scenarios of `skainet-backends/benchmarks/jvm-cpu-publish`, run on the wasm tier
 * with the same shapes, data generators and metrics, so one table compares the two tiers:
 *
 * `[SKAINET_PERF_WASM] tier=<tier> scenario=<id> metric=<gflops|gops|melems_per_sec> value=<v> <params>`
 *
 * Kernel scenarios take the best registered provider (`scalar` before `WasmSimd.install()`,
 * `wasm-simd128` after); ops scenarios go through a fresh `DirectCpuExecutionContext`, exactly as
 * the JVM harness does. A scenario the tier cannot serve yet prints `value=skipped reason=…`.
 */
internal class WasmScenario(
    val id: String,
    val metric: String,
    val params: Map<String, String>,
    val reps: Int,
    val setup: () -> (() -> Double),
)

private fun lcg(seed: Int): () -> Float {
    var s = seed
    return {
        s = s * 1103515245 + 12345
        ((s ushr 16) and 0x7FFF) / 32768f
    }
}

internal fun engineScenarios(): List<WasmScenario> = listOf(
    WasmScenario("engine-fp32-gemm", "gflops", mapOf("size" to "1024"), reps = 3) {
        val n = 1024
        val ctx = DirectCpuExecutionContext()
        val a = ctx.fromFloatArray<FP32, Float>(Shape(n, n), FP32::class, FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f })
        val b = ctx.fromFloatArray<FP32, Float>(Shape(n, n), FP32::class, FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f })
        val flops = 2.0 * n * n * n;
        { flops / seconds { ctx.ops.matmul(a, b) } / 1e9 }
    },
    WasmScenario("engine-fp32-linear", "gflops", mapOf("m" to "16", "k" to "768", "n" to "3072"), reps = 10) {
        val m = 16; val k = 768; val n = 3072
        val ctx = DirectCpuExecutionContext()
        val x = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, FloatArray(m * k) { ((it % 251) - 125).toFloat() / 127f })
        val w = ctx.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, FloatArray(n * k) { ((it * 13 % 257) - 128).toFloat() / 127f })
        ctx.ops.matmulWeightTransposed(x, w) // the resident upload happens here, not in a measured run
        val flops = 2.0 * m * n * k;
        { flops / seconds { ctx.ops.matmulWeightTransposed(x, w) } / 1e9 }
    },
    WasmScenario("engine-kernel-matmul", "gflops", mapOf("size" to "1024"), reps = 3) {
        val n = 1024
        DirectCpuExecutionContext()
        val kernel = checkNotNull(KernelRegistry.bestAvailable()?.matmulFp32()) { "no FP32 kernel" }
        val a = FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f }
        val b = FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f }
        val out = FloatArray(n * n)
        val flops = 2.0 * n * n * n;
        { flops / seconds { kernel.matmul(a, 0, n, b, 0, n, out, 0, n, n, n, n) } / 1e9 }
    },
    WasmScenario("engine-bf16-matmul", "gflops", mapOf("m" to "1024", "n" to "1024", "k" to "1024"), reps = 3) {
        val n = 1024
        DirectCpuExecutionContext()
        val kernel = checkNotNull(KernelRegistry.bestAvailable()?.matmulBf16()) { "no BF16 kernel" }
        val rnd = lcg(n)
        val a = FloatArray(n * n) { rnd() - 0.5f }
        val b = ByteArray(n * n * 2)
        for (i in 0 until n * n) {
            val bits = (rnd() - 0.5f).toRawBits() ushr 16
            b[i * 2] = (bits and 0xFF).toByte()
            b[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        val out = FloatArray(n * n)
        val flops = 2.0 * n * n * n;
        { flops / seconds { kernel.matmul(a, 0, n, b, 0, n * 2, out, 0, n, n, n, n) } / 1e9 }
    },
    WasmScenario("engine-q8-matmul", "gops", mapOf("input_dim" to "4096", "output_dim" to "4096"), reps = 10) {
        val inputDim = 4096; val outputDim = 4096
        DirectCpuExecutionContext()
        val kernel = checkNotNull(KernelRegistry.bestAvailable()?.matmulQ8_0()) { "no Q8_0 kernel" }
        val numBlocks = inputDim / 32 * outputDim
        val rnd = lcg(inputDim + outputDim)
        val weight = ByteArray(numBlocks * 34) { (rnd() * 255f).toInt().toByte() }
        for (block in 0 until numBlocks) { weight[block * 34] = 0x00; weight[block * 34 + 1] = 0x22 }
        val input = FloatArray(inputDim) { rnd() - 0.5f }
        val output = FloatArray(outputDim)
        val ops = 2.0 * inputDim * outputDim;
        { ops / seconds { kernel.matmul(input, 0, weight, 0, inputDim, outputDim, output, 0) } / 1e9 }
    },
    WasmScenario("engine-q4-gemm", "gops", mapOf("input_dim" to "4096", "output_dim" to "4096"), reps = 10) {
        val inputDim = 4096; val outputDim = 4096
        DirectCpuExecutionContext()
        val kernel = checkNotNull(KernelRegistry.bestAvailable()?.matmulQ4K()) { "no Q4_K kernel" }
        val numBlocks = inputDim / 256 * outputDim
        val rnd = lcg(42)
        val weight = ByteArray(numBlocks * 144) { (rnd() * 255f).toInt().toByte() }
        for (block in 0 until numBlocks) {
            val base = block * 144
            weight[base] = 0x00; weight[base + 1] = 0x3C; weight[base + 2] = 0x00; weight[base + 3] = 0x3C
        }
        val input = FloatArray(inputDim) { ((it % 251) - 125).toFloat() / 127f }
        val output = FloatArray(outputDim)
        val ops = 2.0 * inputDim * outputDim;
        { ops / seconds { kernel.matmul(input, 0, weight, 0, inputDim, outputDim, output, 0) } / 1e9 }
    },
    WasmScenario("engine-ternary-f32-gemv", "gops", mapOf("input_dim" to "2560", "output_dim" to "6912"), reps = 5) {
        val inputDim = 2560; val outputDim = 6912
        val rnd = lcg(inputDim + outputDim)
        val values = FloatArray(outputDim * inputDim) { ((rnd() * 3f).toInt() - 1) * 0.5f }
        val weight = BitNetB158TensorData.fromFloats(Shape(outputDim, inputDim), values).packedView
        val activation = TensorView.dense(Storage.Heap.wrap(FloatArray(inputDim) { rnd() - 0.5f }), Shape(1, inputDim), FP32)
        // The decoding reference kernel is orders of magnitude slower than a real kernel and
        // takes hours at this shape on wasm; measure only when a ternary kernel is registered.
        checkNotNull(KernelDispatch.find(KernelKey.matmul(activation, weight))) { "no ternary kernel registered on this tier" }
        val out = TensorView.dense(Storage.Heap.floats(outputDim), Shape(1, outputDim), FP32)
        val ops = 2.0 * inputDim * outputDim;
        { ops / seconds { KernelDispatch.matmul(activation, weight, out, Scope.Ambient) } / 1e9 }
    },
    WasmScenario("engine-elementwise-add", "melems_per_sec", mapOf("elements" to "1000000"), reps = 10) {
        val n = 1_000_000
        val ctx = DirectCpuExecutionContext()
        val a = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { it.toFloat() * 0.5f })
        val b = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { 1f });
        { n / seconds { ctx.ops.add(a, b) } / 1e6 }
    },
    // The same three ops on operands already resident in the sidecar (the shape a chain between two
    // matmuls has); heap operands above stay on the common lane by design.
    WasmScenario("engine-elementwise-add", "melems_per_sec", mapOf("elements" to "1000000", "operands" to "resident"), reps = 10) {
        val n = 1_000_000
        val ctx = DirectCpuExecutionContext()
        val ops = ctx.ops as? WasmCpuOps ?: error("resident operands need the sidecar")
        val a = ops.toResident(ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { it.toFloat() * 0.5f }))
        val b = ops.toResident(ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { 1f }));
        { n / seconds { ctx.ops.add(a, b) } / 1e6 }
    },
    WasmScenario("engine-reductions-sum", "melems_per_sec", mapOf("elements" to "1000000", "op" to "sum", "operands" to "resident"), reps = 10) {
        val n = 1_000_000
        val ctx = DirectCpuExecutionContext()
        val ops = ctx.ops as? WasmCpuOps ?: error("resident operands need the sidecar")
        val a = ops.toResident(ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { (it % 1024).toFloat() * 0.25f }));
        { n / seconds { ctx.ops.sum(a).data.copyToFloatArray() } / 1e6 }
    },
    WasmScenario("engine-reductions-sum", "melems_per_sec", mapOf("elements" to "1000000", "op" to "sum"), reps = 10) {
        val n = 1_000_000
        val ctx = DirectCpuExecutionContext()
        val a = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { (it % 1024).toFloat() * 0.25f });
        { n / seconds { ctx.ops.sum(a) } / 1e6 }
    },
    WasmScenario("engine-reductions-mean", "melems_per_sec", mapOf("elements" to "1000000", "op" to "mean"), reps = 10) {
        val n = 1_000_000
        val ctx = DirectCpuExecutionContext()
        val a = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { (it % 1024).toFloat() * 0.25f });
        { n / seconds { ctx.ops.mean(a) } / 1e6 }
    },
    // The fused row ops (softmax, a LayerNormalization layer, causal attention), the shapes of the JVM harness.
    WasmScenario("engine-softmax", "melems_per_sec", mapOf("rows" to "1024", "cols" to "1024"), reps = 10) {
        val rows = 1024; val cols = 1024
        val ctx = DirectCpuExecutionContext()
        val rnd = lcg(3)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, FloatArray(rows * cols) { (rnd() - 0.5f) * 8f })
        ctx.ops.softmax(x, -1);
        { rows.toDouble() * cols / seconds { ctx.ops.softmax(x, -1) } / 1e6 }
    },
    WasmScenario("engine-layer-norm", "melems_per_sec", mapOf("rows" to "1024", "cols" to "768", "affine" to "true"), reps = 10) {
        val rows = 1024; val cols = 768
        val ctx = DirectCpuExecutionContext()
        val rnd = lcg(5)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, FloatArray(rows * cols) { (rnd() - 0.5f) * 4f })
        val gamma = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, FloatArray(cols) { 0.5f + (it % 7) * 0.1f })
        val beta = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, FloatArray(cols) { ((it % 5) - 2) * 0.05f })
        val layer = LayerNormalization<FP32, Float>(intArrayOf(cols), eps = 1e-5, elementwiseAffine = true, name = "ln", initGamma = gamma, initBeta = beta)
        layer.forward(x, ctx);
        { rows.toDouble() * cols / seconds { layer.forward(x, ctx) } / 1e6 }
    },
    WasmScenario("engine-sdpa", "gflops", mapOf("batch" to "1", "heads" to "12", "seq" to "150", "head_dim" to "64", "causal" to "true"), reps = 5) {
        val heads = 12; val seq = 150; val headDim = 64
        val ctx = DirectCpuExecutionContext()
        val rnd = lcg(7)
        val n = heads * seq * headDim
        val q = ctx.fromFloatArray<FP32, Float>(Shape(1, heads, seq, headDim), FP32::class, FloatArray(n) { rnd() - 0.5f })
        val k = ctx.fromFloatArray<FP32, Float>(Shape(1, heads, seq, headDim), FP32::class, FloatArray(n) { rnd() - 0.5f })
        val v = ctx.fromFloatArray<FP32, Float>(Shape(1, heads, seq, headDim), FP32::class, FloatArray(n) { rnd() - 0.5f })
        ctx.ops.scaledDotProductAttention(q, k, v, causal = true)
        val flops = 4.0 * heads * seq * seq * headDim;
        { flops / seconds { ctx.ops.scaledDotProductAttention(q, k, v, causal = true) } / 1e9 }
    },
    // A CNN's 3×3 same-padding layer, 64 → 64 channels over a 56 × 56 map, with bias: im2col + GEMM
    // in the sidecar (the common scalar loop before WasmSimd.install()).
    WasmScenario("engine-conv2d", "gflops", mapOf("batch" to "1", "c_in" to "64", "c_out" to "64", "hw" to "56", "kernel_size" to "3", "bias" to "true"), reps = 5) {
        val cIn = 64; val cOut = 64; val hw = 56; val k = 3
        val ctx = DirectCpuExecutionContext()
        val rnd = lcg(9)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, cIn, hw, hw), FP32::class, FloatArray(cIn * hw * hw) { rnd() - 0.5f })
        val w = ctx.fromFloatArray<FP32, Float>(Shape(cOut, cIn, k, k), FP32::class, FloatArray(cOut * cIn * k * k) { (rnd() - 0.5f) * 0.2f })
        val b = ctx.fromFloatArray<FP32, Float>(Shape(cOut), FP32::class, FloatArray(cOut) { rnd() - 0.5f })
        ctx.ops.conv2d(x, w, b, 1 to 1, 1 to 1, 1 to 1, 1) // the weight's resident upload happens here
        val flops = 2.0 * cOut * hw * hw * cIn * k * k;
        { flops / seconds { ctx.ops.conv2d(x, w, b, 1 to 1, 1 to 1, 1 to 1, 1) } / 1e9 }
    },
)

private inline fun seconds(block: () -> Unit): Double {
    val t0 = TimeSource.Monotonic.markNow()
    block()
    return t0.elapsedNow().inWholeNanoseconds / 1e9
}

/** Run every scenario once for warm-up, then report the median of [WasmScenario.reps] runs. */
internal fun runEngineScenarios(tier: String) {
    for (s in engineScenarios()) {
        val params = s.params.entries.joinToString(" ") { "${it.key}=${it.value}" }
        val run = try {
            s.setup()
        } catch (e: Throwable) {
            println("[SKAINET_PERF_WASM] tier=$tier scenario=${s.id} metric=${s.metric} value=skipped reason=${e.message} $params")
            continue
        }
        run()
        val samples = DoubleArray(s.reps) { run() }
        samples.sort()
        val median = samples[s.reps / 2]
        val text = ((median * 100).roundToInt() / 100.0).toString()
        println("[SKAINET_PERF_WASM] tier=$tier scenario=${s.id} metric=${s.metric} value=$text kernel=${KernelRegistry.bestAvailable()?.name} $params")
    }
}
