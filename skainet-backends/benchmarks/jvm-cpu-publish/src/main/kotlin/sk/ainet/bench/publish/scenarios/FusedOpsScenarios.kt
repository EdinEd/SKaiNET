package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.NativeKernelProvider
import sk.ainet.exec.kernel.PanamaVectorKernelProvider
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/*
 * The fused row ops through the public ops layer (SKEEP-007 fused ops): softmax, a `LayerNormalization`
 * layer and scaled dot-product attention. The provider selects the path, as in [Fp32LinearScenario]:
 * `ffm` runs the native fused kernels (`f32_fused.c` through FFM), `panama` and `scalar` the common
 * Kotlin implementation (attention parallel over heads through the context's schedule, the
 * normalisation as its decomposed elementwise chain). Only the chosen provider is registered.
 */

private fun registerOnly(providerName: String) {
    KernelRegistry.clearForTesting()
    when (providerName.lowercase()) {
        "panama", "panama-vector" -> KernelRegistry.register(PanamaVectorKernelProvider)
        "ffm", "native", "native-ffm" -> {
            check(NativeKernelProvider.isAvailable()) { "bundled libskainet_kernels missing — the ffm provider cannot run" }
            KernelRegistry.register(NativeKernelProvider)
        }
        "scalar" -> KernelRegistry.register(ScalarKernelProvider)
        else -> error("unknown kernel provider: $providerName (use 'panama', 'ffm' or 'scalar')")
    }
}

private fun pseudo(n: Int, seed: Int, scale: Float = 1f): FloatArray {
    var s = seed
    return FloatArray(n) {
        s = s * 1103515245 + 12345
        ((s ushr 16) % 2001 - 1000) / 1000f * scale
    }
}

/** `ctx.ops.softmax` over the last dimension of `[rows, 1024]` FP32. Primary metric: million elements per second. */
internal class SoftmaxScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-softmax"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "melems_per_sec"
    override val unit: String = "melems_per_sec"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val rows: Int = if (smoke) 64 else 1024
    private val cols: Int = 1024
    override val parameters: Map<String, String> = mapOf(
        "rows" to rows.toString(),
        "cols" to cols.toString(),
        "kernel" to providerName,
    )

    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var x: Tensor<FP32, Float>

    override fun setup() {
        registerOnly(providerName)
        ctx = DirectCpuExecutionContext()
        x = ctx.fromFloatArray(Shape(rows, cols), FP32::class, pseudo(rows * cols, 3, 4f))
        ctx.ops.softmax(x, -1)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.softmax(x, -1)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        return (rows.toDouble() * cols / (elapsedNs / 1_000_000_000.0)) / 1e6
    }

    override fun teardown() = KernelRegistry.clearForTesting()
}

/** A `LayerNormalization` layer with affine over `[rows, 768]` FP32. Primary metric: million elements per second. */
internal class LayerNormScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-layer-norm"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "melems_per_sec"
    override val unit: String = "melems_per_sec"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val rows: Int = if (smoke) 64 else 1024
    private val cols: Int = 768
    override val parameters: Map<String, String> = mapOf(
        "rows" to rows.toString(),
        "cols" to cols.toString(),
        "affine" to "true",
        "kernel" to providerName,
    )

    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var x: Tensor<FP32, Float>
    private lateinit var layer: LayerNormalization<FP32, Float>

    override fun setup() {
        registerOnly(providerName)
        ctx = DirectCpuExecutionContext()
        x = ctx.fromFloatArray(Shape(rows, cols), FP32::class, pseudo(rows * cols, 5, 2f))
        val gamma = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, FloatArray(cols) { 0.5f + (it % 7) * 0.1f })
        val beta = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, FloatArray(cols) { ((it % 5) - 2) * 0.05f })
        layer = LayerNormalization(intArrayOf(cols), eps = 1e-5, elementwiseAffine = true, name = "ln", initGamma = gamma, initBeta = beta)
        layer.forward(x, ctx)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = layer.forward(x, ctx)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        return (rows.toDouble() * cols / (elapsedNs / 1_000_000_000.0)) / 1e6
    }

    override fun teardown() = KernelRegistry.clearForTesting()
}

/**
 * `ctx.ops.scaledDotProductAttention`, causal, at an encoder's shape: 12 heads, 150 queries over 150
 * keys, head dim 64. Primary metric: GFLOP/s over `4 · heads · seqQ · seqKV · headDim` (the two
 * products; the causal half that is never computed is counted, as the dense reference counts it).
 */
internal class SdpaScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-sdpa"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val heads: Int = 12
    private val seq: Int = if (smoke) 32 else 150
    private val headDim: Int = 64
    override val parameters: Map<String, String> = mapOf(
        "batch" to "1",
        "heads" to heads.toString(),
        "seq" to seq.toString(),
        "head_dim" to headDim.toString(),
        "causal" to "true",
        "kernel" to providerName,
    )

    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var q: Tensor<FP32, Float>
    private lateinit var k: Tensor<FP32, Float>
    private lateinit var v: Tensor<FP32, Float>

    override fun setup() {
        registerOnly(providerName)
        ctx = DirectCpuExecutionContext()
        val n = heads * seq * headDim
        q = ctx.fromFloatArray(Shape(1, heads, seq, headDim), FP32::class, pseudo(n, 7))
        k = ctx.fromFloatArray(Shape(1, heads, seq, headDim), FP32::class, pseudo(n, 11))
        v = ctx.fromFloatArray(Shape(1, heads, seq, headDim), FP32::class, pseudo(n, 13))
        ctx.ops.scaledDotProductAttention(q, k, v, causal = true)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.scaledDotProductAttention(q, k, v, causal = true)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val flops = 4.0 * heads * seq * seq * headDim
        return (flops / (elapsedNs / 1_000_000_000.0)) / 1e9
    }

    override fun teardown() = KernelRegistry.clearForTesting()
}

/**
 * `ctx.ops.conv2d`: a CNN's 3 × 3 same-padding layer, 64 → 64 channels over a 56 × 56 map, with bias.
 * Primary metric: GFLOP/s over `2 · c_out · out_h · out_w · c_in · k_h · k_w`. On the JVM every
 * provider measures the same path — `DefaultCpuOpsJvm`'s im2col + Vector API kernels, which no
 * provider replaces; the `kernel` parameter records which one was registered. The wasm tier runs the
 * same scenario as im2col + GEMM in the sidecar.
 */
internal class Conv2dScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-conv2d"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val cIn: Int = if (smoke) 16 else 64
    private val cOut: Int = if (smoke) 16 else 64
    private val hw: Int = if (smoke) 28 else 56
    private val k: Int = 3
    override val parameters: Map<String, String> = mapOf(
        "batch" to "1",
        "c_in" to cIn.toString(),
        "c_out" to cOut.toString(),
        "hw" to hw.toString(),
        "kernel_size" to k.toString(),
        "bias" to "true",
        "kernel" to providerName,
    )

    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var x: Tensor<FP32, Float>
    private lateinit var w: Tensor<FP32, Float>
    private lateinit var b: Tensor<FP32, Float>

    override fun setup() {
        registerOnly(providerName)
        ctx = DirectCpuExecutionContext()
        x = ctx.fromFloatArray(Shape(1, cIn, hw, hw), FP32::class, pseudo(cIn * hw * hw, 9))
        w = ctx.fromFloatArray(Shape(cOut, cIn, k, k), FP32::class, pseudo(cOut * cIn * k * k, 17, 0.2f))
        b = ctx.fromFloatArray(Shape(cOut), FP32::class, pseudo(cOut, 19))
        ctx.ops.conv2d(x, w, b, 1 to 1, 1 to 1, 1 to 1, 1)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.conv2d(x, w, b, 1 to 1, 1 to 1, 1 to 1, 1)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val flops = 2.0 * cOut * hw * hw * cIn * k * k
        return (flops / (elapsedNs / 1_000_000_000.0)) / 1e9
    }

    override fun teardown() = KernelRegistry.clearForTesting()
}
