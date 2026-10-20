package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.NativeKernelProvider
import sk.ainet.exec.kernel.PanamaVectorKernelProvider
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * A `Linear` layer's product, `x · Wᵀ` with the weight stored `[out, in]`, through the public
 * `ctx.ops.matmulWeightTransposed` path — what every projection of a transformer pays, as opposed to
 * the `[k, n]` GEMM of `engine-fp32-gemm`. Shape: a 16-token prefill against a 768 → 3072 FFN
 * projection. Primary metric: GFLOP/s over `2·m·n·k`.
 *
 * The provider selects the path: `panama` (the default) transposes the weight once into `[k, n]`
 * and runs the Vector API GEMM; `ffm` runs the native direct kernel (`skainet_fp32_matmul_wt`) over a
 * weight made resident off-heap, no transposed copy; `scalar` is the common row-dot. Only the chosen
 * provider is registered, so the ops layer cannot fall through to another.
 */
internal class Fp32LinearScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-fp32-linear"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val m: Int = 16
    private val k: Int = 768
    private val n: Int = if (smoke) 768 else 3072
    override val parameters: Map<String, String> = mapOf(
        "m" to m.toString(),
        "k" to k.toString(),
        "n" to n.toString(),
        "kernel" to providerName,
    )

    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var x: Tensor<FP32, Float>
    private lateinit var w: Tensor<FP32, Float>

    override fun setup() {
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
        ctx = DirectCpuExecutionContext()
        x = ctx.fromFloatArray(Shape(m, k), FP32::class, FloatArray(m * k) { ((it % 251) - 125).toFloat() / 127f })
        w = ctx.fromFloatArray(Shape(n, k), FP32::class, FloatArray(n * k) { ((it * 13 % 257) - 128).toFloat() / 127f })
        // First call: the transpose cache or the resident copy is built here, not in a measured run.
        ctx.ops.matmulWeightTransposed(x, w)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.matmulWeightTransposed(x, w)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val flops = 2.0 * m.toDouble() * n.toDouble() * k.toDouble()
        return (flops / (elapsedNs / 1_000_000_000.0)) / 1e9
    }

    override fun teardown() {
        KernelRegistry.clearForTesting()
    }
}
