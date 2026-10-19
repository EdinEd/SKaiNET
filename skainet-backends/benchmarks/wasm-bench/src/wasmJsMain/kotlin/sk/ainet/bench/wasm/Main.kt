package sk.ainet.bench.wasm

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.wasm.WasmCpuOps
import sk.ainet.exec.wasm.WasmKernels
import sk.ainet.exec.wasm.WasmSimd
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.time.TimeSource

/**
 * SKEEP-007 wasm baseline: the matmul shapes one decision / query / decode step of the shipped
 * models puts through `matmulWeightTransposed`, dense FP32 and packed Q8_0, timed in the eager
 * backend exactly as the samples run it. Output is one line per case:
 *
 * `[SKAINET_PERF_WASM] <case> m=<m> n=<n> k=<k> form=<dense|q8_0> ms=<median> gflops=<rate>`
 *
 * Every later tier (SIMD128 sidecar, quantize-at-load) is measured against these lines. Each pass
 * also runs the `engine-*` scenarios of the JVM publish harness ([runEngineScenarios]) so the two
 * tiers are compared on identical shapes.
 */
private data class Case(val name: String, val m: Int, val n: Int, val k: Int)

private val cases = listOf(
    Case("laya.Wqkv", 150, 2304, 768),
    Case("laya.Wi", 150, 2304, 768),
    Case("laya.mlpWo", 150, 768, 1152),
    Case("laya.Wo", 150, 768, 768),
    Case("leaf.ffn", 20, 1536, 384),
    Case("leaf.proj", 20, 384, 384),
    Case("smollm2.gemv", 1, 1536, 576),
    Case("qwen06b.gemv", 1, 3072, 1024),
)

private fun pseudo(n: Int, seed: Int): FloatArray {
    var s = seed
    return FloatArray(n) {
        s = s * 1103515245 + 12345
        ((s ushr 16) % 2001 - 1000) / 1000f
    }
}

/** Row-major Q8_0 blocks (fp16 scale + 32 int8), the GGUF layout, from dense rows. */
@Suppress("UNCHECKED_CAST")
private fun q8_0Weight(ctx: DirectCpuExecutionContext, n: Int, k: Int, seed: Int): Tensor<FP32, Float> {
    require(k % 32 == 0)
    val values = pseudo(n * k, seed)
    val blocksPerRow = k / 32
    val bytes = ByteArray(n * blocksPerRow * 34)
    for (r in 0 until n) for (b in 0 until blocksPerRow) {
        val base = (r * blocksPerRow + b) * 34
        val off = r * k + b * 32
        var amax = 0f
        for (i in 0 until 32) amax = maxOf(amax, abs(values[off + i]))
        val d = amax / 127f
        val id = if (d != 0f) 1f / d else 0f
        val bits = Fp16Codec.encode(d)
        bytes[base] = (bits and 0xFF).toByte()
        bytes[base + 1] = ((bits ushr 8) and 0xFF).toByte()
        for (i in 0 until 32) bytes[base + 2 + i] = (values[off + i] * id).roundToInt().coerceIn(-127, 127).toByte()
    }
    return ctx.fromData(Q8_0BlockTensorData(Shape(n, k), bytes) as TensorData<FP32, Float>, FP32::class)
}

private fun timeMedianMs(reps: Int, block: () -> Unit): Double {
    val clock = TimeSource.Monotonic
    val samples = DoubleArray(reps) {
        val t0 = clock.markNow()
        block()
        t0.elapsedNow().inWholeMicroseconds / 1000.0
    }
    samples.sort()
    return samples[reps / 2]
}

private fun report(tier: String, case: Case, form: String, ms: Double) {
    val flops = 2.0 * case.m * case.n * case.k
    val gflops = flops / (ms / 1000.0) / 1e9
    val msText = ((ms * 100).roundToInt() / 100.0).toString()
    val gfText = ((gflops * 100).roundToInt() / 100.0).toString()
    println("[SKAINET_PERF_WASM] tier=$tier ${case.name} m=${case.m} n=${case.n} k=${case.k} form=$form ms=$msText gflops=$gfText")
}

/**
 * Three passes: the scalar lane, then the SIMD128 sidecar single-threaded, then with pthreads when
 * the runtime allows them (Node, or a cross-origin-isolated page). Each line carries `tier=`.
 */
suspend fun main() {
    runPass("scalar")
    WasmSimd.install(threads = false)
    runPass(WasmKernels.mode)
    WasmSimd.uninstall()
    try {
        WasmSimd.install(threads = true)
        runPass(WasmKernels.mode)
    } catch (e: Throwable) {
        println("[SKAINET_PERF_WASM] tier=simd128-mt skipped: ${e.message}")
    } finally {
        WasmSimd.uninstall()
    }
    println("[SKAINET_PERF_WASM] done")
}

private fun runPass(tier: String) {
    val ctx = DirectCpuExecutionContext()
    println("[SKAINET_PERF_WASM] tier=$tier backend=${ctx.ops::class.simpleName} cases=${cases.size} features=${WasmKernels.features()}")
    runEngineScenarios(tier)
    for (case in cases) {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(case.m, case.k), FP32::class, pseudo(case.m * case.k, 3))
        val dense = ctx.fromFloatArray<FP32, Float>(Shape(case.n, case.k), FP32::class, pseudo(case.n * case.k, 7))
        val packed = q8_0Weight(ctx, case.n, case.k, 11)
        // Warm-up: first call pays the one-time weight relayout / caches.
        ctx.ops.matmulWeightTransposed(x, dense)
        ctx.ops.matmulWeightTransposed(x, packed)
        val reps = if (case.m == 1) 50 else 5
        report(tier, case, "dense", timeMedianMs(reps) { ctx.ops.matmulWeightTransposed(x, dense) })
        report(tier, case, "q8_0", timeMedianMs(reps) { ctx.ops.matmulWeightTransposed(x, packed) })
    }
    runChain(tier, ctx)
}

/**
 * The resident-activation case: one Laya MLP block's shape of ops between two matmuls —
 * `x · Wᵢᵀ`, GELU, a gated multiply, a bias add, then a reduction that reads the result back —
 * timed end to end. On the scalar lane every op allocates and walks a heap array; on the sidecar
 * the intermediates stay in linear memory and only the final scalar crosses the boundary.
 *
 * `[SKAINET_PERF_WASM] tier=<tier> laya.chain m=150 n=2304 k=768 ops=matmulWt+gelu+mul+add+sum ms=<median>`
 */
private fun runChain(tier: String, ctx: DirectCpuExecutionContext) {
    val m = 150; val n = 2304; val k = 768
    val x = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, pseudo(m * k, 3))
    val w = ctx.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, pseudo(n * k, 7))
    val bias = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, pseudo(n, 11))
    fun chain(): Float {
        val h = ctx.ops.matmulWeightTransposed(x, w)
        val g = ctx.ops.gelu(h)
        val s = ctx.ops.multiply(g, h)
        val o = ctx.ops.add(s, bias)
        return ctx.ops.sum(o).data.copyToFloatArray()[0]
    }
    chain()
    val ms = timeMedianMs(5) { chain() }
    val msText = ((ms * 100).roundToInt() / 100.0).toString()
    val resident = (ctx.ops as? WasmCpuOps) != null
    println("[SKAINET_PERF_WASM] tier=$tier laya.chain m=$m n=$n k=$k ops=matmulWt+gelu+mul+add+sum resident=$resident ms=$msText")
}
