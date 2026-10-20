package sk.ainet.bench.publish.scenarios

import sk.ainet.bench.publish.runner.Scenario

public object ScenarioRegistry {

    public fun ids(): List<String> = listOf(
        "engine-fp32-gemm",
        "engine-fp32-linear",
        "engine-q4-gemm",
        "engine-kernel-matmul",
        "engine-bf16-matmul",
        "engine-q8-matmul",
        "engine-ternary-f32-gemv",
        "engine-elementwise-add",
        "engine-reductions-sum",
        "engine-reductions-mean",
        "engine-softmax",
        "engine-layer-norm",
        "engine-sdpa",
    )

    public fun byId(id: String, provider: String, smoke: Boolean): Scenario? = when (id) {
        "engine-fp32-gemm" -> Fp32GemmScenario(smoke = smoke, providerName = provider)
        "engine-fp32-linear" -> Fp32LinearScenario(smoke = smoke, providerName = provider)
        "engine-q4-gemm" -> Q4GemmScenario(smoke = smoke, providerName = provider)
        "engine-kernel-matmul" -> KernelMatmulScenario(smoke = smoke, providerName = provider)
        "engine-bf16-matmul" -> Bf16MatmulScenario(smoke = smoke, providerName = provider)
        "engine-q8-matmul" -> Q8MatmulScenario(smoke = smoke, providerName = provider)
        "engine-ternary-f32-gemv" -> TernaryF32GemvScenario(smoke = smoke, providerName = provider)
        "engine-elementwise-add" -> ElementwiseAddScenario(smoke = smoke, providerName = provider)
        "engine-reductions-sum" -> ReductionsScenario(op = ReductionOp.SUM, smoke = smoke, providerName = provider)
        "engine-reductions-mean" -> ReductionsScenario(op = ReductionOp.MEAN, smoke = smoke, providerName = provider)
        "engine-softmax" -> SoftmaxScenario(smoke = smoke, providerName = provider)
        "engine-layer-norm" -> LayerNormScenario(smoke = smoke, providerName = provider)
        "engine-sdpa" -> SdpaScenario(smoke = smoke, providerName = provider)
        else -> null
    }
}
