package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory, sk.ainet.context.schedule.Schedule) -> TensorOps {
    // Non-JVM has no ServiceLoader; register the scalar packed-quant kernels
    // (Q4_K/Q6_K/Q5_1/Q5_0/Q8_0/Q4_0) so DefaultCpuOpsBase can dispatch them.
    KernelRegistry.register(ScalarKernelProvider)
    // A browser kernel pack (SKEEP-007 Tier 1, `skainet-backend-wasm-simd`) cannot be discovered
    // here — there is no ServiceLoader on wasm and this module must not depend on it — so it
    // installs its ops subclass through [WasmCpuOpsOverride] and every context built afterwards
    // picks it up. Nothing installed means the common scalar ops, as before.
    return { factory, schedule -> WasmCpuOpsOverride.factory?.invoke(factory, schedule) ?: DefaultCpuOps(factory, schedule) }
}

/**
 * The one hook a wasm kernel pack has into context construction (SKEEP-007 §T1): set [factory] to
 * have `DirectCpuExecutionContext` build that `TensorOps` instead of the scalar [DefaultCpuOps].
 * The pack sets it from its `install()`; tests clear it with `null`.
 */
public object WasmCpuOpsOverride {
    public var factory: ((TensorDataFactory, sk.ainet.context.schedule.Schedule) -> TensorOps)? = null
}


internal actual fun platformDefaultSchedule(): sk.ainet.context.schedule.Schedule =
    sk.ainet.context.schedule.Schedule.Sequential
