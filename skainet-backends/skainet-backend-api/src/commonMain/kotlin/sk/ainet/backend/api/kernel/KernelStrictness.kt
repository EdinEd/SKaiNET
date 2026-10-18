package sk.ainet.backend.api.kernel

/**
 * Process-global fail-fast policy for kernel resolution.
 *
 * The RFC's "fail before execution" principle says that if a graph
 * can't find a registered kernel for an op/dtype combination, the
 * problem should surface at the load / compile / dispatch boundary
 * rather than as a silent perf regression at forward time (e.g. a
 * Q4_K matmul that falls back to a scalar dequant + FP32 matmul).
 *
 * The switch is common code so every tier honours it the same way:
 * [overrideEnabled] is the programmatic setting (the only one on JS, Wasm and
 * Kotlin/Native, which have no property store); when it is `null` the platform
 * setting is read — `-Dskainet.strict.kernels=true` on the JVM and Android.
 * Dispatch sites call [failIfStrict] right before they would otherwise silently
 * fall back; the call is a no-op when strict mode is off, preserving the
 * existing adaptive behaviour.
 *
 * Per-context strict mode (e.g. `DirectCpuExecutionContext.create(strict = true)`)
 * is a follow-up that requires plumbing the flag through every
 * platform's `platformDefaultCpuOpsFactory`.
 */
public object KernelStrictness {

    /** System-property name that flips fail-fast mode on (JVM and Android). */
    public const val SYSTEM_PROPERTY: String = "skainet.strict.kernels"

    /**
     * Programmatic switch; `null` means "read the platform setting". This is the
     * only switch on targets without system properties.
     */
    public var overrideEnabled: Boolean? = null

    /**
     * Returns `true` when the runtime should fail fast instead of
     * falling back. Reads the platform setting on every call so a
     * test can toggle it via `System.setProperty(...)` between
     * cases without restarting the JVM.
     */
    public fun isEnabled(): Boolean = overrideEnabled ?: platformStrictKernels()

    /**
     * Throws [NoSuchKernelException] with the supplied message
     * builder when strict mode is on; otherwise returns. The
     * message builder is only invoked when the exception is going
     * to be thrown, so callers can include expensive details
     * (provider list, dtype tuples) without paying the cost in the
     * default-adaptive path.
     */
    public inline fun failIfStrict(message: () -> String) {
        if (isEnabled()) throw NoSuchKernelException(message())
    }
}

/** Platform reading of [KernelStrictness.SYSTEM_PROPERTY]; `false` where there is no property store. */
internal expect fun platformStrictKernels(): Boolean

/**
 * Raised by dispatch sites when [KernelStrictness] is enabled and
 * no registered kernel matches the requested op/dtype combination.
 * The message includes the op name, the failing dtype tuple, and
 * the list of currently-registered providers so the operator can
 * see exactly which capability is missing.
 */
public class NoSuchKernelException(message: String) : RuntimeException(message)
