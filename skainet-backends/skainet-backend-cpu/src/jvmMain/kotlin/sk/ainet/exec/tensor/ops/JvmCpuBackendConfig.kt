package sk.ainet.exec.tensor.ops

internal object JvmCpuBackendConfig {
    private const val ENV_VECTOR = "SKAINET_CPU_VECTOR_ENABLED"
    private const val PROP_VECTOR = "skainet.cpu.vector.enabled"
    private const val ENV_BLAS = "SKAINET_CPU_BLAS_ENABLED"
    private const val PROP_BLAS = "skainet.cpu.blas.enabled"

    val vectorEnabled: Boolean
        get() = readFlag(ENV_VECTOR, PROP_VECTOR, default = true)
    /**
     * Whether dense FP32 products go to the platform BLAS ([JvmBlas]) when one resolves. On by
     * default on macOS, where the BLAS is Accelerate and runs on the matrix units of Apple silicon;
     * off elsewhere, where a BLAS is an optional install and the bundled kernels are the known
     * quantity. `-Dskainet.cpu.blas.enabled` / `SKAINET_CPU_BLAS_ENABLED` override either way, and
     * `-Dskainet.cpu.blas.library` / `SKAINET_CPU_BLAS_LIBRARY` name the library to load.
     */
    val blasEnabled: Boolean
        get() = readFlag(ENV_BLAS, PROP_BLAS, default = isMacOs)

    private val isMacOs: Boolean by lazy {
        val os = runCatching { System.getProperty("os.name") }.getOrNull()?.lowercase() ?: ""
        os.contains("mac") || os.contains("darwin")
    }

    private fun readFlag(envKey: String, propKey: String, default: Boolean): Boolean {
        val sysProp = runCatching { System.getProperty(propKey) }.getOrNull()
            ?.toBooleanStrictOrNull()
        val env = runCatching { System.getenv(envKey) }.getOrNull()
            ?.toBooleanStrictOrNull()
        return sysProp ?: env ?: default
    }
}
