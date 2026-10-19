package sk.ainet.backend.api.kernel

internal actual fun platformStrictKernels(): Boolean =
    System.getProperty(KernelStrictness.SYSTEM_PROPERTY) == "true"
