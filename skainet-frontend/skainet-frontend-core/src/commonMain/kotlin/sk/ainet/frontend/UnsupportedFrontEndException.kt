package sk.ainet.frontend

/**
 * Thrown when a checkpoint carries evidence for a front end that no provider or algorithm
 * supports, or when a side file is malformed. A wrong heuristic MUST fail loudly with this
 * exception, never degrade silently (FE-R-05). The message names the side file and, where known,
 * the families the registry knows about.
 */
public open class UnsupportedFrontEndException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
