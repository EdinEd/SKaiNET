package sk.ainet.lang.ops

/**
 * Declares how an op lowers on the compile path, for the generated operator documentation
 * (the *Lowering* column of the coverage matrix) and for tooling that wants one place to read
 * what an op can become (#1342).
 *
 * @param composite Namespaced `stablehlo.composite` name emitted when a target's
 *   `OpGranularityPolicy` keeps the op fused (e.g. `"skainet.gelu"`); empty when the op has no
 *   composite form.
 * @param decomposes `true` when the op has a decomposition into StableHLO/CHLO primitives
 *   (the portable default emission).
 * @param native `true` when the op maps 1:1 onto a target primitive (e.g. `stablehlo.tanh`).
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class Lowering(
    val composite: String = "",
    val decomposes: Boolean = true,
    val native: Boolean = false,
)
