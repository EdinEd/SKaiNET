package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Scalar Gauss error function and the exact GELU built on it (#1342).
 *
 * `erf` is Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7) evaluated in double precision; the float
 * result absorbs that error below its last bit. The expression order is the one the consumers
 * validated against their reference goldens, so the kernel reproduces them bit-for-bit.
 *
 * The `*Float` variants evaluate the same polynomial in single precision with the same operation
 * order; they serve the scalar tail of the JVM Vector API lane kernel, so lanes and tail agree.
 */
internal object ErfKernel {
    private const val P = 0.3275911
    private const val A1 = 0.254829592
    private const val A2 = -0.284496736
    private const val A3 = 1.421413741
    private const val A4 = -1.453152027
    private const val A5 = 1.061405429

    const val P_F: Float = 0.3275911f
    const val A1_F: Float = 0.254829592f
    const val A2_F: Float = -0.284496736f
    const val A3_F: Float = 1.421413741f
    const val A4_F: Float = -1.453152027f
    const val A5_F: Float = 1.061405429f
    /** 1/√2 in single precision. */
    const val SQRT1_2_F: Float = 0.70710678f

    /** `erf(x)` with absolute error below 1.5e-7, odd, saturating at ±1. */
    fun erf(x: Double): Double {
        // The polynomial's coefficients sum to 1 only to ~1e-9, so pin the origin (and keep the
        // sign of zero) instead of returning that residue with the sign of +0.
        if (x == 0.0) return x
        val sign = if (x < 0) -1.0 else 1.0
        val a = abs(x)
        val t = 1.0 / (1.0 + P * a)
        val y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * exp(-a * a)
        return sign * y
    }

    /** `0.5 · x · (1 + erf(x / √2))` in double precision, rounded once to float. */
    fun geluExact(x: Float): Float {
        val xd = x.toDouble()
        return (0.5 * xd * (1.0 + erf(xd / sqrt(2.0)))).toFloat()
    }

    /** Single-precision erf with the lane kernel's operation order (see `DefaultCpuOpsJvm.erfVector`). */
    fun erfFloat(x: Float): Float {
        if (x == 0f) return x
        val a = abs(x)
        val t = 1f / (a * P_F + 1f)
        var poly = t * A5_F + A4_F
        poly = poly * t + A3_F
        poly = poly * t + A2_F
        poly = poly * t + A1_F
        val y = 1f - poly * t * exp(-(a * a))
        return if (x < 0f) -y else y
    }

    /** Single-precision exact GELU matching the lane kernel. */
    fun geluExactFloat(x: Float): Float = 0.5f * x * (1f + erfFloat(x * SQRT1_2_F))
}
