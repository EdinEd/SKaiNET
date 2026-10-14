package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext

/**
 * Minimal emission surface a lowering builder writes through: fresh SSA names and operation
 * lines. [ConversionContext] is one implementation; a buffer that collects the body of a
 * `func.func private` decomposition is another, so one builder serves both the inline form and
 * a `stablehlo.composite` decomposition without the two ever drifting apart.
 */
public interface HloEmitter {
    /** A fresh, module-unique SSA value name. */
    public fun newValue(): String

    /** Emit one operation line. */
    public fun op(line: String)
}

/** [HloEmitter] over a [ConversionContext]; remembers the emitted lines for the conversion result. */
public class ContextEmitter(private val context: ConversionContext) : HloEmitter {
    private val _lines = mutableListOf<String>()

    /** Lines emitted through this emitter, in order. */
    public val lines: List<String> get() = _lines

    override fun newValue(): String = context.nextTempValue()

    override fun op(line: String) {
        _lines += line
        context.emitOperation(line)
    }
}

/**
 * The GELU lowering (#1342), one builder for both forms:
 *
 * - `approximate = true` (default): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` in StableHLO primitives.
 * - `approximate = false`: `0.5·x·(1 + erf(x/√2))` with `chlo.erf`.
 *
 * Constants are full-shape splats of the output type, as the other activation lowerings emit.
 */
public object GeluLowering {

    /** 1/√2 as the converter prints it. */
    public const val SQRT1_2: String = "0.70710678118654752"

    public fun emit(e: HloEmitter, x: String, type: String, approximate: Boolean): String =
        if (approximate) emitTanh(e, x, type) else emitExact(e, x, type)

    /** Tanh approximation: byte-for-byte the chain the converter emitted before `approximate` existed. */
    public fun emitTanh(e: HloEmitter, x: String, type: String): String {
        val halfConst = e.newValue()
        val oneConst = e.newValue()
        val coeff1Const = e.newValue()  // sqrt(2/π) ≈ 0.7978845608
        val coeff2Const = e.newValue()  // 0.044715
        val xSquared = e.newValue()
        val xCubed = e.newValue()
        val cubicTerm = e.newValue()
        val innerSum = e.newValue()
        val scaledSum = e.newValue()
        val tanhValue = e.newValue()
        val onePlusTanh = e.newValue()
        val halfX = e.newValue()
        val resultValue = e.newValue()

        e.op("$halfConst = stablehlo.constant dense<0.5> : $type")
        e.op("$oneConst = stablehlo.constant dense<1.0> : $type")
        e.op("$coeff1Const = stablehlo.constant dense<0.7978845608> : $type")
        e.op("$coeff2Const = stablehlo.constant dense<0.044715> : $type")
        e.op("$xSquared = stablehlo.multiply $x, $x : $type")
        e.op("$xCubed = stablehlo.multiply $xSquared, $x : $type")
        e.op("$cubicTerm = stablehlo.multiply $coeff2Const, $xCubed : $type")
        e.op("$innerSum = stablehlo.add $x, $cubicTerm : $type")
        e.op("$scaledSum = stablehlo.multiply $coeff1Const, $innerSum : $type")
        e.op("$tanhValue = stablehlo.tanh $scaledSum : $type")
        e.op("$onePlusTanh = stablehlo.add $oneConst, $tanhValue : $type")
        e.op("$halfX = stablehlo.multiply $halfConst, $x : $type")
        e.op("$resultValue = stablehlo.multiply $halfX, $onePlusTanh : $type")
        return resultValue
    }

    /** Exact form through `chlo.erf`. */
    public fun emitExact(e: HloEmitter, x: String, type: String): String {
        val halfConst = e.newValue()
        val oneConst = e.newValue()
        val sqrt12Const = e.newValue()
        val scaled = e.newValue()
        val erfValue = e.newValue()
        val onePlusErf = e.newValue()
        val halfX = e.newValue()
        val resultValue = e.newValue()

        e.op("$halfConst = stablehlo.constant dense<0.5> : $type")
        e.op("$oneConst = stablehlo.constant dense<1.0> : $type")
        e.op("$sqrt12Const = stablehlo.constant dense<$SQRT1_2> : $type")
        e.op("$scaled = stablehlo.multiply $x, $sqrt12Const : $type")
        e.op("$erfValue = chlo.erf $scaled : $type -> $type")
        e.op("$onePlusErf = stablehlo.add $oneConst, $erfValue : $type")
        e.op("$halfX = stablehlo.multiply $halfConst, $x : $type")
        e.op("$resultValue = stablehlo.multiply $halfX, $onePlusErf : $type")
        return resultValue
    }
}
