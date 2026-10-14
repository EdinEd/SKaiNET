package sk.ainet.compile.hlo.lowering

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.converters.ContextEmitter
import sk.ainet.compile.hlo.converters.HloEmitter

/**
 * The formula of one fused-able op, written once through an [HloEmitter].
 *
 * The same builder produces the inline (decomposed) form and the body of the
 * `func.func private` a `stablehlo.composite` points at, so the two can never drift.
 */
public fun interface HloDecomposition {
    /** Emit the op for [operand] of MLIR [type] and return the result value name. */
    public fun emit(e: HloEmitter, operand: String, type: String, params: Map<String, Any>): String
}

/**
 * Lowering metadata of one op (#1342): how it is named as a `stablehlo.composite`, which
 * node parameters become `composite_attributes`, the composite `version`, and its
 * [decomposition]. Whether a target sees the composite or the decomposition is decided
 * per op by [sk.ainet.compile.target.OpGranularityPolicy] in [lower].
 */
public data class LoweringSpec(
    /** Graph op name (lower-case), the key [sk.ainet.compile.target.OpGranularityPolicy.keepFused] receives. */
    val op: String,
    /** Namespaced composite name, e.g. `skainet.gelu` (the StableHLO spec requires a namespace). */
    val compositeName: String,
    /** Node parameter keys copied into `composite_attributes`, in this order. */
    val attributes: List<String>,
    /** Composite version; bump when the decomposition's semantics change. */
    val version: Int,
    val decomposition: HloDecomposition,
) {
    init {
        require('.' in compositeName) { "composite name must be namespaced: $compositeName" }
    }

    /**
     * Lower one unary node: a `stablehlo.composite` plus its private decomposition function
     * when `context.granularity?.keepFused(op) == true`, else the decomposition inline.
     * Returns the result value name; the emitted lines are in [emitter].
     */
    public fun lower(
        context: ConversionContext,
        emitter: ContextEmitter,
        operand: String,
        type: String,
        params: Map<String, Any>,
    ): String =
        if (context.granularity?.keepFused(op) == true) {
            context.emitComposite(this, emitter, listOf(operand), type, params)
        } else {
            decomposition.emit(emitter, operand, type, params)
        }
}

/**
 * Emit `%r = stablehlo.composite "<name>" %x { composite_attributes = {…}, decomposition = @fn, version = N : i32 } : (T) -> T`
 * and, once per (op, attributes, type), the `func.func private @fn` holding [LoweringSpec.decomposition].
 *
 * The decomposition body takes its SSA names from this context's counter, so the module-wide
 * uniqueness rule holds across the entry function and every helper; `%arg0` is the helper's own
 * argument. Helpers go to the module trailer, after the entry function.
 */
public fun ConversionContext.emitComposite(
    spec: LoweringSpec,
    emitter: ContextEmitter,
    operands: List<String>,
    type: String,
    params: Map<String, Any>,
): String {
    require(operands.size == 1) { "composite emission currently supports unary ops, got ${operands.size} operands for ${spec.op}" }
    val attrs = spec.attributes.mapNotNull { key -> params[key]?.let { key to it } }
    val symbol = compositeSymbol(spec, attrs, type)

    if (claimSymbol(symbol)) {
        val body = BufferEmitter(this)
        val result = spec.decomposition.emit(body, "%arg0", type, params)
        emitModuleTrailer("func.func private @$symbol(%arg0: $type) -> $type {")
        body.lines.forEach { emitModuleTrailer("  $it") }
        emitModuleTrailer("  return $result : $type")
        emitModuleTrailer("}")
    }

    val resultValue = nextTempValue()
    val attrText = if (attrs.isEmpty()) "" else "composite_attributes = {" +
        attrs.joinToString(", ") { (k, v) -> "$k = ${mlirAttribute(v)}" } + "}, "
    emitter.op(
        "$resultValue = stablehlo.composite \"${spec.compositeName}\" ${operands[0]} " +
            "{${attrText}decomposition = @$symbol, version = ${spec.version} : i32} : ($type) -> $type"
    )
    return resultValue
}

/** `skainet_gelu_approximate_false_f32_2x3`: op, attribute values, then the sanitized type. */
internal fun compositeSymbol(spec: LoweringSpec, attrs: List<Pair<String, Any>>, type: String): String {
    val attrPart = attrs.joinToString("") { (k, v) -> "_${k}_${v.toString().lowercase()}" }
    val typePart = type.removePrefix("tensor<").removeSuffix(">")
        .replace('?', 'd').replace('*', 's').replace("x", "x")
    val sanitized = (spec.op + attrPart + "_" + typePart).replace(Regex("[^A-Za-z0-9_]"), "_")
    return "skainet_$sanitized"
}

/** Render a node parameter as an MLIR attribute value. */
internal fun mlirAttribute(value: Any): String = when (value) {
    is Boolean -> value.toString()
    is Int, is Long, is Short, is Byte -> "$value : i32"
    is Float -> "${value.toDouble()} : f32"
    is Double -> "$value : f64"
    else -> "\"$value\""
}

/** [HloEmitter] that buffers lines but still draws SSA names from the shared context counter. */
private class BufferEmitter(private val context: ConversionContext) : HloEmitter {
    val lines = mutableListOf<String>()
    override fun newValue(): String = context.nextTempValue()
    override fun op(line: String) { lines += line }
}
