package sk.ainet.compile.hlo.lowering

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.converters.ContextEmitter
import sk.ainet.compile.hlo.converters.HloEmitter

/**
 * The formula of one fused-able op, written once through an [HloEmitter].
 *
 * The same builder produces the inline (decomposed) form and the body of the
 * `func.func private` a `stablehlo.composite` points at, so the two can never drift.
 * The builder sees only operand value names, their MLIR types, the result type and the node
 * parameters — exactly what a decomposition function has — so every shape it needs must be
 * readable from the types.
 */
public fun interface HloDecomposition {
    /** Emit the op for [operands] of MLIR [operandTypes] producing [resultType]; return the result value name. */
    public fun emit(e: HloEmitter, operands: List<String>, operandTypes: List<String>, resultType: String, params: Map<String, Any>): String

    public companion object {
        /** A decomposition of a unary op whose operand and result share one type. */
        public fun unary(block: (e: HloEmitter, operand: String, type: String, params: Map<String, Any>) -> String): HloDecomposition =
            HloDecomposition { e, operands, _, resultType, params ->
                require(operands.size == 1) { "unary decomposition got ${operands.size} operands" }
                block(e, operands[0], resultType, params)
            }
    }
}

/**
 * Lowering metadata of one op (#1342, #1343): how it is named as a `stablehlo.composite`, which
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
     * Lower one node: a `stablehlo.composite` plus its private decomposition function when
     * `context.granularity?.keepFused(op) == true`, else the decomposition inline.
     * Returns the result value name; the emitted lines are in [emitter].
     */
    public fun lower(
        context: ConversionContext,
        emitter: ContextEmitter,
        operands: List<String>,
        operandTypes: List<String>,
        resultType: String,
        params: Map<String, Any>,
    ): String =
        if (context.granularity?.keepFused(op) == true) {
            context.emitComposite(this, emitter, operands, operandTypes, resultType, params)
        } else {
            decomposition.emit(emitter, operands, operandTypes, resultType, params)
        }

    /** [lower] for a unary op whose operand and result share [type]. */
    public fun lower(
        context: ConversionContext,
        emitter: ContextEmitter,
        operand: String,
        type: String,
        params: Map<String, Any>,
    ): String = lower(context, emitter, listOf(operand), listOf(type), type, params)
}

/**
 * Emit `%r = stablehlo.composite "<name>" %a, %b, … { composite_attributes = {…}, decomposition = @fn, version = N : i32 } : (Ta, Tb, …) -> R`
 * and, once per (op, attributes, operand types), the `func.func private @fn(%arg0: Ta, %arg1: Tb, …) -> R`
 * holding [LoweringSpec.decomposition].
 *
 * The decomposition body takes its SSA names from this context's counter, so the module-wide
 * uniqueness rule holds across the entry function and every helper; `%argN` are the helper's own
 * arguments. Helpers go to the module trailer, after the entry function.
 */
public fun ConversionContext.emitComposite(
    spec: LoweringSpec,
    emitter: ContextEmitter,
    operands: List<String>,
    operandTypes: List<String>,
    resultType: String,
    params: Map<String, Any>,
): String {
    require(operands.isNotEmpty() && operands.size == operandTypes.size) { "composite emission needs one type per operand for ${spec.op}" }
    val attrs = spec.attributes.mapNotNull { key -> params[key]?.let { key to it } }
    val symbol = compositeSymbol(spec, attrs, operandTypes)

    if (claimSymbol(symbol)) {
        val body = BufferEmitter(this)
        val args = operandTypes.indices.map { "%arg$it" }
        val result = spec.decomposition.emit(body, args, operandTypes, resultType, params)
        val signature = operandTypes.indices.joinToString(", ") { "%arg$it: ${operandTypes[it]}" }
        emitModuleTrailer("func.func private @$symbol($signature) -> $resultType {")
        body.lines.forEach { emitModuleTrailer("  $it") }
        emitModuleTrailer("  return $result : $resultType")
        emitModuleTrailer("}")
    }

    val resultValue = nextTempValue()
    val attrText = if (attrs.isEmpty()) "" else "composite_attributes = {" +
        attrs.joinToString(", ") { (k, v) -> "$k = ${mlirAttribute(v)}" } + "}, "
    emitter.op(
        "$resultValue = stablehlo.composite \"${spec.compositeName}\" ${operands.joinToString(", ")} " +
            "{${attrText}decomposition = @$symbol, version = ${spec.version} : i32} : (${operandTypes.joinToString(", ")}) -> $resultType"
    )
    return resultValue
}

/** Unary form of [emitComposite]: operand and result share [type]. */
public fun ConversionContext.emitComposite(
    spec: LoweringSpec,
    emitter: ContextEmitter,
    operands: List<String>,
    type: String,
    params: Map<String, Any>,
): String = emitComposite(spec, emitter, operands, operands.map { type }, type, params)

/** `skainet_gelu_approximate_false_2x3xf32`: op, attribute values, then the sanitized operand types. */
internal fun compositeSymbol(spec: LoweringSpec, attrs: List<Pair<String, Any>>, operandTypes: List<String>): String {
    val attrPart = attrs.joinToString("") { (k, v) -> "_${k}_${v.toString().lowercase()}" }
    val typePart = operandTypes.joinToString("_") { type ->
        type.removePrefix("tensor<").removeSuffix(">").replace('?', 'd').replace('*', 's')
    }
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
