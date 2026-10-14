package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.lowering.LoweringSpec

/** `erf` lowering: `chlo.erf` inline, or composite `skainet.erf` (version 1) when kept fused. */
public object ErfLowering {
    public val spec: LoweringSpec = LoweringSpec(
        op = "erf",
        compositeName = "skainet.erf",
        attributes = emptyList(),
        version = 1,
    ) { e, x, type, _ -> emit(e, x, type) }

    public fun emit(e: HloEmitter, x: String, type: String): String {
        val resultValue = e.newValue()
        e.op("$resultValue = chlo.erf $x : $type -> $type")
        return resultValue
    }
}
