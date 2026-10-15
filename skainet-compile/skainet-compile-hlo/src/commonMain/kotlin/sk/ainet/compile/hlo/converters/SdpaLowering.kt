package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.TypeMapper
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.hasDynamic
import sk.ainet.lang.tensor.ops.AttentionMask
import sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation

/**
 * The `scaledDotProductAttention` decomposition (#1343), written once through an [HloEmitter]:
 *
 *     scores = (Q · scale) · Kᵀ             (dot_general, contract head_dim)
 *     masked = scores + rules + mask       (iota/compare/select from the recorded rules, dense mask added)
 *     attn   = softmax(masked, axis = -1)  (max/sub/exp/sum/div, numerically stable)
 *     out    = attn · V                    (dot_general, contract key length)
 *
 * Everything the builder needs is in the operand MLIR types and the node parameters, so the same
 * code produces the inline form and the body of a `stablehlo.composite` decomposition function.
 *
 * **Mask rules** ([AttentionMask]): `causal`, a sliding `window` with `rightContext`, and key
 * padding from the `keyLengths` operand (`[batch]` integers) become one `i1` keep mask built with
 * `stablehlo.iota` over the query and key axes of the scores, `compare`, `and`, then
 * `select keep, 0, -1e30` added to the scores — no `[batch, 1, Sq, Sk]` constant enters the
 * module. The causal diagonal is offset by `Sk − Sq` like the CPU kernel, so a KV-cache decode
 * step attends to its whole cache. A dense `mask` operand is added on top, as the CPU kernel does.
 * Positional rules need static scores; a dynamic key length (`?`) is supported with a dense mask only.
 *
 * **Grouped-query attention** (SKEEP-005 phase 2): when K/V carry fewer heads than Q, Q is viewed
 * as `[b, nKV, nRep, Sq, hd]` and both `dot_general`s batch over `[b, nKV]`; K/V are never tiled.
 */
public object SdpaLowering {

    /** `tensor<1x2x?x8xf32>` → dims `[1, 2, -1, 8]` and element type `f32`; `tensor<f32>` → `[]`. */
    public fun parseTensorType(type: String): Pair<List<Int>, String> {
        require(type.startsWith("tensor<") && type.endsWith(">")) { "not a tensor type: $type" }
        val parts = type.removePrefix("tensor<").removeSuffix(">").split('x')
        val elem = parts.last()
        val dims = parts.dropLast(1).map { if (it == "?") TypeMapper.DYNAMIC_DIM else it.toInt() }
        return dims to elem
    }

    public fun renderType(dims: List<Int>, elem: String): String =
        if (dims.isEmpty()) "tensor<$elem>" else "tensor<${dims.joinToString("x") { Dim.render(it) }}x$elem>"

    /** The scale the formula uses: the recorded value, or `1/√headDim` when it was left at `0`. */
    public fun effectiveScale(scale: Float, headDim: Int): Float =
        if (scale != 0f) scale else (1.0f / kotlin.math.sqrt(headDim.toFloat()))

    /**
     * Emit the decomposition for [operands] (`q, k, v[, mask][, keyLengths]`, see
     * [ScaledDotProductAttentionOperation.operandLayout]) of MLIR [operandTypes] producing
     * [resultType]; returns the result value. [params] are the node parameters with `scale`
     * already resolved ([effectiveScale]). Shapes are validated by the converter beforehand.
     */
    public fun emit(
        e: HloEmitter,
        operands: List<String>,
        operandTypes: List<String>,
        resultType: String,
        params: Map<String, Any>,
    ): String {
        require(operands.size >= 3 && operands.size == operandTypes.size) { "SDPA needs q, k, v operands with types" }
        val layout = ScaledDotProductAttentionOperation.operandLayout(operands.size, params)
        val (qShape, elem) = parseTensorType(operandTypes[0])
        val (kShape, _) = parseTensorType(operandTypes[1])
        val qType = operandTypes[0]; val kType = operandTypes[1]; val vType = operandTypes[2]
        val rank = qShape.size
        val headDim = qShape[rank - 1]
        val keyLen = kShape[rank - 2]
        val outputType = resultType
        fun typeOf(shape: List<Int>) = renderType(shape, elem)

        val gqa = rank == 4 && kShape.size == 4 && kShape[1] != qShape[1]
        val qWork: List<Int> = if (gqa) listOf(qShape[0], kShape[1], qShape[1] / kShape[1], qShape[2], headDim) else qShape
        val qWorkType = if (gqa) typeOf(qWork) else qType
        val scoresShape = qWork.dropLast(1) + keyLen   // [.., Sq, Sk]  (GQA: [b, nKV, nRep, Sq, Sk])
        val scoresType = typeOf(scoresShape)
        val outWorkType = if (gqa) typeOf(qWork.dropLast(1) + headDim) else outputType

        val scaleVal = effectiveScale((params["scale"] as? Number)?.toFloat() ?: 0f, headDim)
        val causal = (params["causal"] as? Boolean) ?: false
        val window = (params["window"] as? Int) ?: 0
        val rightContext = (params["rightContext"] as? Int) ?: 0
        AttentionMask.validate(window, rightContext)

        val hasBatch = rank > 2
        val batchList = (0 until rank - 2).joinToString(", ")   // K/V batching dims; == Q's ([b, nKV]) under GQA too
        val batchClause = if (hasBatch) "batching_dims = [$batchList] x [$batchList], " else ""
        val contractQ = qWork.size - 1            // contract head_dim of Q …
        val contractK = rank - 1                  // … and K
        val sdAxis = scoresShape.size - 1         // softmax over key length
        val qAxis = scoresShape.size - 2          // query position in scores [.., Sq, Sk]
        val reducedShape = scoresShape.dropLast(1)
        val reducedType = if (reducedShape.isEmpty()) "tensor<$elem>" else typeOf(reducedShape)
        val bcastDims = (scoresShape.indices).filter { it != sdAxis }.joinToString(", ")
        val contractAttn = scoresShape.size - 1   // attn key-length axis
        val contractV = rank - 2                  // V key-length axis
        val scoresI32Type = "tensor<${scoresShape.joinToString("x") { Dim.render(it) }}xi32>"
        val scoresI1Type = "tensor<${scoresShape.joinToString("x") { Dim.render(it) }}xi1>"

        // Scale is applied to Q *before* the QK dot — scores·s == (q·s)@kᵀ exactly. This keeps the scale
        // constant at the STATIC query type (`[.., Sq, headDim]`) rather than a splat sized to the scores
        // shape `[.., Sq, Sk]`, whose Sk (key/cache length) may be dynamic (`?`) in KV-cache decode — a
        // dynamic-shape splat constant is invalid StableHLO. Also drops a full scores-sized dense constant
        // from static graphs.
        val scaleC = e.newValue(); val scaleB = e.newValue(); val qScaled = e.newValue(); val scores = e.newValue()
        e.op("$scaleC = stablehlo.constant dense<$scaleVal> : tensor<$elem>")
        e.op("$scaleB = stablehlo.broadcast_in_dim $scaleC, dims = [] : (tensor<$elem>) -> $qType")
        e.op("$qScaled = stablehlo.multiply ${operands[0]}, $scaleB : $qType")
        // GQA: expose the head groups as a batching dim of Q — a static, copy-free reshape.
        val qForDot = if (gqa) {
            val qg = e.newValue()
            e.op("$qg = stablehlo.reshape $qScaled : ($qType) -> $qWorkType")
            qg
        } else qScaled
        e.op("$scores = stablehlo.dot_general $qForDot, ${operands[1]}, ${batchClause}contracting_dims = [$contractQ] x [$contractK] : ($qWorkType, $kType) -> $scoresType")

        // When the scores shape is dynamic (the `?` key/cache dim of KV-cache decode), everything broadcast onto
        // it — the explicit mask and the softmax's reduced max/sum — must use `stablehlo.dynamic_broadcast_in_dim`:
        // a static `stablehlo.broadcast_in_dim` cannot target a dynamic shape. Its runtime `output_dimensions`
        // operand is built once, here, from the scores tensor via `get_dimension_size`.
        // Static graphs keep the original explicit `broadcast_in_dim` path (byte-for-byte unchanged).
        val dyn = scoresShape.hasDynamic()
        val shapeType = "tensor<${scoresShape.size}xi32>"
        val scoresShapeOperand: String = if (!dyn) "" else run {
            val parts = scoresShape.indices.map { d ->
                if (Dim.isStatic(scoresShape[d])) {
                    val c = e.newValue()
                    e.op("$c = stablehlo.constant dense<${scoresShape[d]}> : tensor<1xi32>")
                    c
                } else {
                    val gd = e.newValue(); val gr = e.newValue()
                    e.op("$gd = stablehlo.get_dimension_size $scores, dim = $d : ($scoresType) -> tensor<i32>")
                    e.op("$gr = stablehlo.reshape $gd : (tensor<i32>) -> tensor<1xi32>")
                    gr
                }
            }
            val sh = e.newValue()
            e.op("$sh = stablehlo.concatenate ${parts.joinToString(", ")}, dim = 0 : (${parts.joinToString(", ") { "tensor<1xi32>" }}) -> $shapeType")
            sh
        }

        var softmaxIn = scores   // scores are already scaled (scale folded into Q above)

        // Explicit additive mask — e.g. a sliding-window+causal mask the caller built, or a cross-attention
        // padding mask. Brought to the scores shape and added. Without this the masked layers run UNMASKED
        // (attend to future tokens) — correct only at position 0.
        //
        // Shape rules: trailing-aligned broadcast. Under GQA a rank-4 mask [b, M, Sq, Sk] keeps its batch on
        // scores dim 0 and skips the nRep axis (dim 2); M may be 1 (head-shared) or nKV (one row per group).
        // A per-head mask (M = H) is first VIEWED as [b, nKV, nRep, Sq, Sk] in Q's own h = kv * nRep + r order
        // (reshape, or dynamic_reshape when a dim is dynamic) — broadcasting H onto nKV is invalid IR.
        val maskOperand = layout.maskIndex?.let { operands[it] }
        if (maskOperand != null) {
            var maskType = operandTypes[layout.maskIndex!!]
            var maskShape: List<Int> = parseTensorType(maskType).first
            var maskVal = maskOperand
            var maskDims: List<Int> = if (gqa && maskShape.size == 4) listOf(0, 1, 3, 4) else {
                val offset = scoresShape.size - maskShape.size
                maskShape.indices.map { it + offset }
            }
            if (gqa && maskShape.size == 4 && maskShape[1] != 1 && maskShape[1] != kShape[1]) {
                val nKV = kShape[1]; val nH = qShape[1]
                require(maskShape[1] == nH) {
                    "SDPA grouped-query attention mask head dim must be 1, K/V heads ($nKV) or Q heads ($nH), got $maskShape"
                }
                val split = listOf(maskShape[0], nKV, nH / nKV, maskShape[2], maskShape[3])
                val splitType = typeOf(split)
                val r = e.newValue()
                if (!maskShape.hasDynamic()) {
                    e.op("$r = stablehlo.reshape $maskVal : ($maskType) -> $splitType")
                } else {
                    // split dim -> source mask dim (the two group dims are static by construction)
                    val source = listOf(0, -1, -1, 2, 3)
                    val parts = split.indices.map { d ->
                        if (Dim.isStatic(split[d])) {
                            val c = e.newValue()
                            e.op("$c = stablehlo.constant dense<${split[d]}> : tensor<1xi32>")
                            c
                        } else {
                            val gd = e.newValue(); val gr = e.newValue()
                            e.op("$gd = stablehlo.get_dimension_size $maskVal, dim = ${source[d]} : ($maskType) -> tensor<i32>")
                            e.op("$gr = stablehlo.reshape $gd : (tensor<i32>) -> tensor<1xi32>")
                            gr
                        }
                    }
                    val sh = e.newValue()
                    e.op("$sh = stablehlo.concatenate ${parts.joinToString(", ")}, dim = 0 : (${parts.joinToString(", ") { "tensor<1xi32>" }}) -> tensor<5xi32>")
                    e.op("$r = stablehlo.dynamic_reshape $maskVal, $sh : ($maskType, tensor<5xi32>) -> $splitType")
                }
                maskVal = r; maskShape = split; maskType = splitType; maskDims = split.indices.toList()
            }
            val maskBc = if (maskShape == scoresShape) {
                maskVal
            } else if (!dyn) {
                val mb = e.newValue()
                e.op("$mb = stablehlo.broadcast_in_dim $maskVal, dims = [${maskDims.joinToString(", ")}] : ($maskType) -> $scoresType")
                mb
            } else {
                // Dynamic target: dynamic_broadcast_in_dim, stating which operand dims expand. Without the
                // hints a dynamic operand dim mapped onto a dynamic result dim is ambiguous (1 -> N or N -> N),
                // and backends such as IREE refuse to lower it. A dynamic mask dim must match the scores dim
                // it maps to (the key length), so it is non-expanding. Generic op syntax: the hint attributes
                // are not part of every StableHLO version's pretty form.
                val expanding = mutableListOf<Int>(); val nonExpanding = mutableListOf<Int>()
                maskShape.indices.forEach { i ->
                    val o = maskShape[i]; val t = scoresShape[maskDims[i]]
                    when {
                        !Dim.isStatic(o) -> nonExpanding += i
                        Dim.isStatic(t) && o == t -> nonExpanding += i
                        Dim.isStatic(t) && o == 1 -> expanding += i
                        !Dim.isStatic(t) && o != 1 -> nonExpanding += i
                    }
                }
                fun arr(xs: List<Int>) = if (xs.isEmpty()) "array<i64>" else "array<i64: ${xs.joinToString(", ")}>"
                val mb = e.newValue()
                e.op("$mb = \"stablehlo.dynamic_broadcast_in_dim\"($maskVal, $scoresShapeOperand) <{broadcast_dimensions = ${arr(maskDims)}, " +
                    "known_expanding_dimensions = ${arr(expanding)}, known_nonexpanding_dimensions = ${arr(nonExpanding)}}> : " +
                    "($maskType, $shapeType) -> $scoresType")
                mb
            }
            val masked = e.newValue()
            e.op("$masked = stablehlo.add $softmaxIn, $maskBc : $scoresType")
            softmaxIn = masked
        }

        // Positional and padding rules (#1343) as one i1 keep mask over the scores shape. Each term is the
        // AttentionMask rule written with iota/compare; terms are and-ed, then masked-filled with a large
        // finite negative (not -inf): matches AttentionMask.MASKED / the dense masks callers build and
        // avoids a -inf splat in the select, which can trip downstream greedy constant-folding.
        val lengthsOperand = layout.lengthsIndex?.let { operands[it] }
        if (causal || window > 0 || lengthsOperand != null) {
            require(!dyn) { "SDPA positional/padding rules need a static scores shape, got $scoresShape — pass a dense mask for a dynamic key length" }
            val seqQ = scoresShape[qAxis]; val seqKV = scoresShape[sdAxis]
            val iotaQ = e.newValue(); val iotaK = e.newValue()
            e.op("$iotaQ = stablehlo.iota dim = $qAxis : $scoresI32Type")
            e.op("$iotaK = stablehlo.iota dim = $sdAxis : $scoresI32Type")
            // absQ = Sk - Sq + qi: the last query aligns with the last key (AttentionMask.queryPosition).
            val offset = seqKV - seqQ
            val qPos = if (offset == 0) iotaQ else shiftedI32(e, iotaQ, offset, scoresI32Type)
            val terms = mutableListOf<String>()
            if (causal) {
                val keep = e.newValue()
                e.op("$keep = stablehlo.compare GE, $qPos, $iotaK : ($scoresI32Type, $scoresI32Type) -> $scoresI1Type")
                terms += keep
            }
            if (window > 0) {
                val lower = shiftedI32(e, qPos, 1 - window, scoresI32Type)
                val keepLower = e.newValue()
                e.op("$keepLower = stablehlo.compare GE, $iotaK, $lower : ($scoresI32Type, $scoresI32Type) -> $scoresI1Type")
                terms += keepLower
                if (!causal) {
                    // The band's right edge; under `causal` the diagonal already bounds it (rightContext <= 0 there).
                    val upper = if (rightContext == 0) qPos else shiftedI32(e, qPos, rightContext, scoresI32Type)
                    val keepUpper = e.newValue()
                    e.op("$keepUpper = stablehlo.compare LE, $iotaK, $upper : ($scoresI32Type, $scoresI32Type) -> $scoresI1Type")
                    terms += keepUpper
                }
            }
            if (lengthsOperand != null) {
                val lengthsType = operandTypes[layout.lengthsIndex!!]
                val (lenShape, lenElem) = parseTensorType(lengthsType)
                require(lenShape.size == 1 && lenShape[0] == scoresShape[0]) { "SDPA keyLengths must be [batch=${scoresShape[0]}], got $lengthsType" }
                var lengths = lengthsOperand
                if (lenElem != "i32") {
                    val conv = e.newValue()
                    e.op("$conv = stablehlo.convert $lengths : ($lengthsType) -> tensor<${lenShape[0]}xi32>")
                    lengths = conv
                }
                val lenB = e.newValue(); val keepLen = e.newValue()
                e.op("$lenB = stablehlo.broadcast_in_dim $lengths, dims = [0] : (tensor<${lenShape[0]}xi32>) -> $scoresI32Type")
                e.op("$keepLen = stablehlo.compare LT, $iotaK, $lenB : ($scoresI32Type, $scoresI32Type) -> $scoresI1Type")
                terms += keepLen
            }
            var keep = terms[0]
            for (t in terms.drop(1)) {
                val both = e.newValue()
                e.op("$both = stablehlo.and $keep, $t : $scoresI1Type")
                keep = both
            }
            val zeros = e.newValue(); val ninf = e.newValue(); val maskAdd = e.newValue(); val masked = e.newValue()
            e.op("$zeros = stablehlo.constant dense<0.0> : $scoresType")
            e.op("$ninf = stablehlo.constant dense<-1.000000e+30> : $scoresType")
            e.op("$maskAdd = stablehlo.select $keep, $zeros, $ninf : $scoresI1Type, $scoresType")
            e.op("$masked = stablehlo.add $softmaxIn, $maskAdd : $scoresType")
            softmaxIn = masked
        }

        fun broadcastBack(src: String, dst: String) {
            if (dyn) {
                e.op("$dst = stablehlo.dynamic_broadcast_in_dim $src, $scoresShapeOperand, dims = [$bcastDims] : ($reducedType, $shapeType) -> $scoresType")
            } else {
                e.op("$dst = stablehlo.broadcast_in_dim $src, dims = [$bcastDims] : ($reducedType) -> $scoresType")
            }
        }
        val maxInit = e.newValue(); val maxV = e.newValue(); val maxB = e.newValue()
        val shifted = e.newValue(); val expV = e.newValue()
        val sumInit = e.newValue(); val sumV = e.newValue(); val sumB = e.newValue()
        val attn = e.newValue(); val out = e.newValue()
        e.op("$maxInit = stablehlo.constant dense<${TypeMapper().negInfBits(elem)}> : tensor<$elem>")
        e.op("$maxV = stablehlo.reduce($softmaxIn init: $maxInit) applies stablehlo.maximum across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType")
        broadcastBack(maxV, maxB)
        e.op("$shifted = stablehlo.subtract $softmaxIn, $maxB : $scoresType")
        e.op("$expV = stablehlo.exponential $shifted : $scoresType")
        e.op("$sumInit = stablehlo.constant dense<0.0> : tensor<$elem>")
        e.op("$sumV = stablehlo.reduce($expV init: $sumInit) applies stablehlo.add across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType")
        broadcastBack(sumV, sumB)
        e.op("$attn = stablehlo.divide $expV, $sumB : $scoresType")
        if (gqa) {
            val outG = e.newValue()
            e.op("$outG = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outWorkType")
            e.op("$out = stablehlo.reshape $outG : ($outWorkType) -> $outputType")
        } else {
            e.op("$out = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outputType")
        }
        return out
    }

    /** `value + delta` as an i32 splat add over the scores shape. */
    private fun shiftedI32(e: HloEmitter, value: String, delta: Int, i32Type: String): String {
        val c = e.newValue(); val r = e.newValue()
        e.op("$c = stablehlo.constant dense<$delta> : $i32Type")
        e.op("$r = stablehlo.add $value, $c : $i32Type")
        return r
    }
}
