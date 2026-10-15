package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.hasDynamic
import sk.ainet.lang.graph.GraphNode

/**
 * Converts `scaledDotProductAttention` to the standard StableHLO attention subgraph through
 * [SdpaLowering], the one builder of the formula (#1343):
 *
 *     scores = Q · Kᵀ                      (dot_general, contract head_dim)
 *     scaled = scores * scale              (scale = arg, or 1/sqrt(head_dim))
 *     attn   = softmax(scaled + mask)      (max/sub/exp/sum/div, numerically stable)
 *     out    = attn · V                    (dot_general, contract key length)
 *
 * Q/K/V are batched `[.., S, D]`; batching dims are every leading dim except the
 * last two. The softmax decomposition mirrors ActivationOperationsConverter.
 *
 * SDPA is a core `TensorOps` op (KSP-generated), so its converter lives here in
 * core alongside dot_general/softmax — the transformer modules just decompose to it.
 *
 * Emission goes through `SdpaLowering.spec`: a `stablehlo.composite "skainet.sdpa"` with its private
 * decomposition function when the target's `OpGranularityPolicy` keeps `scaleddotproductattention`
 * fused, the decomposition inline otherwise (#1343 part 2).
 *
 * Masking: the node's positional rules (`causal`, `window`/`rightContext`) and the `keyLengths`
 * operand become an `iota`/`compare`/`select` keep mask, and a dense `mask` operand is added on
 * top — the same rules the CPU kernel applies, defined once in `AttentionMask`. The rules need a
 * static scores shape; a dynamic key length (`?`) is exported with a dense mask only.
 *
 * Grouped-query attention (SKEEP-005 phase 2, "structure at compile time"): when K/V
 * carry fewer heads than Q (`[b, nKV, Sk, hd]` vs `[b, H, Sq, hd]`, `nKV | H`), Q is
 * reshaped to `[b, nKV, nRep, Sq, hd]` and both `dot_general`s batch over `[b, nKV]`
 * with `nRep` a free axis — K/V are never broadcast or concatenated. The group
 * structure is exactly what the compiler backend tiles over; the core count is not
 * written anywhere (it is a run-time property of the device).
 */
public class AttentionOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        "scaledDotProductAttention", "scaleddotproductattention", "sdpa"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 3) {
            return ConversionResult.Failure(
                "scaledDotProductAttention requires q, k, v (>=3 operands), got ${operands.size}",
                "Unsupported SDPA arity for node ${node.id}"
            )
        }

        val qShape = node.inputs.getOrNull(0)?.shape
            ?: return ConversionResult.Failure("SDPA requires a known query shape", "Missing query shape for ${node.id}")
        val kShape = node.inputs.getOrNull(1)?.shape ?: qShape
        val vShape = node.inputs.getOrNull(2)?.shape ?: qShape
        val rank = qShape.size
        if (rank < 2) {
            return ConversionResult.Failure("SDPA query must have rank >= 2", "Bad query rank for ${node.id}")
        }

        val outSpec = node.outputs.firstOrNull()
        val mapper = context.getTypeMapper()
        val elem = outSpec?.let { mapper.mapDType(it.dtype) } ?: "f32"
        fun typeOf(shape: List<Int>): String = SdpaLowering.renderType(shape, elem)

        // Operand MLIR types: the registered value type, else the node's input spec. The builder reads
        // every shape it needs from these, as it must inside a composite decomposition function.
        val operandTypes = operands.mapIndexed { i, operand ->
            context.getValueType(operand)
                ?: node.inputs.getOrNull(i)?.let { spec ->
                    val shape = spec.shape
                    if (i < 3 && shape != null) typeOf(shape) else mapper.mapTensorType(spec)
                }
                ?: when (i) { 0 -> typeOf(qShape); 1 -> typeOf(kShape); 2 -> typeOf(vShape)
                    else -> return ConversionResult.Failure("SDPA operand $i has no known type", "Missing operand type for ${node.id}") }
        }
        val headDim = qShape[rank - 1]
        val outputType = outSpec?.let { mapper.mapTensorType(it) } ?: typeOf(qShape.dropLast(1) + headDim)

        // Grouped-query attention: K/V heads divide Q heads → work on Q as [b, nKV, nRep, Sq, hd].
        val gqa = rank == 4 && kShape.size == 4 && kShape[1] != qShape[1]
        if (gqa) {
            val nKV = kShape[1]; val nH = qShape[1]
            if (!Dim.isStatic(nKV) || !Dim.isStatic(nH) || nKV <= 0 || nH % nKV != 0) {
                return ConversionResult.Failure(
                    "SDPA grouped-query attention needs static head counts with K/V heads dividing Q heads, got Q=$nH K/V=$nKV",
                    "Unsupported GQA head counts for ${node.id}",
                )
            }
            if (qShape.hasDynamic()) {
                return ConversionResult.Failure(
                    "SDPA grouped-query attention needs a static query shape (reshape to [b, nKV, nRep, Sq, hd]), got $qShape",
                    "Dynamic query shape under GQA for ${node.id}",
                )
            }
        }

        val params = node.operation.parameters
        val layout = sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation.operandLayout(operands.size, params)
        val causal = (params["causal"] as? Boolean) ?: false
        val window = (params["window"] as? Int) ?: 0
        val rightContext = (params["rightContext"] as? Int) ?: 0
        val scoresShape = qShape.dropLast(1) + kShape[rank - 2]
        if ((causal || window > 0 || layout.lengthsIndex != null) && scoresShape.hasDynamic()) {
            return ConversionResult.Failure(
                "SDPA positional/padding rules (causal/window/keyLengths) need a static scores shape, got $scoresShape — pass a dense mask for a dynamic key length",
                "Structured mask on a dynamic shape for ${node.id}",
            )
        }
        val maskIndex = layout.maskIndex
        if (maskIndex != null && gqa) {
            val maskShape = node.inputs.getOrNull(maskIndex)?.shape
            if (maskShape != null && maskShape.size == 4 && maskShape[1] != 1 && maskShape[1] != kShape[1] && maskShape[1] != qShape[1]) {
                return ConversionResult.Failure(
                    "SDPA grouped-query attention mask head dim must be 1, K/V heads (${kShape[1]}) or Q heads (${qShape[1]}), got $maskShape",
                    "Unsupported GQA mask shape for ${node.id}",
                )
            }
        }
        val scaleParam = (params["scale"] as? Number)?.toFloat() ?: 0f
        val lowered = params + mapOf(
            "scale" to SdpaLowering.effectiveScale(scaleParam, headDim),
            "causal" to causal, "window" to window, "rightContext" to rightContext,
        )
        return try {
            // Composite `skainet.sdpa` when the target keeps the op fused, else the decomposition inline.
            val emitter = ContextEmitter(context)
            val out = SdpaLowering.spec.lower(context, emitter, operands, operandTypes, outputType, lowered)
            ConversionResult.Success(outputValueName = out, emittedOperations = emitter.lines)
        } catch (e: IllegalArgumentException) {
            ConversionResult.Failure(e.message ?: "SDPA lowering failed", "SDPA lowering failed for ${node.id}")
        }
    }
}
