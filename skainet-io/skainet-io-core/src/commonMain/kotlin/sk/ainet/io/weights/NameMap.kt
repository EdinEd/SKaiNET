@file:Suppress("DEPRECATION", "DEPRECATION_ERROR")

package sk.ainet.io.weights

import sk.ainet.lang.tensor.TensorId

/** Moved to `skainet-model-identity` (SKEEP-006); alias an error since 0.60, removed in 0.61. */
@Deprecated("Moved to sk.ainet.model.NameMap (SKEEP-006); the alias is an error since 0.60 and is removed in 0.61.", ReplaceWith("NameMap", "sk.ainet.model.NameMap"), level = DeprecationLevel.ERROR)
public typealias NameMap = sk.ainet.model.NameMap

/** Moved to `skainet-model-identity` (SKEEP-006); alias an error since 0.60, removed in 0.61. */
@Deprecated("Moved to sk.ainet.model.RoleTableNameMap (SKEEP-006); the alias is an error since 0.60 and is removed in 0.61.", ReplaceWith("RoleTableNameMap", "sk.ainet.model.RoleTableNameMap"), level = DeprecationLevel.ERROR)
public typealias RoleTableNameMap = sk.ainet.model.RoleTableNameMap

/**
 * This map as a legacy [WeightNameResolver]: `(modulePath, paramName)` in the slash form
 * (`model/layers[3]/attn`, `q_proj.weight`) → checkpoint name. An extension since SKEEP-006
 * because [sk.ainet.model.NameMap] sits below the module that owns [WeightNameResolver].
 */
public fun sk.ainet.model.NameMap.asWeightNameResolver(): WeightNameResolver = WeightNameResolver { modulePath, paramName ->
    val segments = (modulePath.takeIf { it.isNotEmpty() }?.split("/") ?: emptyList())
    val parts = paramName.split(".")
    // a dotted paramName ("q_proj.weight") contributes its leading parts to the module path
    toCheckpointName(TensorId(segments + parts.dropLast(1), parts.last()))
}

/** SAM-style constructor for [WeightNameResolver]. */
public fun WeightNameResolver(block: (modulePath: String, paramName: String) -> String?): WeightNameResolver =
    object : WeightNameResolver {
        override fun resolve(modulePath: String, paramName: String): String? = block(modulePath, paramName)
    }
