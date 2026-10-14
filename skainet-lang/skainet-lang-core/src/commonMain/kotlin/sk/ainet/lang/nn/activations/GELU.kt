package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.types.DType

/**
 * GELU activation module. [approximate] = `true` (default) applies the tanh form; `false`
 * applies the exact erf form — see [sk.ainet.lang.tensor.ops.TensorOps.gelu].
 */
public class GELU<T : DType, V>(
    override val name: String = "GELU",
    public val approximate: Boolean = true,
) : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) { with(input) { gelu(approximate) } }
}