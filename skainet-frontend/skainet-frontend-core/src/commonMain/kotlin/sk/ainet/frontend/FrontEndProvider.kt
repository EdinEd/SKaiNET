package sk.ainet.frontend

import sk.ainet.context.ExecutionContext
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelIdentity

/**
 * One modality's algorithmic resolution (FE-R-01 step 2), implemented by each
 * `skainet-frontend-<modality>` module and composed into a [FrontEndResolver] by the caller —
 * the contract module cannot depend on the kernel modules.
 */
public interface FrontEndProvider {
    public val modality: Modality

    /** What this provider would build for [view], or `null` when the view carries no evidence for its modality. */
    public fun describe(view: CheckpointView, identity: ModelIdentity): FrontEndSpec?

    /**
     * Build the front end, or return `null` when the view carries no evidence for this modality.
     * Evidence that exists but is unsupported MUST throw [UnsupportedFrontEndException] (FE-R-05).
     *
     * @param ctx the execution context tensor-producing front ends need; text providers ignore it
     */
    public fun build(view: CheckpointView, identity: ModelIdentity, ctx: ExecutionContext?): FrontEnd<*, *>?
}

/**
 * Family policy for front ends (FE-R-01 step 3). A `ModelFamilyDescriptor` that also implements
 * this interface decorates the algorithmic base: special tokens, stop set, prompt renderer,
 * placeholder policy, a composite for multimodal checkpoints. Default: no change.
 */
public interface FamilyFrontEnds {
    public fun decorate(view: CheckpointView, identity: ModelIdentity, base: FrontEndSet, ctx: ExecutionContext?): FrontEndSet = base
}
