package sk.ainet.frontend

import sk.ainet.context.ExecutionContext
import sk.ainet.frontend.audio.AudioFrontEnd
import sk.ainet.frontend.image.ImageFrontEnd
import sk.ainet.frontend.text.TextFrontEnd
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFamilyRegistry
import sk.ainet.model.ModelIdentity

/**
 * The single entry point model loaders, Java callers and cartridge facades use to obtain a
 * checkpoint's input front ends (SKEEP-006 §Resolution).
 *
 * Resolution order is fixed (FE-R-01): (1) identity from [registry]; (2) algorithmic base from
 * side files and metadata through the [providers]; (3) the family descriptor decorates the base
 * when it implements [FamilyFrontEnds]; (4) modalities with nothing found are `null` in the set,
 * and the `resolveX` methods throw an [UnsupportedFrontEndException] naming the side files and
 * the registry's families.
 *
 * @param providers one per modality module, e.g. `TextFrontEnds.provider` from `skainet-frontend-text`
 * @param registry the family registry; downstream repositories pass `builtIn + theirFamilies`
 * @param ctx required only by tensor-producing providers (audio, image); text ignores it
 */
public class FrontEndResolver(
    private val providers: List<FrontEndProvider>,
    private val registry: ModelFamilyRegistry = ModelFamilyRegistry.builtIn,
    private val ctx: ExecutionContext? = null,
) {
    public constructor(provider: FrontEndProvider, registry: ModelFamilyRegistry = ModelFamilyRegistry.builtIn, ctx: ExecutionContext? = null) :
        this(listOf(provider), registry, ctx)

    init {
        val dup = providers.groupBy { it.modality }.filterValues { it.size > 1 }.keys
        require(dup.isEmpty()) { "more than one provider for $dup" }
    }

    /** The registry's identity for [view]. */
    public fun identify(view: CheckpointView): ModelIdentity = registry.identify(view)

    /** Every modality the checkpoint carries, decorated by its family. */
    public fun resolve(view: CheckpointView): FrontEndSet {
        val identity = registry.identify(view)
        var set = FrontEndSet.EMPTY
        for (p in providers) {
            val fe = p.build(view, identity, ctx) ?: continue
            set = when (p.modality) {
                Modality.TEXT -> set.copy(text = fe as? TextFrontEnd ?: TextFrontEnd(fe as Tokenizer))
                Modality.AUDIO -> set.copy(audio = fe as AudioFrontEnd)
                Modality.IMAGE -> set.copy(image = fe as ImageFrontEnd)
                Modality.COMPOSITE -> set.copy(composite = fe as CompositeFrontEnd)
            }
        }
        val family = registry.descriptorFor(identity)
        if (family is FamilyFrontEnds) set = family.decorate(view, identity, set, ctx)
        return set
    }

    public fun resolveText(view: CheckpointView): TextFrontEnd =
        resolve(view).text ?: throw unsupported(view, Modality.TEXT)

    public fun resolveAudio(view: CheckpointView): AudioFrontEnd =
        resolve(view).audio ?: throw unsupported(view, Modality.AUDIO)

    public fun resolveImage(view: CheckpointView): ImageFrontEnd =
        resolve(view).image ?: throw unsupported(view, Modality.IMAGE)

    /** Dry run: what would be built, from which side file (before family decoration). */
    public fun describe(view: CheckpointView): List<FrontEndSpec> {
        val identity = registry.identify(view)
        return providers.mapNotNull { it.describe(view, identity) }
    }

    private fun unsupported(view: CheckpointView, modality: Modality): UnsupportedFrontEndException {
        val identity = registry.identify(view)
        return UnsupportedFrontEndException(
            "no ${modality.name.lowercase()} front end for ${identity.label} (${view.format}): " +
                "side files ${view.listSideFiles().sorted()}, providers ${providers.map { it.modality }}, " +
                "registry families ${registry.families}",
        )
    }
}
