package sk.ainet.frontend

import sk.ainet.frontend.audio.AudioFrontEnd
import sk.ainet.frontend.image.ImageFrontEnd
import sk.ainet.frontend.text.TextFrontEnd

/**
 * The front ends a checkpoint resolved to, one slot per modality; `null` where the checkpoint
 * carries no evidence for that modality (FE-R-01 step 4).
 */
public class FrontEndSet(
    public val text: TextFrontEnd? = null,
    public val audio: AudioFrontEnd? = null,
    public val image: ImageFrontEnd? = null,
    public val composite: CompositeFrontEnd? = null,
) {
    public val isEmpty: Boolean get() = text == null && audio == null && image == null && composite == null

    public val specs: List<FrontEndSpec> get() = listOfNotNull(text?.spec, audio?.spec, image?.spec, composite?.spec)

    public fun copy(
        text: TextFrontEnd? = this.text,
        audio: AudioFrontEnd? = this.audio,
        image: ImageFrontEnd? = this.image,
        composite: CompositeFrontEnd? = this.composite,
    ): FrontEndSet = FrontEndSet(text, audio, image, composite)

    public companion object {
        public val EMPTY: FrontEndSet = FrontEndSet()
    }
}
