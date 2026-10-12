package sk.ainet.frontend

import sk.ainet.frontend.audio.AudioFeatures
import sk.ainet.frontend.audio.AudioFrontEnd
import sk.ainet.frontend.audio.PcmAudio
import sk.ainet.frontend.image.ImageFeatures
import sk.ainet.frontend.image.ImageFrontEnd
import sk.ainet.frontend.image.RgbImage
import sk.ainet.frontend.text.TextFrontEnd
import sk.ainet.frontend.text.Vocabulary

/** One interleaved user turn: text and media in the order the user gave them. */
public sealed interface InputSegment {
    public data class Text(val text: String) : InputSegment
    public class Image(public val image: RgbImage) : InputSegment
    public class Audio(public val audio: PcmAudio) : InputSegment
}

public class MultimodalInput(public val segments: List<InputSegment>) {
    public companion object {
        public fun text(text: String): MultimodalInput = MultimodalInput(listOf(InputSegment.Text(text)))
    }
}

/** For each media item: the token positions its placeholder tokens occupy, in order. */
public data class PlaceholderSpan(val modality: Modality, val index: Int, val start: Int, val length: Int)

/**
 * Everything the model's forward needs, with the splice points the runtime uses to replace
 * placeholder embeddings (FE-C-12: the front end records *where* to splice, never *what*).
 */
public class ModelInputs(
    public val tokenIds: IntArray,
    public val attentionMask: IntArray?,
    public val images: List<ImageFeatures>,
    public val audio: List<AudioFeatures>,
    public val placeholders: List<PlaceholderSpan>,
)

/**
 * Family-supplied rule for how one image or audio clip appears in the token stream (FE-C-13), e.g.
 * Gemma 3: `<start_of_image>`, 256 × `<image_soft_token>`, `<end_of_image>`; Qwen2-VL: a count
 * that depends on the image grid.
 */
public interface PlaceholderPolicy {
    /** A short stable name, recorded in [FrontEndSpec.Composite.policy]. */
    public val name: String

    public fun placeholderTokens(modality: Modality, features: Any, vocabulary: Vocabulary): IntArray
}

/**
 * Composes the per-modality front ends and the policy. Resolved like any other front end; the
 * family descriptor builds it when the checkpoint carries more than one modality (FE-C-14: single
 * modality checkpoints never pay for it).
 */
public class CompositeFrontEnd(
    public val text: TextFrontEnd,
    public val image: ImageFrontEnd?,
    public val audio: AudioFrontEnd?,
    public val policy: PlaceholderPolicy,
) : FrontEnd<MultimodalInput, ModelInputs> {

    override val spec: FrontEndSpec = FrontEndSpec.Composite(
        parts = listOfNotNull(text.spec, image?.spec, audio?.spec),
        policy = policy.name,
    )

    override fun process(input: MultimodalInput): ModelInputs {
        val ids = ArrayList<Int>()
        val images = ArrayList<ImageFeatures>()
        val audios = ArrayList<AudioFeatures>()
        val spans = ArrayList<PlaceholderSpan>()
        val vocab = text.tokenizer.vocabulary
        for (segment in input.segments) {
            when (segment) {
                is InputSegment.Text -> for (id in text.tokenizer.encode(segment.text)) ids.add(id)
                is InputSegment.Image -> {
                    val fe = image ?: throw UnsupportedFrontEndException("checkpoint has no image front end; cannot place an image segment")
                    val features = fe.process(segment.image)
                    val placeholder = policy.placeholderTokens(Modality.IMAGE, features, vocab)
                    spans.add(PlaceholderSpan(Modality.IMAGE, images.size, ids.size, placeholder.size))
                    for (id in placeholder) ids.add(id)
                    images.add(features)
                }
                is InputSegment.Audio -> {
                    val fe = audio ?: throw UnsupportedFrontEndException("checkpoint has no audio front end; cannot place an audio segment")
                    val features = fe.openStream().use { s -> s.push(segment.audio); (s.poll().firstOrNull() ?: s.finish()) }
                        ?: throw UnsupportedFrontEndException("audio front end produced no features for a ${segment.audio.durationMillis} ms clip")
                    val placeholder = policy.placeholderTokens(Modality.AUDIO, features, vocab)
                    spans.add(PlaceholderSpan(Modality.AUDIO, audios.size, ids.size, placeholder.size))
                    for (id in placeholder) ids.add(id)
                    audios.add(features)
                }
            }
        }
        val tokenIds = IntArray(ids.size) { ids[it] }
        return ModelInputs(tokenIds, null, images, audios, spans)
    }
}
