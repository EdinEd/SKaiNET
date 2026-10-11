package sk.ainet.frontend.audio

import sk.ainet.frontend.StreamingFrontEnd
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/** Plain host buffer; interleaved if [channels] > 1. */
public class PcmAudio(
    public val samples: FloatArray,
    public val sampleRate: Int,
    public val channels: Int = 1,
) {
    init {
        require(sampleRate > 0) { "sampleRate must be positive: $sampleRate" }
        require(channels > 0) { "channels must be positive: $channels" }
    }

    /** Frames (samples per channel). */
    public val frames: Int get() = samples.size / channels

    /** Duration in milliseconds. */
    public val durationMillis: Float get() = frames * 1000f / sampleRate
}

/**
 * @property tensor `[1, nMels, frames]` for log-mel; `[1, T, dim]` for learned front ends
 * @property frameHopMillis lets a decoder map frames back to time
 * @property validFrames frames before padding
 */
public class AudioFeatures(
    public val tensor: Tensor<FP32, Float>,
    public val frameHopMillis: Float,
    public val validFrames: Int,
)

/**
 * Audio feature extraction contract (SKEEP-006 §Contract). Kernels (`LogMelFrontEnd`, DSP) are the
 * audio lane's `skainet-frontend-audio`; this interface is declared with the contract so that
 * `FrontEndSet.audio` is typed.
 */
public interface AudioFrontEnd : StreamingFrontEnd<PcmAudio, AudioFeatures> {
    public val expectedSampleRate: Int

    /** Window the model consumes in milliseconds (Whisper: 30 000); `null` for frame-streaming front ends. */
    public val chunkMillis: Int?
}
