package sk.ainet.frontend.image

import sk.ainet.frontend.FrontEnd
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/** Platform-free packed RGB8, row-major; the only image input type the front ends understand. */
public class RgbImage(
    public val width: Int,
    public val height: Int,
    public val rgb: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "image must be non-empty: ${width}x$height" }
        require(rgb.size == width * height * 3) { "rgb has ${rgb.size} bytes, expected ${width * height * 3}" }
    }
}

/** Enough to map boxes and masks back to the source image. */
public data class ImageGeometry(
    val srcWidth: Int,
    val srcHeight: Int,
    val scale: Float,
    val padLeft: Int,
    val padTop: Int,
)

/** @property tensor NCHW, batch 1 */
public class ImageFeatures(
    public val tensor: Tensor<FP32, Float>,
    public val geometry: ImageGeometry,
)

/**
 * Image preprocessing contract (SKEEP-006 §Contract). Kernels (`LetterboxFrontEnd`,
 * `ResizeCropNormalizeFrontEnd`) are the image lane's `skainet-frontend-image`.
 */
public interface ImageFrontEnd : FrontEnd<RgbImage, ImageFeatures> {
    /** `(width, height)` the model expects. */
    public val inputSize: Pair<Int, Int>
}
