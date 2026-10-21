package sk.ainet.data.media

import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Aspect-preserving fit of a source image into a square [targetSize] canvas, padded with a constant
 * color: the usual input preparation for single-shot detectors.
 *
 * The geometry is kept so predictions made on the canvas can be mapped back to the source image
 * with [toSource].
 *
 * Example:
 * ```kotlin
 * val lb = Letterbox.fit(width = 1280, height = 720, targetSize = 640)
 * val nchw = lb.fillNchw(rgb888)                  // FloatArray, 1 x 3 x 640 x 640, values in [0, 1]
 * val box = lb.toSource(BoundingBox(100f, 200f, 300f, 400f))
 * ```
 *
 * @property sourceWidth width of the source image in pixels
 * @property sourceHeight height of the source image in pixels
 * @property targetSize side of the square canvas in pixels
 * @property scale source → canvas scale factor
 * @property resizedWidth width of the scaled image on the canvas
 * @property resizedHeight height of the scaled image on the canvas
 * @property padLeft horizontal offset of the scaled image on the canvas
 * @property padTop vertical offset of the scaled image on the canvas
 */
public data class Letterbox(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val targetSize: Int,
    val scale: Float,
    val resizedWidth: Int,
    val resizedHeight: Int,
    val padLeft: Int,
    val padTop: Int
) {
    /** Maps a box in canvas pixels back to source pixels, clamped to the source image. */
    public fun toSource(box: BoundingBox): BoundingBox = BoundingBox(
        x1 = ((box.x1 - padLeft) / scale).coerceIn(0f, sourceWidth.toFloat()),
        y1 = ((box.y1 - padTop) / scale).coerceIn(0f, sourceHeight.toFloat()),
        x2 = ((box.x2 - padLeft) / scale).coerceIn(0f, sourceWidth.toFloat()),
        y2 = ((box.y2 - padTop) / scale).coerceIn(0f, sourceHeight.toFloat())
    )

    /**
     * Letterboxes packed RGB888 pixels (row-major, 3 bytes per pixel, no padding) into a planar
     * NCHW float array of size `3 * targetSize * targetSize`, values divided by 255. Resizing is
     * bilinear; the padding is [padValue] (0..255) on every channel.
     */
    public fun fillNchw(rgb888: ByteArray, padValue: Int = 114): FloatArray {
        require(rgb888.size == sourceWidth * sourceHeight * 3) {
            "Expected ${sourceWidth * sourceHeight * 3} bytes for ${sourceWidth}x$sourceHeight RGB888, got ${rgb888.size}"
        }
        val plane = targetSize * targetSize
        val out = FloatArray(3 * plane)
        val pad = padValue / 255f
        out.fill(pad)
        val sx = sourceWidth.toFloat() / resizedWidth
        val sy = sourceHeight.toFloat() / resizedHeight
        for (y in 0 until resizedHeight) {
            // pixel-center sampling, clamped to the source edge
            val fy = ((y + 0.5f) * sy - 0.5f).coerceIn(0f, (sourceHeight - 1).toFloat())
            val y0 = fy.toInt()
            val y1 = min(y0 + 1, sourceHeight - 1)
            val wy = fy - y0
            val row = (y + padTop) * targetSize
            for (x in 0 until resizedWidth) {
                val fx = ((x + 0.5f) * sx - 0.5f).coerceIn(0f, (sourceWidth - 1).toFloat())
                val x0 = fx.toInt()
                val x1 = min(x0 + 1, sourceWidth - 1)
                val wx = fx - x0
                val o = row + x + padLeft
                for (c in 0 until 3) {
                    val p00 = rgb888[(y0 * sourceWidth + x0) * 3 + c].toInt() and 0xFF
                    val p01 = rgb888[(y0 * sourceWidth + x1) * 3 + c].toInt() and 0xFF
                    val p10 = rgb888[(y1 * sourceWidth + x0) * 3 + c].toInt() and 0xFF
                    val p11 = rgb888[(y1 * sourceWidth + x1) * 3 + c].toInt() and 0xFF
                    val top = p00 + (p01 - p00) * wx
                    val bottom = p10 + (p11 - p10) * wx
                    out[c * plane + o] = (top + (bottom - top) * wy) / 255f
                }
            }
        }
        return out
    }

    public companion object {
        /** Computes the letterbox geometry for a [width] x [height] source on a [targetSize] canvas. */
        public fun fit(width: Int, height: Int, targetSize: Int): Letterbox {
            require(width > 0 && height > 0) { "Source size must be positive, got ${width}x$height" }
            require(targetSize > 0) { "Target size must be positive, got $targetSize" }
            val scale = min(targetSize.toFloat() / width, targetSize.toFloat() / height)
            val rw = (width * scale).roundToInt().coerceIn(1, targetSize)
            val rh = (height * scale).roundToInt().coerceIn(1, targetSize)
            return Letterbox(
                sourceWidth = width,
                sourceHeight = height,
                targetSize = targetSize,
                scale = scale,
                resizedWidth = rw,
                resizedHeight = rh,
                padLeft = (targetSize - rw) / 2,
                padTop = (targetSize - rh) / 2
            )
        }
    }
}
