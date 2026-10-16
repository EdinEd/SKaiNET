package sk.ainet.lang.tensor.ops

/**
 * Interpolation modes for 2D upsampling.
 *
 * Detector necks typically use nearest-neighbor. Bilinear is provided for parity
 * but may ignore `alignCorners` when not applicable.
 */
public enum class UpsampleMode {
    Nearest,
    Bilinear
}
