package sk.ainet.data.media

import kotlin.math.max
import kotlin.math.min

/**
 * Axis-aligned box in pixel coordinates, origin top-left, `x1 <= x2` and `y1 <= y2` for a
 * non-empty box.
 */
public data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    public val width: Float get() = max(0f, x2 - x1)
    public val height: Float get() = max(0f, y2 - y1)
    public val area: Float get() = width * height

    /** Intersection over union with [other]; 0 when the union is empty. */
    public fun iou(other: BoundingBox): Float {
        val iw = max(0f, min(x2, other.x2) - max(x1, other.x1))
        val ih = max(0f, min(y2, other.y2) - max(y1, other.y1))
        val inter = iw * ih
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    public companion object {
        /** Box from a center point and size. */
        public fun fromCenter(cx: Float, cy: Float, w: Float, h: Float): BoundingBox =
            BoundingBox(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
    }
}

/** A box with a class id and a confidence score, as produced by a detector before or after NMS. */
public data class ScoredBox(
    val box: BoundingBox,
    val classId: Int,
    val score: Float
)

/**
 * Greedy non-maximum suppression.
 *
 * Candidates are visited by descending score; a candidate is dropped when it overlaps an already
 * kept box by more than [iouThreshold]. With [classAware] (the default) only boxes of the same
 * class suppress each other. At most [maxDetections] boxes are kept. Ties keep input order.
 */
public fun nonMaxSuppression(
    candidates: List<ScoredBox>,
    iouThreshold: Float,
    maxDetections: Int = Int.MAX_VALUE,
    classAware: Boolean = true
): List<ScoredBox> {
    require(maxDetections >= 0) { "maxDetections must be >= 0, got $maxDetections" }
    val sorted = candidates.sortedByDescending { it.score }
    val kept = ArrayList<ScoredBox>()
    for (c in sorted) {
        if (kept.size >= maxDetections) break
        val suppressed = kept.any { k ->
            (!classAware || k.classId == c.classId) && k.box.iou(c.box) > iouThreshold
        }
        if (!suppressed) kept += c
    }
    return kept
}
