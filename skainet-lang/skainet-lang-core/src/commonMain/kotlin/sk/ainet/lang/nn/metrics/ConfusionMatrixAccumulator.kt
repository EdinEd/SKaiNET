package sk.ainet.lang.nn.metrics

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Immutable snapshot of per-class confusion-matrix counts.
 *
 * @property tp True positives for the class (`predicted == c && target == c`).
 * @property fp False positives for the class (`predicted == c && target != c`).
 * @property fn False negatives for the class (`predicted != c && target == c`).
 */
internal data class ClassCounts(
    val tp: Long,
    val fp: Long,
    val fn: Long
) {
    init {
        require(tp >= 0L && fp >= 0L && fn >= 0L) {
            "ClassCounts must be non-negative, got tp=$tp, fp=$fp, fn=$fn"
        }
    }
}

/**
 * Stateful per-class confusion-matrix accumulator shared by classification metrics
 * (`Precision`, `Recall`, `F1Score`).
 *
 * Walks batches via [ClassificationSupport.forEachPrediction] and maintains sparse
 * counters over the union of classes observed in `targets ∪ predictions`.
 *
 * @param dim Dimension along which class logits/probabilities lie (default: `-1`, last dimension).
 * @param threshold Optional threshold for binary classification (`classCount <= 2`).
 */
internal class ConfusionMatrixAccumulator(
    private val dim: Int = -1,
    private val threshold: Float? = null
) {
    private class MutableCounts(
        var tp: Long = 0L,
        var fp: Long = 0L,
        var fn: Long = 0L
    )

    private val counters = mutableMapOf<Int, MutableCounts>()

    internal fun <T : DType, V> update(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>
    ) {
        ClassificationSupport.forEachPrediction(predictions, targets, dim, threshold) { predicted, target ->
            if (predicted == target) {
                counters.getOrPut(target) { MutableCounts() }.tp++
            } else {
                counters.getOrPut(predicted) { MutableCounts() }.fp++
                counters.getOrPut(target) { MutableCounts() }.fn++
            }
        }
    }

    internal fun counts(): Map<Int, ClassCounts> =
        counters.mapValues { (_, c) -> ClassCounts(c.tp, c.fp, c.fn) }

    internal fun reset() {
        counters.clear()
    }
}

/**
 * Reduces per-class [counts] into a single scalar metric according to [averaging],
 * applying the `0.0` zero-division convention (never `NaN`) in one place.
 *
 * @param counts Per-class `(tp, fp, fn)` counts, typically from [ConfusionMatrixAccumulator.counts].
 * @param averaging Aggregation mode ([Averaging.BINARY], [Averaging.MACRO], or [Averaging.MICRO]).
 * @param perClass Metric formula mapping `(tp, fp, fn)` to a score. Any non-finite value
 *                 (`NaN` or `Infinity` from zero-division) is normalized to `0.0`.
 */
internal fun reduce(
    counts: Map<Int, ClassCounts>,
    averaging: Averaging,
    perClass: (tp: Long, fp: Long, fn: Long) -> Double
): Double = when (averaging) {
    Averaging.BINARY -> {
        require(counts.keys.all { it == 0 || it == 1 }) {
            "Averaging.BINARY requires binary class indices (0 or 1), got classes ${counts.keys.sorted()}"
        }
        val positive = counts[1]
        if (positive == null || positive.tp + positive.fp + positive.fn == 0L) {
            0.0
        } else {
            evalSafe(positive.tp, positive.fp, positive.fn, perClass)
        }
    }

    Averaging.MACRO -> {
        if (counts.isEmpty()) {
            0.0
        } else {
            var sum = 0.0
            for (c in counts.values) {
                sum += if (c.tp + c.fp + c.fn == 0L) 0.0 else evalSafe(c.tp, c.fp, c.fn, perClass)
            }
            sum / counts.size
        }
    }

    Averaging.MICRO -> {
        if (counts.isEmpty()) {
            0.0
        } else {
            var totalTp = 0L
            var totalFp = 0L
            var totalFn = 0L
            for (c in counts.values) {
                totalTp += c.tp
                totalFp += c.fp
                totalFn += c.fn
            }
            if (totalTp + totalFp + totalFn == 0L) {
                0.0
            } else {
                evalSafe(totalTp, totalFp, totalFn, perClass)
            }
        }
    }
}

private inline fun evalSafe(
    tp: Long,
    fp: Long,
    fn: Long,
    perClass: (tp: Long, fp: Long, fn: Long) -> Double
): Double {
    val value = perClass(tp, fp, fn)
    return if (!value.isFinite()) 0.0 else value
}
