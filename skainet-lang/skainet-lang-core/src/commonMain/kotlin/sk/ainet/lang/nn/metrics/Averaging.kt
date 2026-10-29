package sk.ainet.lang.nn.metrics

/**
 * Reduction strategy used to aggregate per-class confusion-matrix statistics
 * into a single scalar classification metric (such as Precision, Recall, or F1Score).
 *
 * Zero-division convention: whenever a per-class or pooled denominator is zero
 * (for example, on an empty accumulator or when a class has no predicted or true samples),
 * the affected score evaluates to `0.0` (never `NaN`).
 */
public enum class Averaging {
    /**
     * Reports the metric for the positive class (class index `1`) only.
     *
     * Requires binary class indices (`0` and/or `1`, i.e., at most 2 classes);
     * throws [IllegalArgumentException] if any class index outside `0..1` has been observed.
     * Returns `0.0` when positive class `1` has no true or predicted samples.
     */
    BINARY,

    /**
     * Computes the metric independently for each observed class in `targets ∪ predictions`
     * and returns their unweighted arithmetic mean.
     *
     * Every observed class contributes equally regardless of class support. Classes that
     * never appear in either predictions or targets are excluded from the average; returns
     * `0.0` when no samples have been accumulated.
     */
    MACRO,

    /**
     * Pools true positives, false positives, and false negatives across all observed
     * classes first (`∑TP`, `∑FP`, `∑FN`), then applies the metric formula once.
     *
     * Every sample contributes equally; for single-label multi-class classification,
     * micro-averaged precision, recall, and F1 all equal overall classification accuracy.
     * Returns `0.0` when no samples have been accumulated.
     */
    MICRO
}
