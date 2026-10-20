package sk.ainet.lang.nn.metrics

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Classification accuracy metric.
 *
 * Computes the fraction of predictions that match the target labels.
 * Supports both hard targets (class indices) and soft targets (one-hot or probabilities).
 *
 * For predictions, the class with maximum value along [dim] is selected.
 * For soft targets, the class with maximum value is used as the ground truth.
 * For hard targets (Int32), the value directly represents the class index.
 *
 * @param dim The dimension along which to find the predicted class (default: -1, last dimension)
 * @param threshold Optional threshold for binary classification. If provided, predictions > threshold
 *                  are classified as class 1, otherwise class 0. Only applicable when dim size is 1 or 2.
 */
public class Accuracy(
    private val dim: Int = -1,
    private val threshold: Float? = null
) : Metric {

    override val name: String = "accuracy"

    private var correctCount: Long = 0
    private var totalCount: Long = 0

    override fun <T : DType, V> update(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext
    ) {
        ClassificationSupport.forEachPrediction(predictions, targets, dim, threshold) { predicted, target ->
            if (predicted == target) {
                correctCount++
            }
            totalCount++
        }
    }

    override fun compute(): Double {
        if (totalCount == 0L) return 0.0
        return correctCount.toDouble() / totalCount.toDouble()
    }

    override fun reset() {
        correctCount = 0
        totalCount = 0
    }
}

/**
 * Factory function for Accuracy metric.
 *
 * @param dim The dimension along which to find the predicted class (default: -1)
 * @param threshold Optional threshold for binary classification
 */
public fun accuracy(dim: Int = -1, threshold: Float? = null): Metric = Accuracy(dim, threshold)

/**
 * Binary accuracy metric with a threshold of 0.5.
 * Useful for binary classification problems with sigmoid outputs.
 */
public fun binaryAccuracy(threshold: Float = 0.5f): Metric = Accuracy(dim = -1, threshold = threshold)
