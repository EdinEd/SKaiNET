package sk.ainet.lang.nn.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class ConfusionMatrixAccumulatorTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun floatTensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        VoidOpsTensor(dataFactory.fromFloatArray(shape, FP32::class, values), FP32::class)

    private fun intTensor(shape: Shape, values: IntArray): Tensor<Int32, Int> =
        VoidOpsTensor(dataFactory.fromIntArray(shape, Int32::class, values), Int32::class)

    @Test
    fun class_counts_validation() {
        val valid = ClassCounts(tp = 10, fp = 5, fn = 2)
        assertEquals(10L, valid.tp)
        assertEquals(5L, valid.fp)
        assertEquals(2L, valid.fn)

        val zero = ClassCounts(tp = 0, fp = 0, fn = 0)
        assertEquals(0L, zero.tp)
        assertEquals(0L, zero.fp)
        assertEquals(0L, zero.fn)

        assertFailsWith<IllegalArgumentException> {
            ClassCounts(tp = -1, fp = 0, fn = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            ClassCounts(tp = 0, fp = -1, fn = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            ClassCounts(tp = 0, fp = 0, fn = -1)
        }
    }

    @Test
    fun accumulator_reference_fixture_index_targets() {
        // Predictions: 4 samples, 3 classes (argmax yields [2, 0, 1, 2])
        val preds = floatTensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )

        // Targets: [2, 1, 1, 0]
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

        val accumulator = ConfusionMatrixAccumulator()
        accumulator.update(preds, targets)

        val expected = mapOf(
            0 to ClassCounts(tp = 0L, fp = 1L, fn = 1L),
            1 to ClassCounts(tp = 1L, fp = 0L, fn = 1L),
            2 to ClassCounts(tp = 1L, fp = 1L, fn = 0L)
        )
        assertEquals(expected, accumulator.counts())
    }

    @Test
    fun accumulator_reference_fixture_soft_targets() {
        val preds = floatTensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )

        // One-hot targets for [2, 1, 1, 0]
        val targets = floatTensor(
            Shape(4, 3),
            floatArrayOf(
                0f, 0f, 1f,  // class 2
                0f, 1f, 0f,  // class 1
                0f, 1f, 0f,  // class 1
                1f, 0f, 0f   // class 0
            )
        )

        val accumulator = ConfusionMatrixAccumulator()
        accumulator.update(preds, targets)

        val expected = mapOf(
            0 to ClassCounts(tp = 0L, fp = 1L, fn = 1L),
            1 to ClassCounts(tp = 1L, fp = 0L, fn = 1L),
            2 to ClassCounts(tp = 1L, fp = 1L, fn = 0L)
        )
        assertEquals(expected, accumulator.counts())
    }

    @Test
    fun accumulator_excludes_unseen_classes_in_tensor_dim() {
        // Tensor dim has 4 classes (0..3), but class 3 never appears in preds or targets
        val preds = floatTensor(
            Shape(3, 4),
            floatArrayOf(
                0.7f, 0.1f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.7f, 0.1f, 0.1f,  // predicts 1
                0.1f, 0.1f, 0.7f, 0.1f   // predicts 2
            )
        )
        val targets = intTensor(Shape(3), intArrayOf(0, 1, 2))

        val accumulator = ConfusionMatrixAccumulator()
        accumulator.update(preds, targets)

        val counts = accumulator.counts()
        assertEquals(setOf(0, 1, 2), counts.keys)
        assertTrue(3 !in counts.keys)
    }

    @Test
    fun accumulator_binary_threshold_single_and_two_logit() {
        // 1. Single-logit binary output: Shape(4, 1) with threshold = 0.5f
        val singleLogitPreds = floatTensor(
            Shape(4, 1),
            floatArrayOf(
                0.3f,   // <= 0.5 -> class 0
                0.5f,   // <= 0.5 -> class 0 (exact tie)
                0.51f,  // > 0.5  -> class 1
                0.8f    // > 0.5  -> class 1
            )
        )
        val singleLogitTargets = intTensor(Shape(4), intArrayOf(0, 0, 1, 1))

        val accSingle = ConfusionMatrixAccumulator(threshold = 0.5f)
        accSingle.update(singleLogitPreds, singleLogitTargets)

        val expectedSingle = mapOf(
            0 to ClassCounts(tp = 2L, fp = 0L, fn = 0L),
            1 to ClassCounts(tp = 2L, fp = 0L, fn = 0L)
        )
        assertEquals(expectedSingle, accSingle.counts())

        // 2. Two-logit binary output: Shape(4, 2) with threshold = 0.5f
        val twoLogitPreds = floatTensor(
            Shape(4, 2),
            floatArrayOf(
                0.8f, 0.2f,  // logit 1 is 0.2 <= 0.5 -> class 0
                0.5f, 0.5f,  // logit 1 is 0.5 <= 0.5 -> class 0 (tie)
                0.3f, 0.7f,  // logit 1 is 0.7 > 0.5  -> class 1
                0.4f, 0.6f   // logit 1 is 0.6 > 0.5  -> class 1
            )
        )
        val twoLogitTargets = intTensor(Shape(4), intArrayOf(0, 1, 1, 0))

        val accTwo = ConfusionMatrixAccumulator(threshold = 0.5f)
        accTwo.update(twoLogitPreds, twoLogitTargets)

        // sample 0: pred 0, target 0 -> c0: tp+1
        // sample 1: pred 0, target 1 -> c0: fp+1, c1: fn+1
        // sample 2: pred 1, target 1 -> c1: tp+1
        // sample 3: pred 1, target 0 -> c1: fp+1, c0: fn+1
        val expectedTwo = mapOf(
            0 to ClassCounts(tp = 1L, fp = 1L, fn = 1L),
            1 to ClassCounts(tp = 1L, fp = 1L, fn = 1L)
        )
        assertEquals(expectedTwo, accTwo.counts())
    }

    @Test
    fun accumulator_custom_class_dim_and_multi_batch_and_reset() {
        // 3D tensor: Shape(2, 3, 2), dim = 1 (class dimension of size 3)
        // Targets shape: Shape(2, 2)
        val batch1Preds = floatTensor(
            Shape(2, 3, 2),
            floatArrayOf(
                // batch 0: seq 0, seq 1
                // class 0:
                0.1f, 0.8f,
                // class 1:
                0.7f, 0.1f,
                // class 2:
                0.2f, 0.1f,

                // batch 1: seq 0, seq 1
                // class 0:
                0.2f, 0.2f,
                // class 1:
                0.3f, 0.7f,
                // class 2:
                0.5f, 0.1f
            )
        )
        // Batch 0: seq 0 predicts class 1; seq 1 predicts class 0
        // Batch 1: seq 0 predicts class 2; seq 1 predicts class 1
        val batch1Targets = intTensor(
            Shape(2, 2),
            intArrayOf(
                1, 0,  // batch 0: both correct
                2, 0   // batch 1: seq 0 correct (2), seq 1 wrong (target 0, predicted 1)
            )
        )

        val acc = ConfusionMatrixAccumulator(dim = 1)
        acc.update(batch1Preds, batch1Targets)

        val snapshotBeforeBatch2 = acc.counts()
        assertEquals(
            mapOf(
                0 to ClassCounts(tp = 1L, fp = 0L, fn = 1L),
                1 to ClassCounts(tp = 1L, fp = 1L, fn = 0L),
                2 to ClassCounts(tp = 1L, fp = 0L, fn = 0L)
            ),
            snapshotBeforeBatch2
        )

        // Second batch: Shape(1, 3, 2), targets Shape(1, 2)
        val batch2Preds = floatTensor(
            Shape(1, 3, 2),
            floatArrayOf(
                0.9f, 0.1f,  // class 0
                0.05f, 0.8f, // class 1
                0.05f, 0.1f  // class 2
            )
        )
        // seq 0 predicts 0, seq 1 predicts 1
        val batch2Targets = intTensor(Shape(1, 2), intArrayOf(0, 1))
        acc.update(batch2Preds, batch2Targets)

        // Snapshot immutability: snapshotBeforeBatch2 should not have mutated
        assertEquals(1L, snapshotBeforeBatch2[0]?.tp)
        assertEquals(1L, snapshotBeforeBatch2[1]?.tp)

        val combinedCounts = acc.counts()
        assertEquals(2L, combinedCounts[0]?.tp)
        assertEquals(2L, combinedCounts[1]?.tp)
        assertEquals(1L, combinedCounts[2]?.tp)

        // Reset
        acc.reset()
        assertTrue(acc.counts().isEmpty())
    }

    @Test
    fun reduce_reference_fixture_macro_and_micro() {
        val counts = mapOf(
            0 to ClassCounts(0L, 1L, 1L),
            1 to ClassCounts(1L, 0L, 1L),
            2 to ClassCounts(1L, 1L, 0L)
        )

        val precision: (Long, Long, Long) -> Double = { tp, fp, _ -> tp.toDouble() / (tp + fp) }
        val recall: (Long, Long, Long) -> Double = { tp, _, fn -> tp.toDouble() / (tp + fn) }
        val f1: (Long, Long, Long) -> Double = { tp, fp, fn -> 2.0 * tp / (2.0 * tp + fp + fn) }

        // MACRO asserts
        assertEquals(0.5, reduce(counts, Averaging.MACRO, precision), 1e-9)
        assertEquals(0.5, reduce(counts, Averaging.MACRO, recall), 1e-9)
        assertEquals(4.0 / 9.0, reduce(counts, Averaging.MACRO, f1), 1e-9)
        assertNotEquals(0.5, reduce(counts, Averaging.MACRO, f1))

        // MICRO asserts
        assertEquals(0.5, reduce(counts, Averaging.MICRO, precision), 1e-9)
        assertEquals(0.5, reduce(counts, Averaging.MICRO, recall), 1e-9)
        assertEquals(0.5, reduce(counts, Averaging.MICRO, f1), 1e-9)
    }

    @Test
    fun reduce_binary_mode_and_validation() {
        val counts = mapOf(
            0 to ClassCounts(3L, 1L, 1L),
            1 to ClassCounts(2L, 1L, 1L)
        )
        val precision: (Long, Long, Long) -> Double = { tp, fp, _ -> tp.toDouble() / (tp + fp) }
        val recall: (Long, Long, Long) -> Double = { tp, _, fn -> tp.toDouble() / (tp + fn) }
        val f1: (Long, Long, Long) -> Double = { tp, fp, fn -> 2.0 * tp / (2.0 * tp + fp + fn) }

        // Binary mode reports positive class 1 only
        assertEquals(2.0 / 3.0, reduce(counts, Averaging.BINARY, precision), 1e-9)
        assertEquals(2.0 / 3.0, reduce(counts, Averaging.BINARY, recall), 1e-9)
        assertEquals(2.0 / 3.0, reduce(counts, Averaging.BINARY, f1), 1e-9)

        // All-negative batch (class 1 absent): returns 0.0
        val allNeg = mapOf(0 to ClassCounts(3L, 0L, 0L))
        assertEquals(0.0, reduce(allNeg, Averaging.BINARY, precision), 1e-9)

        // Class 1 present but has 0 total counts
        val zeroPos = mapOf(0 to ClassCounts(3L, 0L, 0L), 1 to ClassCounts(0L, 0L, 0L))
        assertEquals(0.0, reduce(zeroPos, Averaging.BINARY, precision), 1e-9)

        // Class index >= 2 throws IllegalArgumentException
        val invalidClass2 = mapOf(0 to ClassCounts(1L, 0L, 0L), 2 to ClassCounts(1L, 0L, 0L))
        assertFailsWith<IllegalArgumentException> {
            reduce(invalidClass2, Averaging.BINARY, precision)
        }

        // Negative class index throws IllegalArgumentException
        val invalidClassNeg = mapOf(-1 to ClassCounts(1L, 0L, 0L), 1 to ClassCounts(1L, 0L, 0L))
        assertFailsWith<IllegalArgumentException> {
            reduce(invalidClassNeg, Averaging.BINARY, precision)
        }
    }

    @Test
    fun reduce_zero_division_conventions() {
        val precision: (Long, Long, Long) -> Double = { tp, fp, _ -> tp.toDouble() / (tp + fp) }
        val recall: (Long, Long, Long) -> Double = { tp, _, fn -> tp.toDouble() / (tp + fn) }
        val f1: (Long, Long, Long) -> Double = { tp, fp, fn -> 2.0 * tp / (2.0 * tp + fp + fn) }

        // Empty counts map evaluates to 0.0 across all averaging strategies
        assertEquals(0.0, reduce(emptyMap(), Averaging.BINARY, precision), 1e-9)
        assertEquals(0.0, reduce(emptyMap(), Averaging.MACRO, precision), 1e-9)
        assertEquals(0.0, reduce(emptyMap(), Averaging.MICRO, precision), 1e-9)

        // Class with 0/0/0 counts contributes 0.0
        val countsWithZeroClass = mapOf(
            0 to ClassCounts(0L, 0L, 0L),
            1 to ClassCounts(1L, 0L, 0L)
        )
        // MACRO: (0.0 + 1.0) / 2 = 0.5
        assertEquals(0.5, reduce(countsWithZeroClass, Averaging.MACRO, precision), 1e-9)
        // MICRO: total tp=1, fp=0, fn=0 -> 1.0
        assertEquals(1.0, reduce(countsWithZeroClass, Averaging.MICRO, precision), 1e-9)

        val allZeroCounts = mapOf(
            0 to ClassCounts(0L, 0L, 0L),
            1 to ClassCounts(0L, 0L, 0L)
        )
        assertEquals(0.0, reduce(allZeroCounts, Averaging.BINARY, precision), 1e-9)
        assertEquals(0.0, reduce(allZeroCounts, Averaging.MACRO, precision), 1e-9)
        assertEquals(0.0, reduce(allZeroCounts, Averaging.MICRO, precision), 1e-9)

        // Class only in targets (tp=0, fp=0, fn=2) -> Precision tp / (tp+fp) = 0.0 / 0.0 = NaN -> 0.0
        val onlyInTargets = mapOf(1 to ClassCounts(0L, 0L, 2L))
        assertEquals(0.0, reduce(onlyInTargets, Averaging.BINARY, precision), 1e-9)
        assertEquals(0.0, reduce(onlyInTargets, Averaging.MACRO, precision), 1e-9)
        assertEquals(0.0, reduce(onlyInTargets, Averaging.MICRO, precision), 1e-9)

        // Class only in predictions (tp=0, fp=2, fn=0) -> Recall tp / (tp+fn) = 0.0 / 0.0 = NaN -> 0.0
        val onlyInPreds = mapOf(1 to ClassCounts(0L, 2L, 0L))
        assertEquals(0.0, reduce(onlyInPreds, Averaging.BINARY, recall), 1e-9)
        assertEquals(0.0, reduce(onlyInPreds, Averaging.MACRO, recall), 1e-9)
        assertEquals(0.0, reduce(onlyInPreds, Averaging.MICRO, recall), 1e-9)

        // F1 with 0 counts -> 0.0
        assertEquals(0.0, reduce(onlyInTargets, Averaging.BINARY, f1), 1e-9)
        assertEquals(0.0, reduce(onlyInTargets, Averaging.MACRO, f1), 1e-9)
        assertEquals(0.0, reduce(onlyInTargets, Averaging.MICRO, f1), 1e-9)

        // Non-finite results (Double.POSITIVE_INFINITY / Double.NEGATIVE_INFINITY) normalized to 0.0
        val infiniteFormula: (Long, Long, Long) -> Double = { tp, _, _ -> tp.toDouble() / 0.0 }
        val positiveClassMap = mapOf(1 to ClassCounts(1L, 0L, 0L))
        assertEquals(0.0, reduce(positiveClassMap, Averaging.BINARY, infiniteFormula), 1e-9)
        assertEquals(0.0, reduce(positiveClassMap, Averaging.MACRO, infiniteFormula), 1e-9)
        assertEquals(0.0, reduce(positiveClassMap, Averaging.MICRO, infiniteFormula), 1e-9)
    }

    @Test
    fun accumulator_1d_prediction_scalar_target() {
        // Single sample: 1D predictions Shape(3) with scalar target Shape()
        val preds = floatTensor(Shape(3), floatArrayOf(0.1f, 0.7f, 0.2f)) // predicts class 1
        val targets = intTensor(Shape(), intArrayOf(1))

        val acc = ConfusionMatrixAccumulator()
        acc.update(preds, targets)

        val expected = mapOf(1 to ClassCounts(tp = 1L, fp = 0L, fn = 0L))
        assertEquals(expected, acc.counts())
    }
}
