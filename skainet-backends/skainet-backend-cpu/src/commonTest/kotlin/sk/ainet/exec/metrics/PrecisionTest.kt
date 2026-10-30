package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.Averaging
import sk.ainet.lang.nn.metrics.Precision
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.nn.metrics.binaryPrecision
import sk.ainet.lang.nn.metrics.computeForBatch
import sk.ainet.lang.nn.metrics.precision
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class PrecisionTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun intTensor(shape: Shape, data: IntArray): Tensor<Int32, Int> =
        ctx.fromIntArray(shape, Int32::class, data)

    @Test
    fun macro_multiclass_with_index_targets() {
        // Reference fixture from #1224:
        // Predictions argmax: [2, 0, 1, 2]
        // Targets: [2, 1, 1, 0]
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.2f, 0.3f, 0.5f   // predicts class 2
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

        val metric = precision()
        metric.update(preds, targets, ctx)

        // Class 0: TP=0, FP=1, FN=1 -> Prec = 0.0
        // Class 1: TP=1, FP=0, FN=1 -> Prec = 1.0
        // Class 2: TP=1, FP=1, FN=0 -> Prec = 0.5
        // Macro: (0.0 + 1.0 + 0.5) / 3 = 0.5
        assertEquals(0.5, metric.compute(), 1e-6)

        // Explicit Precision class with default Averaging.MACRO
        val explicitMetric = Precision()
        explicitMetric.update(preds, targets, ctx)
        assertEquals(0.5, explicitMetric.compute(), 1e-6)
    }

    @Test
    fun micro_multiclass_matches_accuracy_and_diverges_from_macro_on_imbalanced_batch() {
        // Reference fixture: pooled TP=2, FP=2, FN=2 -> Micro = 2 / (2 + 2) = 0.5
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f,  // predicts 1
                0.2f, 0.3f, 0.5f   // predicts 2
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

        val microMetric = precision(Averaging.MICRO)
        microMetric.update(preds, targets, ctx)

        val accMetric = accuracy()
        accMetric.update(preds, targets, ctx)

        assertEquals(0.5, microMetric.compute(), 1e-6)
        assertEquals(accMetric.compute(), microMetric.compute(), 1e-6)

        // Asymmetric batch proving MACRO and MICRO follow distinct reduction paths:
        // preds argmax: [0, 0, 0, 1]
        // targets:      [0, 1, 1, 1]
        // Class 0: TP=1, FP=2, FN=0 -> Prec = 1/3
        // Class 1: TP=1, FP=0, FN=2 -> Prec = 1.0
        // Macro = (1/3 + 1.0) / 2 = 2/3 (~0.666667)
        // Micro = 2 / 4 = 0.5 == accuracy
        val asymPreds = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.9f, 0.1f,  // predicts 0
                0.8f, 0.2f,  // predicts 0
                0.7f, 0.3f,  // predicts 0
                0.1f, 0.9f   // predicts 1
            )
        )
        val asymTargets = intTensor(Shape(4), intArrayOf(0, 1, 1, 1))

        val asymMacro = precision(Averaging.MACRO)
        asymMacro.update(asymPreds, asymTargets, ctx)
        assertEquals(2.0 / 3.0, asymMacro.compute(), 1e-6)

        val asymMicro = precision(Averaging.MICRO)
        asymMicro.update(asymPreds, asymTargets, ctx)
        assertEquals(0.5, asymMicro.compute(), 1e-6)

        val asymAcc = accuracy()
        asymAcc.update(asymPreds, asymTargets, ctx)
        assertEquals(asymAcc.compute(), asymMicro.compute(), 1e-6)
    }

    @Test
    fun soft_one_hot_targets_match_index_targets() {
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f,  // predicts 1
                0.2f, 0.3f, 0.5f   // predicts 2
            )
        )

        // Soft one-hot targets encoding [2, 1, 1, 0]
        val softTargets = tensor(
            Shape(4, 3),
            floatArrayOf(
                0f, 0f, 1f,  // class 2
                0f, 1f, 0f,  // class 1
                0f, 1f, 0f,  // class 1
                1f, 0f, 0f   // class 0
            )
        )

        val macroMetric = precision(Averaging.MACRO)
        macroMetric.update(preds, softTargets, ctx)
        assertEquals(0.5, macroMetric.compute(), 1e-6)

        val microMetric = precision(Averaging.MICRO)
        microMetric.update(preds, softTargets, ctx)
        assertEquals(0.5, microMetric.compute(), 1e-6)
    }

    @Test
    fun binary_precision_with_threshold_two_class_and_single_logit() {
        // Part A: 2-class logits with custom threshold = 0.7f
        val preds2Class = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.1f, 0.9f,   // logit[1]=0.9 > 0.7 -> predicts 1
                0.2f, 0.8f,   // logit[1]=0.8 > 0.7 -> predicts 1
                0.25f, 0.75f, // logit[1]=0.75 > 0.7 -> predicts 1
                0.35f, 0.65f  // logit[1]=0.65 <= 0.7 -> predicts 0
            )
        )
        val targets2Class = intTensor(Shape(4), intArrayOf(1, 1, 0, 0))

        // Under threshold 0.7f: predictions [1, 1, 1, 0] -> TP_1=2, FP_1=1 -> Prec = 2/3
        val customMetric = binaryPrecision(threshold = 0.7f)
        customMetric.update(preds2Class, targets2Class, ctx)
        assertEquals(2.0 / 3.0, customMetric.compute(), 1e-6)

        val explicitClassCustom = Precision(averaging = Averaging.BINARY, threshold = 0.7f)
        explicitClassCustom.update(preds2Class, targets2Class, ctx)
        assertEquals(2.0 / 3.0, explicitClassCustom.compute(), 1e-6)

        // Under default threshold 0.5f: sample 3 predicts 1 -> [1, 1, 1, 1] -> TP_1=2, FP_1=2 -> Prec = 0.5
        val defaultMetric = binaryPrecision()
        defaultMetric.update(preds2Class, targets2Class, ctx)
        assertEquals(0.5, defaultMetric.compute(), 1e-6)

        // Part B: Single-logit sigmoid (Shape(4, 1)) with threshold 0.5f
        val predsSingleLogit = tensor(
            Shape(4, 1),
            floatArrayOf(
                0.8f,  // > 0.5 -> predicts 1
                0.6f,  // > 0.5 -> predicts 1
                0.5f,  // <= 0.5 (exact tie) -> predicts 0
                0.2f   // <= 0.5 -> predicts 0
            )
        )
        val targetsSingleLogit = intTensor(Shape(4), intArrayOf(1, 0, 1, 0))

        // Predictions [1, 1, 0, 0], targets [1, 0, 1, 0] -> TP_1=1, FP_1=1 -> Prec = 0.5
        val singleLogitMetric = binaryPrecision()
        singleLogitMetric.update(predsSingleLogit, targetsSingleLogit, ctx)
        assertEquals(0.5, singleLogitMetric.compute(), 1e-6)
    }

    @Test
    fun accumulation_across_two_batches_equals_concatenated_batch_and_reset() {
        val preds1 = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f   // predicts 1
            )
        )
        val targets1 = intTensor(Shape(3), intArrayOf(2, 1, 1))

        val preds2 = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.1f, 0.8f, 0.1f,  // predicts 1
                0.8f, 0.1f, 0.1f   // predicts 0
            )
        )
        val targets2 = intTensor(Shape(3), intArrayOf(0, 1, 0))

        val predsConcat = tensor(
            Shape(6, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.1f, 0.2f, 0.7f,
                0.1f, 0.8f, 0.1f,
                0.8f, 0.1f, 0.1f
            )
        )
        val targetsConcat = intTensor(Shape(6), intArrayOf(2, 1, 1, 0, 1, 0))

        // Macro accumulation vs concatenated
        val macroAccum = precision(Averaging.MACRO)
        macroAccum.update(preds1, targets1, ctx)
        macroAccum.update(preds2, targets2, ctx)

        val macroConcat = precision(Averaging.MACRO)
        macroConcat.update(predsConcat, targetsConcat, ctx)

        assertEquals(2.0 / 3.0, macroAccum.compute(), 1e-6)
        assertEquals(macroConcat.compute(), macroAccum.compute(), 1e-6)

        // Micro accumulation vs concatenated
        val microAccum = precision(Averaging.MICRO)
        microAccum.update(preds1, targets1, ctx)
        microAccum.update(preds2, targets2, ctx)

        val microConcat = precision(Averaging.MICRO)
        microConcat.update(predsConcat, targetsConcat, ctx)

        assertEquals(4.0 / 6.0, microAccum.compute(), 1e-6)
        assertEquals(microConcat.compute(), microAccum.compute(), 1e-6)

        // Reset clears accumulator
        macroAccum.reset()
        assertEquals(0.0, macroAccum.compute(), 1e-6)

        microAccum.reset()
        assertEquals(0.0, microAccum.compute(), 1e-6)
    }

    @Test
    fun zero_division_and_fresh_metric_conventions() {
        // Fresh metric before any update returns 0.0
        assertEquals(0.0, precision(Averaging.MACRO).compute(), 1e-6)
        assertEquals(0.0, precision(Averaging.MICRO).compute(), 1e-6)
        assertEquals(0.0, binaryPrecision().compute(), 1e-6)

        // Class with zero predicted positives (TP=0, FP=0, FN>0):
        // preds: argmax [0, 0, 1]
        // targets:      [0, 1, 2]
        // Class 0: TP=1, FP=1 -> Prec = 0.5
        // Class 1: TP=0, FP=1 -> Prec = 0.0
        // Class 2: TP=0, FP=0, FN=1 -> Prec = 0.0 (never NaN!)
        // Macro: (0.5 + 0.0 + 0.0) / 3 = 1.0 / 6.0
        val predsZeroPred = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f   // predicts 1
            )
        )
        val targetsZeroPred = intTensor(Shape(3), intArrayOf(0, 1, 2))

        val zeroPredMetric = precision(Averaging.MACRO)
        zeroPredMetric.update(predsZeroPred, targetsZeroPred, ctx)
        assertEquals(1.0 / 6.0, zeroPredMetric.compute(), 1e-6)

        // Unseen class:
        // 1. In BINARY mode on an all-negative batch:
        val predsAllNeg = tensor(
            Shape(2, 2),
            floatArrayOf(
                0.9f, 0.1f,  // predicts 0
                0.8f, 0.2f   // predicts 0
            )
        )
        val targetsAllNeg = intTensor(Shape(2), intArrayOf(0, 0))
        val binMetric = binaryPrecision()
        binMetric.update(predsAllNeg, targetsAllNeg, ctx)
        assertEquals(0.0, binMetric.compute(), 1e-6)

        // 2. In MACRO mode on Shape(4, 4) where class 3 never appears in predictions or targets:
        val preds4Class = tensor(
            Shape(4, 4),
            floatArrayOf(
                0.1f, 0.2f, 0.7f, 0.0f,  // predicts 2
                0.8f, 0.1f, 0.1f, 0.0f,  // predicts 0
                0.1f, 0.8f, 0.1f, 0.0f,  // predicts 1
                0.2f, 0.3f, 0.5f, 0.0f   // predicts 2
            )
        )
        val targetsSeen = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))
        val macroUnseen = precision(Averaging.MACRO)
        macroUnseen.update(preds4Class, targetsSeen, ctx)
        // Macro average is over seen classes {0, 1, 2} only: (0.0 + 1.0 + 0.5) / 3 = 0.5
        assertEquals(0.5, macroUnseen.compute(), 1e-6)

        // All predictions wrong
        val allWrongPreds = tensor(
            Shape(3, 2),
            floatArrayOf(
                0.9f, 0.1f,  // predicts 0
                0.9f, 0.1f,  // predicts 0
                0.9f, 0.1f   // predicts 0
            )
        )
        val allWrongTargets = intTensor(Shape(3), intArrayOf(1, 1, 1))

        val wrongMacro = precision(Averaging.MACRO)
        wrongMacro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMacro.compute(), 1e-6)

        val wrongMicro = precision(Averaging.MICRO)
        wrongMicro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMicro.compute(), 1e-6)
    }

    @Test
    fun precision_metadata_and_custom_dimension() {
        // Metric name property
        val metric = precision()
        assertEquals("precision", metric.name)

        // computeForBatch helper
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.2f, 0.3f, 0.5f
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))
        val batchResult = precision().computeForBatch(preds, targets, ctx)
        assertEquals(0.5, batchResult, 1e-6)

        // Custom dimension: Shape(2, 3, 2) where dim=1 is the class dimension (3 classes)
        // Layout: b * 6 + c * 2 + s
        val preds3D = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                // b=0: c=0 -> [s0, s1], c=1 -> [s0, s1], c=2 -> [s0, s1]
                0.1f, 0.8f,
                0.2f, 0.1f,
                0.7f, 0.1f,
                // b=1: c=0 -> [s0, s1], c=1 -> [s0, s1], c=2 -> [s0, s1]
                0.1f, 0.2f,
                0.8f, 0.3f,
                0.1f, 0.5f
            )
        )
        // Targets shape: Shape(2, 2) -> (b0, s0)=2, (b0, s1)=1, (b1, s0)=1, (b1, s1)=0
        val targets2D = intTensor(Shape(2, 2), intArrayOf(2, 1, 1, 0))

        val customDimMetric = precision(dim = 1)
        customDimMetric.update(preds3D, targets2D, ctx)
        assertEquals(0.5, customDimMetric.compute(), 1e-6)
    }

    @Test
    fun binary_precision_error_on_multiclass() {
        // Averaging.BINARY requires classes in 0..1; observing class 2 must throw IllegalArgumentException on compute()
        val preds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f, // class 0
                0.1f, 0.8f, 0.1f, // class 1
                0.1f, 0.1f, 0.8f  // class 2
            )
        )
        val targets = intTensor(Shape(3), intArrayOf(0, 1, 2))

        val binaryMetric = Precision(averaging = Averaging.BINARY)
        binaryMetric.update(preds, targets, ctx)

        assertFailsWith<IllegalArgumentException> {
            binaryMetric.compute()
        }
    }

    @Test
    fun binary_precision_argmax_without_threshold() {
        // 2-class logits where threshold is null -> uses argmax
        val preds = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.8f, 0.2f,  // argmax 0
                0.3f, 0.7f,  // argmax 1
                0.4f, 0.6f,  // argmax 1
                0.9f, 0.1f   // argmax 0
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(0, 1, 0, 1))

        // predictions: [0, 1, 1, 0], targets: [0, 1, 0, 1]
        // class 1: sample 1 is TP, sample 2 is FP, sample 3 is FN
        // TP=1, FP=1 -> Prec = 1 / 2 = 0.5
        val metric = Precision(averaging = Averaging.BINARY)
        metric.update(preds, targets, ctx)
        assertEquals(0.5, metric.compute(), 1e-6)
    }

    @Test
    fun binary_precision_single_logit_with_custom_threshold() {
        val preds = tensor(
            Shape(4, 1),
            floatArrayOf(
                0.8f,   // > 0.6 -> predicts 1
                0.65f,  // > 0.6 -> predicts 1
                0.55f,  // <= 0.6 -> predicts 0
                0.2f    // <= 0.6 -> predicts 0
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(1, 0, 1, 0))

        // predictions: [1, 1, 0, 0], targets: [1, 0, 1, 0]
        // class 1: TP=1, FP=1 -> Prec = 0.5
        val metric = binaryPrecision(threshold = 0.6f)
        metric.update(preds, targets, ctx)
        assertEquals(0.5, metric.compute(), 1e-6)
    }

    @Test
    fun soft_targets_with_custom_dimension_3d() {
        // 3D tensor: Shape(2, 3, 2) where dim=1 is class dimension
        val preds3D = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                0.1f, 0.8f,
                0.2f, 0.1f,
                0.7f, 0.1f,

                0.1f, 0.2f,
                0.8f, 0.3f,
                0.1f, 0.5f
            )
        )
        // Soft one-hot targets matching index targets [2, 1, 1, 0]
        val softTargets3D = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                0f, 0f,  // b=0, c=0
                0f, 1f,  // b=0, c=1 (s1 is class 1)
                1f, 0f,  // b=0, c=2 (s0 is class 2)

                0f, 1f,  // b=1, c=0 (s1 is class 0)
                1f, 0f,  // b=1, c=1 (s0 is class 1)
                0f, 0f   // b=1, c=2
            )
        )

        val metric = precision(dim = 1)
        metric.update(preds3D, softTargets3D, ctx)
        assertEquals(0.5, metric.compute(), 1e-6)
    }

    @Test
    fun precision_1d_prediction_scalar_target() {
        // Single sample: 1D predictions Shape(3) with scalar target Shape()
        val preds = tensor(Shape(3), floatArrayOf(0.1f, 0.7f, 0.2f)) // predicts class 1
        val targets = intTensor(Shape(), intArrayOf(1))

        val metric = precision()
        metric.update(preds, targets, ctx)
        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun precision_empty_batch_returns_zero() {
        val preds = tensor(Shape(0, 3), floatArrayOf())
        val targets = intTensor(Shape(0), intArrayOf())

        val metric = precision()
        metric.update(preds, targets, ctx)
        assertEquals(0.0, metric.compute(), 1e-6)
    }
}

