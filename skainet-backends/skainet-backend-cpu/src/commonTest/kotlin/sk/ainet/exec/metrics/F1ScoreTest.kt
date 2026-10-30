package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.Averaging
import sk.ainet.lang.nn.metrics.F1Score
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.nn.metrics.binaryF1Score
import sk.ainet.lang.nn.metrics.computeForBatch
import sk.ainet.lang.nn.metrics.f1Score
import sk.ainet.lang.nn.metrics.precision
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class F1ScoreTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun intTensor(shape: Shape, data: IntArray): Tensor<Int32, Int> =
        ctx.fromIntArray(shape, Int32::class, data)

    private fun referencePredictions(): Tensor<FP32, Float> = tensor(
        Shape(4, 3),
        floatArrayOf(
            0.1f, 0.2f, 0.7f,  // predicts class 2
            0.8f, 0.1f, 0.1f,  // predicts class 0
            0.1f, 0.8f, 0.1f,  // predicts class 1
            0.2f, 0.3f, 0.5f   // predicts class 2
        )
    )

    private fun referenceTargets(): Tensor<Int32, Int> =
        intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

    @Test
    fun macro_multiclass_is_mean_of_per_class_f1_not_f1_of_macro_precision_and_recall() {
        // Reference fixture from #1224:
        // predictions = [2, 0, 1, 2], targets = [2, 1, 1, 0]
        //
        // Counts as (TP, FP, FN):
        // Class 0: (0, 1, 1) -> F1 = 0 / 2 = 0.0
        // Class 1: (1, 0, 1) -> F1 = 2 / 3
        // Class 2: (1, 1, 0) -> F1 = 2 / 3
        // Macro F1 = (0 + 2/3 + 2/3) / 3 = 4/9
        //
        // Macro precision and macro recall are both 0.5, so F1-of-means would be 0.5.
        val preds = referencePredictions()
        val targets = referenceTargets()

        val metricDefault = f1Score()
        metricDefault.update(preds, targets, ctx)
        val macroF1 = metricDefault.compute()
        assertEquals(4.0 / 9.0, macroF1, 1e-6)

        val macroPrecision = precision(Averaging.MACRO)
        macroPrecision.update(preds, targets, ctx)
        val macroPrecisionValue = macroPrecision.compute()
        val macroRecallValue = 0.5
        val f1OfMacroMeans =
            2.0 * macroPrecisionValue * macroRecallValue / (macroPrecisionValue + macroRecallValue)
        assertEquals(0.5, macroPrecisionValue, 1e-6)
        assertEquals(0.5, f1OfMacroMeans, 1e-6)
        assertNotEquals(f1OfMacroMeans, macroF1)

        val metricExplicit = F1Score()
        metricExplicit.update(preds, targets, ctx)
        assertEquals(4.0 / 9.0, metricExplicit.compute(), 1e-6)
    }

    @Test
    fun micro_multiclass_matches_precision_accuracy_and_imbalanced_reference() {
        // Reference fixture pooled counts: TP=2, FP=2, FN=2.
        // Micro F1 = 2*2 / (2*2 + 2 + 2) = 0.5.
        // In single-label classification, micro precision == micro recall == accuracy.
        val preds = referencePredictions()
        val targets = referenceTargets()

        val microF1 = f1Score(Averaging.MICRO)
        microF1.update(preds, targets, ctx)

        val microPrecision = precision(Averaging.MICRO)
        microPrecision.update(preds, targets, ctx)

        val accuracyMetric = accuracy()
        accuracyMetric.update(preds, targets, ctx)

        assertEquals(0.5, microF1.compute(), 1e-6)
        assertEquals(microPrecision.compute(), microF1.compute(), 1e-6)
        assertEquals(accuracyMetric.compute(), microF1.compute(), 1e-6)

        // Asymmetric fixture:
        // predictions = [0, 0, 0, 1], targets = [0, 1, 2, 2]
        // Class 0: (TP=1, FP=2, FN=0) -> F1 = 2 / 4 = 0.5
        // Class 1: (TP=0, FP=1, FN=1) -> F1 = 0.0
        // Class 2: (TP=0, FP=0, FN=2) -> F1 = 0.0
        // Macro F1 = 0.5 / 3 = 1/6
        // Pooled: TP=1, FP=3, FN=3 -> Micro F1 = 2 / 8 = 0.25 == Accuracy
        val asymPreds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f   // predicts 1
            )
        )
        val asymTargets = intTensor(Shape(4), intArrayOf(0, 1, 2, 2))

        val asymMacro = f1Score(Averaging.MACRO)
        asymMacro.update(asymPreds, asymTargets, ctx)
        assertEquals(1.0 / 6.0, asymMacro.compute(), 1e-6)

        val asymMicro = f1Score(Averaging.MICRO)
        asymMicro.update(asymPreds, asymTargets, ctx)
        val asymAccuracy = accuracy()
        asymAccuracy.update(asymPreds, asymTargets, ctx)
        assertEquals(0.25, asymMicro.compute(), 1e-6)
        assertEquals(asymAccuracy.compute(), asymMicro.compute(), 1e-6)
    }

    @Test
    fun soft_one_hot_targets_match_index_targets() {
        // FP32 one-hot targets encode the reference targets [2, 1, 1, 0].
        val softTargets = tensor(
            Shape(4, 3),
            floatArrayOf(
                0f, 0f, 1f,  // class 2
                0f, 1f, 0f,  // class 1
                0f, 1f, 0f,  // class 1
                1f, 0f, 0f   // class 0
            )
        )

        val macro = f1Score(Averaging.MACRO)
        macro.update(referencePredictions(), softTargets, ctx)
        assertEquals(4.0 / 9.0, macro.compute(), 1e-6)

        val micro = f1Score(Averaging.MICRO)
        micro.update(referencePredictions(), softTargets, ctx)
        assertEquals(0.5, micro.compute(), 1e-6)
    }

    @Test
    fun binary_f1_with_threshold_two_class_and_single_logit_tie() {
        val twoClassPreds = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.1f, 0.9f,    // positive logit 0.9
                0.35f, 0.65f,  // positive logit 0.65
                0.6f, 0.4f,    // positive logit 0.4
                0.8f, 0.2f     // positive logit 0.2
            )
        )
        val twoClassTargets = intTensor(Shape(4), intArrayOf(1, 1, 1, 0))

        // Threshold 0.5 -> predictions [1, 1, 0, 0]
        // Positive class: TP=2, FP=0, FN=1 -> F1 = 4 / 5 = 0.8
        val defaultThreshold = binaryF1Score()
        defaultThreshold.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(0.8, defaultThreshold.compute(), 1e-6)

        // Threshold 0.7 -> predictions [1, 0, 0, 0]
        // Positive class: TP=1, FP=0, FN=2 -> F1 = 2 / 4 = 0.5
        val customFactory = binaryF1Score(threshold = 0.7f)
        customFactory.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(0.5, customFactory.compute(), 1e-6)

        val customDirect = F1Score(averaging = Averaging.BINARY, threshold = 0.7f)
        customDirect.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(0.5, customDirect.compute(), 1e-6)

        // Single sigmoid logit; exact tie 0.5 must map to class 0.
        // Predictions [1, 0, 0, 1], targets [1, 1, 1, 0]
        // Positive class: TP=1, FP=1, FN=2 -> F1 = 2 / 5 = 0.4
        val singleLogitPreds = tensor(Shape(4, 1), floatArrayOf(0.8f, 0.5f, 0.3f, 0.9f))
        val singleLogitTargets = intTensor(Shape(4), intArrayOf(1, 1, 1, 0))
        val singleLogit = binaryF1Score(threshold = 0.5f)
        singleLogit.update(singleLogitPreds, singleLogitTargets, ctx)
        assertEquals(0.4, singleLogit.compute(), 1e-6)
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
        val targets1 = intTensor(Shape(3), intArrayOf(2, 1, 2))

        val preds2 = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.1f, 0.8f, 0.1f,  // predicts 1
                0.8f, 0.1f, 0.1f   // predicts 0
            )
        )
        val targets2 = intTensor(Shape(3), intArrayOf(0, 1, 2))

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
        val targetsConcat = intTensor(Shape(6), intArrayOf(2, 1, 2, 0, 1, 2))

        // Combined predictions = [2, 0, 1, 2, 1, 0]
        // Combined targets     = [2, 1, 2, 0, 1, 2]
        // Class 0: TP=0, FP=2, FN=1 -> F1 = 0.0
        // Class 1: TP=1, FP=1, FN=1 -> F1 = 2 / 4 = 0.5
        // Class 2: TP=1, FP=1, FN=2 -> F1 = 2 / 5 = 0.4
        // Macro F1 = (0 + 0.5 + 0.4) / 3 = 0.3
        val macroAccumulated = f1Score(Averaging.MACRO)
        macroAccumulated.update(preds1, targets1, ctx)
        macroAccumulated.update(preds2, targets2, ctx)
        val macroConcat = f1Score(Averaging.MACRO)
        macroConcat.update(predsConcat, targetsConcat, ctx)
        assertEquals(0.3, macroAccumulated.compute(), 1e-6)
        assertEquals(macroConcat.compute(), macroAccumulated.compute(), 1e-6)

        // Pooled TP=2, FP=4, FN=4 -> Micro F1 = 4 / 12 = 1/3
        val microAccumulated = f1Score(Averaging.MICRO)
        microAccumulated.update(preds1, targets1, ctx)
        microAccumulated.update(preds2, targets2, ctx)
        val microConcat = f1Score(Averaging.MICRO)
        microConcat.update(predsConcat, targetsConcat, ctx)
        assertEquals(1.0 / 3.0, microAccumulated.compute(), 1e-6)
        assertEquals(microConcat.compute(), microAccumulated.compute(), 1e-6)

        macroAccumulated.reset()
        microAccumulated.reset()
        assertEquals(0.0, macroAccumulated.compute(), 1e-6)
        assertEquals(0.0, microAccumulated.compute(), 1e-6)
    }

    @Test
    fun zero_division_unseen_classes_all_wrong_and_fresh_metric_conventions() {
        assertEquals(0.0, f1Score(Averaging.MACRO).compute(), 1e-6)
        assertEquals(0.0, f1Score(Averaging.MICRO).compute(), 1e-6)
        assertEquals(0.0, binaryF1Score().compute(), 1e-6)

        // Predictions = [0, 1, 2], targets = [0, 0, 1]
        // Class 0: TP=1, FP=0, FN=1 -> 2/3
        // Class 1: TP=0, FP=1, FN=1 -> 0.0
        // Class 2: TP=0, FP=1, FN=0 -> 0.0
        // Macro F1 = 2/9
        val noGroundTruthPreds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,  // 0
                0.1f, 0.8f, 0.1f,  // 1
                0.1f, 0.1f, 0.8f   // 2
            )
        )
        val noGroundTruthTargets = intTensor(Shape(3), intArrayOf(0, 0, 1))
        val noGroundTruth = f1Score(Averaging.MACRO)
        noGroundTruth.update(noGroundTruthPreds, noGroundTruthTargets, ctx)
        val noGroundTruthScore = noGroundTruth.compute()
        assertEquals(2.0 / 9.0, noGroundTruthScore, 1e-6)
        assertTrue(noGroundTruthScore.isFinite())

        // Binary positive class has TP+FP+FN=0 -> 0.0.
        val allNegativePreds = tensor(Shape(2, 2), floatArrayOf(0.9f, 0.1f, 0.9f, 0.1f))
        val allNegativeTargets = intTensor(Shape(2), intArrayOf(0, 0))
        val allNegative = binaryF1Score()
        allNegative.update(allNegativePreds, allNegativeTargets, ctx)
        assertEquals(0.0, allNegative.compute(), 1e-6)

        // Class 3 exists in the tensor width but never appears in predictions or targets.
        // Macro is over observed classes {0, 1, 2}, so the result remains 4/9.
        val unseenClassPreds = tensor(
            Shape(4, 4),
            floatArrayOf(
                0.1f, 0.2f, 0.7f, 0.0f,  // 2
                0.8f, 0.1f, 0.1f, 0.0f,  // 0
                0.1f, 0.8f, 0.1f, 0.0f,  // 1
                0.2f, 0.3f, 0.5f, 0.0f   // 2
            )
        )
        val unseenClass = f1Score()
        unseenClass.update(unseenClassPreds, referenceTargets(), ctx)
        assertEquals(4.0 / 9.0, unseenClass.compute(), 1e-6)

        // Every prediction wrong -> every observed class has TP=0.
        val allWrongPreds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.8f, 0.1f,  // 1
                0.1f, 0.1f, 0.8f,  // 2
                0.8f, 0.1f, 0.1f   // 0
            )
        )
        val allWrongTargets = intTensor(Shape(3), intArrayOf(0, 1, 2))
        val wrongMacro = f1Score(Averaging.MACRO)
        wrongMacro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMacro.compute(), 1e-6)

        val wrongMicro = f1Score(Averaging.MICRO)
        wrongMicro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMicro.compute(), 1e-6)
    }

    @Test
    fun metadata_custom_dimension_and_compute_for_batch_reset_round_trip() {
        val metric = f1Score(Averaging.MACRO, dim = 1)
        assertEquals("f1", metric.name)

        // Shape(2, 3, 2), class dimension = 1.
        // Pairs (predicted, target): (1, 1), (0, 0), (2, 2), (1, 0)
        // Class 0: TP=1, FP=0, FN=1 -> 2/3
        // Class 1: TP=1, FP=1, FN=0 -> 2/3
        // Class 2: TP=1, FP=0, FN=0 -> 1.0
        // Macro F1 = 7/9; pooled TP=3, FP=1, FN=1 -> Micro F1 = 6/8 = 0.75
        val preds = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                0.1f, 0.7f,
                0.8f, 0.2f,
                0.1f, 0.1f,
                0.1f, 0.1f,
                0.2f, 0.8f,
                0.7f, 0.1f
            )
        )
        val targets = intTensor(Shape(2, 2), intArrayOf(1, 0, 2, 0))

        metric.update(preds, targets, ctx)
        assertEquals(7.0 / 9.0, metric.compute(), 1e-6)

        val micro = f1Score(Averaging.MICRO, dim = 1)
        micro.update(preds, targets, ctx)
        assertEquals(0.75, micro.compute(), 1e-6)

        // Prime state with a different batch, then computeForBatch must reset it first.
        val roundTrip = f1Score()
        roundTrip.update(
            tensor(Shape(3, 3), floatArrayOf(0.1f, 0.8f, 0.1f, 0.1f, 0.1f, 0.8f, 0.8f, 0.1f, 0.1f)),
            intTensor(Shape(3), intArrayOf(0, 1, 2)),
            ctx
        )
        assertEquals(0.0, roundTrip.compute(), 1e-6)
        assertEquals(4.0 / 9.0, roundTrip.computeForBatch(referencePredictions(), referenceTargets(), ctx), 1e-6)
        assertEquals(4.0 / 9.0, roundTrip.compute(), 1e-6)
    }

    @Test
    fun binary_f1_rejects_multiclass_counts() {
        val metric = F1Score(averaging = Averaging.BINARY)
        val preds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.1f, 0.1f, 0.8f
            )
        )
        metric.update(preds, intTensor(Shape(3), intArrayOf(0, 1, 2)), ctx)

        assertFailsWith<IllegalArgumentException> {
            metric.compute()
        }
    }

    @Test
    fun binary_f1_argmax_without_threshold_and_single_logit_custom_threshold() {
        // Argmax predictions = [0, 1, 0, 0], targets = [0, 1, 1, 1]
        // Positive class: TP=1, FP=0, FN=2 -> F1 = 2 / 4 = 0.5
        val argmaxMetric = F1Score(averaging = Averaging.BINARY, dim = -1, threshold = null)
        argmaxMetric.update(
            tensor(Shape(4, 2), floatArrayOf(0.8f, 0.2f, 0.3f, 0.7f, 0.9f, 0.1f, 0.6f, 0.4f)),
            intTensor(Shape(4), intArrayOf(0, 1, 1, 1)),
            ctx
        )
        assertEquals(0.5, argmaxMetric.compute(), 1e-6)

        // Threshold 0.6 -> predictions [1, 1, 0, 0], targets [1, 1, 1, 0]
        // Positive class: TP=2, FP=0, FN=1 -> F1 = 0.8
        val singleLogit = binaryF1Score(threshold = 0.6f)
        singleLogit.update(
            tensor(Shape(4, 1), floatArrayOf(0.8f, 0.65f, 0.55f, 0.2f)),
            intTensor(Shape(4), intArrayOf(1, 1, 1, 0)),
            ctx
        )
        assertEquals(0.8, singleLogit.compute(), 1e-6)
    }

    @Test
    fun one_dimensional_prediction_with_scalar_target_and_empty_batch() {
        val singleSample = f1Score()
        singleSample.update(
            tensor(Shape(3), floatArrayOf(0.1f, 0.7f, 0.2f)),
            intTensor(Shape(), intArrayOf(1)),
            ctx
        )
        assertEquals(1.0, singleSample.compute(), 1e-6)

        val emptyBatch = f1Score()
        emptyBatch.update(tensor(Shape(0, 3), floatArrayOf()), intTensor(Shape(0), intArrayOf()), ctx)
        assertEquals(0.0, emptyBatch.compute(), 1e-6)
    }
}
