package sk.ainet.lang.nn.metrics

internal object ClassificationSupport {

    internal inline fun <T : DType, V> forEachPrediction(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        dim: Int = -1,
        threshold: Float? = null,
        action: (predicted: Int, target: Int) -> Unit
    ){
        validatePredictions(predictions)
        val classDim = normalizeDim(dim, predictions.rank)

        when (targets.dtype) {
            Int32::class -> {
                require(targets.rank == predictions.rank - 1) {
                    "Expected target rank ${predictions.rank - 1} for class indices, got ${targets.rank}"
                }
                validateIndexTargetShapes(predictions, targets, classDim)

                val classCount = predictions.shape[classDim]
                @Suppress("UNCHECKED_CAST")
                val intTargets = targets as Tensor<Int32, Int>

                iterateOverSamples(intTargets.shape.dimensions) { sampleIdx ->
                    val targetClass = intTargets.data.get(*sampleIdx)
                    val predictedClass = if (threshold != null && classCount <= 2) {
                        val logitIdx = insertClassIndex(sampleIdx, if (classCount == 2) 1 else 0, classDim, predictions.rank)
                        val logit = predictions.data.get(*logitIdx) as Float
                        if (logit > threshold) 1 else 0
                    } else {
                        argmax(predictions, sampleIdx, classDim)
                    }

                    action(predictedClass, targetClass)
                }
            }
            FP32::class, FP16::class -> {
                require(predictions.shape == targets.shape) {
                    "With soft targets requires matching shapes, got ${predictions.shape.dimensions.contentToString()} vs ${targets.shape.dimensions.contentToString()}"
                }

                val classCount = predictions.shape[classDim]
                val sampleShape = predictions.shape.dimensions.filterIndexed { i, _ -> i != classDim }.toIntArray()

                iterateOverSamples(sampleShape) { sampleIdx ->
                    val predictedClass = if (threshold != null && classCount <= 2) {
                        val logitIdx = insertClassIndex(sampleIdx, if (classCount == 2) 1 else 0, classDim, predictions.rank)
                        val logit = predictions.data.get(*logitIdx) as Float
                        if (logit > threshold) 1 else 0
                    } else {
                        argmax(predictions, sampleIdx, classDim)
                    }

                    val targetClass = argmaxTarget(targets, sampleIdx, classDim)

                    action(predictedClass, targetClass)
                }
            }
            else -> error("Unsupported target dtype for classification: ${targets.dtype}")
        }
    }

    internal fun <T : DType, V> argmax(
        tensor: Tensor<T, V>,
        sampleIdx: IntArray,
        classDim: Int
    ): Int {
        val classCount = tensor.shape[classDim]
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY

        for (c in 0 until classCount) {
            val fullIdx = insertClassIndex(sampleIdx, c, classDim, tensor.rank)
            val value = tensor.data.get(*fullIdx) as Float
            if (value > maxVal) {
                maxVal = value
                maxIdx = c
            }
        }
        return maxIdx
    }

    internal fun argmaxTarget(
        tensor: Tensor<out DType, *>,
        sampleIdx: IntArray,
        classDim: Int
    ): Int {
        val classCount = tensor.shape[classDim]
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY

        for (c in 0 until classCount) {
            val fullIdx = insertClassIndex(sampleIdx, c, classDim, tensor.rank)
            val value = when (tensor.dtype) {
                FP32::class -> tensor.data.get(*fullIdx) as Float
                FP16::class -> (tensor.data.get(*fullIdx) as Number).toFloat()
                else -> error("Unsupported dtype")
            }
            if (value > maxVal) {
                maxVal = value
                maxIdx = c
            }
        }
        return maxIdx
    }

    internal fun insertClassIndex(
        baseIdx: IntArray,
        cls: Int,
        classDim: Int,
        outRank: Int
    ): IntArray {
        val result = IntArray(outRank)
        var bi = 0
        for (i in 0 until outRank) {
            if (i == classDim) {
                result[i] = cls
            } else {
                result[i] = baseIdx[bi++]
            }
        }
        return result
    }

    internal fun validateIndexTargetShapes(
        preds: Tensor<*, *>,
        targets: Tensor<*, *>,
        classDim: Int
    ) {
        val predDims = preds.shape.dimensions
        val targetDims = targets.shape.dimensions
        var ti = 0
        for (pi in predDims.indices) {
            if (pi == classDim) continue
            require(predDims[pi] == targetDims[ti]) {
                "Accuracy target shape mismatch at dim $pi: expected ${predDims[pi]}, got ${targetDims[ti]}"
            }
            ti++
        }
    }

    internal fun <T : DType, V> validatePredictions(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "Accuracy requires floating point predictions, got ${preds.dtype}"
        }
    }

    internal fun normalizeDim(dim: Int, rank: Int): Int {
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "Dimension $dim out of range for rank $rank" }
        return nd
    }

    internal inline fun iterateOverSamples(shape: IntArray, action: (IntArray) -> Unit) {
        if (shape.isEmpty()) {
            action(IntArray(0))
            return
        }

        val idx = IntArray(shape.size)
        val total = shape.fold(1) { acc, d -> acc * d }

        repeat(total) {
            action(idx)
            // Increment index
            for (d in shape.lastIndex downTo 0) {
                idx[d]++
                if (idx[d] < shape[d]) break
                idx[d] = 0
            }
        }
    }
}