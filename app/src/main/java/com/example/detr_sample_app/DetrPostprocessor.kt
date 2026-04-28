package com.example.detr_sample_app

import android.graphics.RectF
import android.util.Log
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val box: RectF,
    val label: String,
    val confidence: Float,
)

data class FrameMetadata(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inputSize: Int,
    val scale: Float,
    val padX: Float,
    val padY: Float,
)

class DetrPostprocessor(
    private val labels: List<String>,
    private val confidenceThreshold: Float = 0.5f,
) {
    fun process(outputs: List<FloatArray>, metadata: FrameMetadata): List<Detection> {
        if (outputs.size < 2) return emptyList()

        val boxes = outputs.firstOrNull { looksLikeBoxes(it) } ?: outputs.minBy { it.size }
        val logits = outputs.firstOrNull { it !== boxes } ?: return emptyList()
        if (boxes.size < 4) return emptyList()

        val queryCount = boxes.size / 4
        val classCount = logits.size / queryCount
        if (queryCount == 0 || classCount <= 1) return emptyList()

        Log.d(TAG, "Postprocess tensors: queries=$queryCount classes=$classCount boxes=${boxes.size} logits=${logits.size}")

        val detections = ArrayList<Detection>()
        for (query in 0 until queryCount) {
            val classOffset = query * classCount
            val scores = logits.copyOfRange(classOffset, classOffset + classCount)
            val probabilities = softmax(scores)
            val noObjectIndex = classCount - 1
            var bestClass = -1
            var bestScore = 0f
            for (classIndex in 0 until classCount) {
                if (classIndex == noObjectIndex) continue
                if (probabilities[classIndex] > bestScore) {
                    bestScore = probabilities[classIndex]
                    bestClass = classIndex
                }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue
            val labelName = labels.getOrElse(bestClass) { "class $bestClass" }
            if (labelName == "N/A") continue

            val boxOffset = query * 4
            val cx = boxes[boxOffset]
            val cy = boxes[boxOffset + 1]
            val w = boxes[boxOffset + 2]
            val h = boxes[boxOffset + 3]

            val inputLeft = (cx - w / 2f) * metadata.inputSize
            val inputTop = (cy - h / 2f) * metadata.inputSize
            val inputRight = (cx + w / 2f) * metadata.inputSize
            val inputBottom = (cy + h / 2f) * metadata.inputSize

            val left = ((inputLeft - metadata.padX) / metadata.scale).coerceIn(0f, metadata.sourceWidth.toFloat())
            val top = ((inputTop - metadata.padY) / metadata.scale).coerceIn(0f, metadata.sourceHeight.toFloat())
            val right = ((inputRight - metadata.padX) / metadata.scale).coerceIn(0f, metadata.sourceWidth.toFloat())
            val bottom = ((inputBottom - metadata.padY) / metadata.scale).coerceIn(0f, metadata.sourceHeight.toFloat())

            if (right <= left || bottom <= top) continue
            detections += Detection(
                box = RectF(left, top, right, bottom),
                label = labelName,
                confidence = bestScore,
            )
        }
        return detections
    }

    private fun looksLikeBoxes(values: FloatArray): Boolean {
        if (values.size % 4 != 0) return false
        val sampleCount = min(values.size, 400)
        val inUnitRange = values.take(sampleCount).count { it in -0.1f..1.1f }
        return inUnitRange >= sampleCount * 0.9f
    }

    private fun softmax(values: FloatArray): FloatArray {
        val maxValue = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size)
        var sum = 0f
        for (i in values.indices) {
            val value = exp((values[i] - maxValue).toDouble()).toFloat()
            exps[i] = value
            sum += value
        }
        return FloatArray(values.size) { exps[it] / max(sum, 1e-6f) }
    }

    private companion object {
        const val TAG = "DetrPostprocessor"
    }
}
