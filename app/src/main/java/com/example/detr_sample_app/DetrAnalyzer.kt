package com.example.detr_sample_app

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean

class DetrAnalyzer(
    private val detector: DetrDetector,
    private val inferenceExecutor: ExecutorService,
    private val onResult: (DetrFrameResult) -> Unit,
    private val onError: (Throwable) -> Unit,
) : ImageAnalysis.Analyzer {
    private val inferenceBusy = AtomicBoolean(false)

    override fun analyze(image: ImageProxy) {
        if (!inferenceBusy.compareAndSet(false, true)) {
            image.close()
            return
        }

        val bitmap = try {
            DetrPreprocessor.imageProxyToBitmap(image)
        } catch (throwable: Throwable) {
            image.close()
            inferenceBusy.set(false)
            onError(throwable)
            return
        }
        image.close()

        inferenceExecutor.execute {
            try {
                val detections = detector.detect(bitmap)
                onResult(
                    DetrFrameResult(
                        detections = detections,
                        frameWidth = bitmap.width,
                        frameHeight = bitmap.height,
                        backend = detector.selectedBackend,
                    ),
                )
            } catch (throwable: Throwable) {
                Log.e(TAG, "DETR inference failed", throwable)
                onError(throwable)
            } finally {
                bitmap.recycleIfMutable()
                inferenceBusy.set(false)
            }
        }
    }

    private fun Bitmap.recycleIfMutable() {
        if (!isRecycled) recycle()
    }

    private companion object {
        const val TAG = "DetrAnalyzer"
    }
}

data class DetrFrameResult(
    val detections: List<Detection> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val backend: String = "",
)
