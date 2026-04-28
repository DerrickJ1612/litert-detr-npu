package com.example.detr_sample_app

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

class DetrDetector(
    context: Context,
    private val modelAssetName: String = MODEL_ASSET_NAME,
    private val confidenceThreshold: Float = 0.8f,
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val labels = loadLabels(appContext)
    private val postprocessor = DetrPostprocessor(labels, confidenceThreshold)
    private var activeModel: ActiveModel? = null
    private var inferenceCount = 0

    val selectedBackend: String
        get() = activeModel?.backendName ?: "uninitialized"

    @Synchronized
    fun detect(bitmap: android.graphics.Bitmap): List<Detection> {
        val startNs = SystemClock.elapsedRealtimeNanos()
        val preprocessed = DetrPreprocessor.preprocess(bitmap)
        val preprocessNs = SystemClock.elapsedRealtimeNanos()
        var model = activeModel ?: createFirstAvailableModel()

        val detections = try {
            runModel(model, preprocessed)
        } catch (throwable: Throwable) {
            Log.w(TAG, "${model.backendName} inference failed; trying fallback", throwable)
            model.close()
            activeModel = null
            model = createFirstAvailableModel(skipBackendsThrough = model.backend)
            runModel(model, preprocessed)
        }
        val endNs = SystemClock.elapsedRealtimeNanos()
        inferenceCount += 1
        if (inferenceCount <= INITIAL_PERF_LOGS || inferenceCount % PERF_LOG_INTERVAL == 0) {
            Log.i(
                "DetrPerf",
                "backend=${model.backendName} total=${nsToMs(endNs - startNs)}ms " +
                    "preprocess=${nsToMs(preprocessNs - startNs)}ms " +
                    "run+post=${nsToMs(endNs - preprocessNs)}ms detections=${detections.size}",
            )
        }
        return detections
    }

    private fun runModel(model: ActiveModel, preprocessed: PreprocessedFrame): List<Detection> {
        model.inputs.first().writeFloat(preprocessed.input)
        model.compiledModel.run(model.inputs, model.outputs)
        val outputArrays = model.outputs.mapIndexed { index, output ->
            output.readFloat().also { Log.v(TAG, "Output[$index] float count=${it.size}") }
        }
        return postprocessor.process(outputArrays, preprocessed.metadata)
    }

    private fun createFirstAvailableModel(skipBackendsThrough: Accelerator? = null): ActiveModel {
        val backends = listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)
        val startIndex = skipBackendsThrough?.let { backends.indexOf(it) + 1 } ?: 0
        val failures = ArrayList<Throwable>()

        for (backend in backends.drop(startIndex)) {
            try {
                Log.i(TAG, "Loading DETR LiteRT model '$modelAssetName' with backend=$backend")
                Log.i("LiteRTDebugger", "Attempting backend: $backend")
                val compiledModel = CompiledModel.create(
                    appContext.assets,
                    modelAssetName,
                    CompiledModel.Options(backend),
                )
                val inputs = compiledModel.createInputBuffers()
                val outputs = compiledModel.createOutputBuffers()
                Log.i(TAG, "Selected LiteRT backend=$backend inputs=${inputs.size} outputs=${outputs.size}")
                Log.i("LiteRTDebugger", ">>> Inference running on: $backend (inputs=${inputs.size} outputs=${outputs.size})")
                return ActiveModel(backend, compiledModel, inputs, outputs).also {
                    activeModel = it
                }
            } catch (throwable: Throwable) {
                Log.w(TAG, "Unable to initialize LiteRT backend=$backend", throwable)
                Log.w("LiteRTDebugger", "Backend $backend unavailable: ${throwable.message}")
                failures += throwable
            }
        }

        throw IllegalStateException(
            "Unable to initialize DETR LiteRT model from assets/$modelAssetName with NPU, GPU, or CPU",
            failures.lastOrNull(),
        )
    }

    @Synchronized
    override fun close() {
        activeModel?.close()
        activeModel = null
    }

    private data class ActiveModel(
        val backend: Accelerator,
        val compiledModel: CompiledModel,
        val inputs: List<TensorBuffer>,
        val outputs: List<TensorBuffer>,
    ) : AutoCloseable {
        val backendName: String = backend.name

        override fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            compiledModel.close()
        }
    }

    private companion object {
        const val TAG = "DetrDetector"
        const val MODEL_ASSET_NAME = "detr_litert_model.tflite"
        const val INITIAL_PERF_LOGS = 5
        const val PERF_LOG_INTERVAL = 30

        fun nsToMs(nanos: Long): String = "%.1f".format(nanos / 1_000_000f)

        fun loadLabels(context: Context): List<String> =
            runCatching {
                context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                    lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
                }
            }.getOrElse { throwable ->
                Log.w(TAG, "labels.txt missing; using class ids", throwable)
                emptyList()
            }
    }
}
