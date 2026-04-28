# DETR LiteRT Camera Sample

Android CameraX sample for running DETR object detection on live camera frames
with LiteRT. The app supports front and rear cameras, overlays boxes on the
preview, and tries LiteRT backends in this order:

```text
NPU -> GPU -> CPU
```

The active model is packaged at:

```text
app/src/main/assets/detr_litert_model.tflite
```

It is converted from `facebook/detr-resnet-50` and expects float32 RGB input
shaped `[1, 640, 640, 3]` with pixel values in `[0, 1]`.

## Qualcomm NPU Runtime

Qualcomm LiteRT/QNN deployment libraries are included under:

```text
litert_npu_runtime_libraries/
```

Included runtime modules:

- `runtime_strings`
- `qualcomm_runtime_v69`
- `qualcomm_runtime_v73`
- `qualcomm_runtime_v75`
- `qualcomm_runtime_v79`
- `qualcomm_runtime_v81`

The app packages these as dynamic feature modules and uses
`app/device_targeting_configuration.xml` for App Bundle delivery to supported
Snapdragon devices. The app does not call QNN directly; `DetrDetector` creates
a LiteRT `CompiledModel` with `Accelerator.NPU`, and LiteRT loads the packaged
Qualcomm dispatch/runtime libraries when available.

## Runtime Logs

Use Logcat filters:

```text
LiteRTDebugger|DetrPerf|DetrDetector
```

Useful log lines:

```text
LiteRTDebugger: Attempting backend: NPU
LiteRTDebugger: >>> Inference running on: NPU
DetrPerf: backend=NPU total=733.1ms preprocess=7.5ms run+post=725.6ms detections=0
```

`backend=NPU` means LiteRT selected the NPU backend. It does not guarantee the
model is fast; DETR's transformer attention ops are not ideal for mobile NPU
execution.

## Build

Build a debug APK:

```bash
./gradlew :app:assembleDebug
```

Build a debug Android App Bundle:

```bash
./gradlew :app:bundleDebug
```

The AAB is the preferred distribution format because it can deliver the matching
Qualcomm runtime split.

## Additional Project Onboarding

For a new project, keep the same runtime packaging
pattern:

1. Copy `litert_npu_runtime_libraries/` into the new project.
2. Add the runtime modules to `settings.gradle.kts`:

```kotlin
include(":litert_npu_runtime_libraries:runtime_strings")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v79")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v81")
```

3. Add dynamic feature and library plugins in the root `build.gradle.kts`.
4. Add `implementation(project(":litert_npu_runtime_libraries:runtime_strings"))`
   in the app module.
5. Add the Qualcomm runtime modules to `dynamicFeatures` in the app module.
6. Copy `app/device_targeting_configuration.xml` and enable device group splits.
7. Set `minSdk = 31`, package only `arm64-v8a`, and keep legacy JNI packaging.
8. Use LiteRT `CompiledModel` with `Accelerator.NPU`, then fall back to GPU and
   CPU on failure.


## Model Conversion

Intermediate conversion outputs are written under `build/`.

### 1. Export Hugging Face DETR to ONNX

```bash
pip install torch transformers
python scripts/export_hf_detr_onnx_nchw.py
```

Output:

```text
build/model_exports/detr_resnet50_640_nchw.onnx
```

### 2. Convert ONNX to TFLite with onnx2tf

```bash
pip install onnx2tf
onnx2tf \
  -i build/model_exports/detr_resnet50_640_nchw.onnx \
  -o build/model_exports/detr_tf_nchw_to_nhwc \
  -osd \
  -cotof \
  -coto
```

Important flags:

- `-osd`: also save a TensorFlow SavedModel.
- `-cotof`: emit float32 TFLite.
- `-coto`: emit float16-weight TFLite.

The converted TFLite model uses NHWC input:

```text
[1, 640, 640, 3]
```

### 3. Optional Quantization

```bash
pip install tensorflow numpy pillow
```

Dynamic range quantization, W8A32:

```bash
python scripts/quantize_tflite.py
```

Full INT8 quantization, W8A8, using auto-downloaded COCO calibration images:

```bash
python scripts/quantize_tflite.py --full-int8
```

Full INT8 with a local calibration image directory:

```bash
python scripts/quantize_tflite.py --full-int8 --calib-dir /path/to/jpegs
```

Note: the full INT8 DETR export may fail at runtime because DETR attention uses
`BATCH_MATMUL` patterns that are not well supported by all TFLite/LiteRT
backends.

### 4. Package the Chosen Model

```bash
cp build/model_exports/detr_tf_nchw_to_nhwc/<variant>.tflite \
   app/src/main/assets/detr_litert_model.tflite
```

Rebuild the app after replacing the asset.
