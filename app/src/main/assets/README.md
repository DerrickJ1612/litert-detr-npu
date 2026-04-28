The checked-in model is converted from:

    facebook/detr-resnet-50

Source:

    https://huggingface.co/facebook/detr-resnet-50

Conversion artifacts/scripts:

    scripts/export_hf_detr_onnx_nchw.py
    build/model_exports/detr_resnet50_640_nchw.onnx
    build/model_exports/detr_tf_nchw_to_nhwc/detr_resnet50_640_nchw_float16.tflite

The packaged model is:

    app/src/main/assets/detr_litert_model.tflite

It accepts float32 RGB input shaped [1, 640, 640, 3] with values in [0, 1].
Hugging Face DETR mean/std normalization is baked into the model graph. It
emits raw class logits and normalized cxcywh boxes.
