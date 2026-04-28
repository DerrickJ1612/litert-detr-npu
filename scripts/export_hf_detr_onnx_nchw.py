#!/usr/bin/env python3
"""Export facebook/detr-resnet-50 to ONNX with a standard NCHW input.

onnx2tf normally converts NCHW model inputs to NHWC TensorFlow/TFLite inputs.
This script relies on that behavior so the final TFLite model can accept
[1, 640, 640, 3] float32 RGB while keeping Hugging Face normalization in-graph.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import DetrForObjectDetection


MODEL_ID = "facebook/detr-resnet-50"
OUTPUT = Path("build/model_exports/detr_resnet50_640_nchw.onnx")


class DetrNchwWrapper(nn.Module):
    def __init__(self, model: DetrForObjectDetection) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = (image_nchw - self.mean) / self.std
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=False,
    ).eval()
    wrapped = DetrNchwWrapper(model).eval()
    example = torch.zeros((1, 3, 640, 640), dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        example,
        OUTPUT.as_posix(),
        input_names=["image"],
        output_names=["logits", "boxes"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
