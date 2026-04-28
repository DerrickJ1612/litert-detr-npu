#!/usr/bin/env python3
"""Mixed-precision int8 quantization for DETR — CPU ONLY.

Most ops are quantized to int8 (ResNet backbone, all linear projections).
The 34 BATCH_MATMUL ops in DETR's transformer attention fall back to float32
because XNNPACK has no int8 kernel for the 3-D x 4-D broadcasting shape
DETR uses.  As a result this model ONLY runs correctly on CPU — GPU and NPU
delegates reject it due to the mixed float32/int8 op graph.

For NPU/GPU use the float16 model instead:
    app/src/main/assets/README.md

Output
------
    build/model_exports/detr_tf_nchw_to_nhwc/detr_resnet50_640_nchw_mixed_int8_cpu.tflite

Usage
-----
# COCO val images downloaded automatically for calibration:
python scripts/quantize_tflite_mixed_int8_cpu.py

# Provide your own JPEG calibration images:
python scripts/quantize_tflite_mixed_int8_cpu.py --calib-dir /path/to/jpegs
"""

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

SAVED_MODEL_DIR = Path("build/model_exports/detr_tf_nchw_to_nhwc")
OUTPUT          = Path("build/model_exports/detr_tf_nchw_to_nhwc/detr_resnet50_640_nchw_mixed_int8_cpu.tflite")
CALIB_CACHE_DIR = Path("build/calib_images")

COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_ENTRY   = "annotations/instances_val2017.json"
COCO_IMG_BASE    = "http://images.cocodataset.org/val2017"
INPUT_SIZE       = 640


# ---------------------------------------------------------------------------
# COCO download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done  = 0
        with dest.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({100 * done // total}%)", end="", flush=True)
    print()


def _fetch_annotations(dest_json: Path) -> None:
    print("Downloading COCO val2017 annotations (one-time, ~241 MB) …")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as ntf:
        tmp = Path(ntf.name)
    try:
        _download_with_progress(COCO_ANN_ZIP_URL, tmp)
        with zipfile.ZipFile(tmp) as zf:
            with zf.open(COCO_ANN_ENTRY) as src:
                dest_json.write_bytes(src.read())
        print(f"  Saved image list → {dest_json}")
    finally:
        tmp.unlink(missing_ok=True)


def _ensure_coco_images(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ann_path = dest / "instances_val2017.json"
    if not ann_path.exists():
        _fetch_annotations(ann_path)
    with ann_path.open() as f:
        all_images = json.load(f)["images"]
    filenames = [img["file_name"] for img in all_images[:200]]
    needed    = [fn for fn in filenames if not (dest / fn).exists()]
    if not needed:
        print(f"Using 200 cached COCO calibration images from {dest}")
        return
    print(f"Downloading {len(needed)} COCO val2017 images into {dest} …")
    for i, fname in enumerate(needed, 1):
        urllib.request.urlretrieve(f"{COCO_IMG_BASE}/{fname}", dest / fname)
        print(f"\r  {i}/{len(needed)}: {fname}", end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------

def make_representative_dataset(calib_dir: Path):
    try:
        import cv2  # type: ignore[import-untyped]
        def _load(path):
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    except ImportError:
        from PIL import Image
        def _load(path):
            return np.array(Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))

    images = sorted(calib_dir.glob("*.jpg"))[:200]
    if not images:
        raise FileNotFoundError(f"No JPEG images found in {calib_dir}")
    print(f"Calibrating with {len(images)} images …")

    def gen():
        for path in images:
            tensor = (_load(path) / 255.0).astype(np.float32)[np.newaxis]
            yield [tensor]

    return gen


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(calib_dir: Path | None) -> None:
    if not SAVED_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"SavedModel not found at {SAVED_MODEL_DIR}.\n"
            "Run onnx2tf first to produce the SavedModel."
        )

    if calib_dir is None:
        _ensure_coco_images(CALIB_CACHE_DIR)
        calib_dir = CALIB_CACHE_DIR

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calib_dir)
    # INT8 for everything that has an int8 kernel; float32 fallback for
    # BATCH_MATMUL ops that XNNPACK cannot run in int8 (3-D x 4-D shape).
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    print("Converting (mixed_int8_cpu) …")
    tflite_model = converter.convert()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(tflite_model)
    mb = OUTPUT.stat().st_size / 1_000_000
    print(f"Wrote {OUTPUT} ({mb:.1f} MB)")
    print()
    print("NOTE: This model is CPU-ONLY.")
    print("      GPU/NPU delegates reject it due to mixed float32/int8 ops.")
    print(f"      To deploy: cp {OUTPUT} app/src/main/assets/detr_litert_model.tflite")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of JPEG images for calibration "
             "(defaults to auto-downloaded COCO val2017 subset).",
    )
    args = ap.parse_args()
    convert(calib_dir=args.calib_dir)


if __name__ == "__main__":
    main()
