#!/usr/bin/env python3
"""Re-quantize the DETR SavedModel to int8.

Two modes
---------
dynamic  (default)
    Weights → int8, activations → float32 at runtime.
    No calibration data needed.  ~40 MB.  Fast on CPU; most GPU/NPU
    delegates do NOT accelerate dynamic-range kernels.

full-int8
    Weights and activations → int8.  ~40 MB.  This is the quantization
    format most Android NPU delegates require.  Calibration images are
    downloaded automatically from COCO val2017 if --calib-dir is omitted.

Usage
-----
# Dynamic range (no images needed):
python scripts/quantize_tflite.py

# Full INT8 — COCO val images downloaded automatically:
python scripts/quantize_tflite.py --full-int8

# Full INT8 — use your own directory of JPEG images:
python scripts/quantize_tflite.py --full-int8 --calib-dir /path/to/jpegs
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
OUTPUT_DIR      = Path("build/model_exports/detr_tf_nchw_to_nhwc")
CALIB_CACHE_DIR = Path("build/calib_images")

COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_ENTRY   = "annotations/instances_val2017.json"
COCO_IMG_BASE    = "http://images.cocodataset.org/val2017"
# NUM_CALIB_IMAGES = 200
INPUT_SIZE       = 640


# ---------------------------------------------------------------------------
# COCO download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done  = 0
        with dest.open("wb") as f:
            while chunk := resp.read(1 << 20):  # 1 MiB chunks
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done // total
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({pct}%)", end="", flush=True)
    print()


def _fetch_annotations(dest_json: Path) -> None:
    """Download the COCO annotations zip and extract just the instances JSON."""
    print(f"Downloading COCO val2017 annotations (one-time, ~241 MB) …")
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


def _ensure_coco_images(dest: Path, n: int) -> None:
    """Download n COCO val2017 images into dest/, skipping already-present ones."""
    dest.mkdir(parents=True, exist_ok=True)

    ann_path = dest / "instances_val2017.json"
    if not ann_path.exists():
        _fetch_annotations(ann_path)

    with ann_path.open() as f:
        all_images = json.load(f)["images"]

    filenames = [img["file_name"] for img in all_images[:n]]
    needed    = [fn for fn in filenames if not (dest / fn).exists()]

    if not needed:
        print(f"Using {len(filenames)} cached COCO calibration images from {dest}")
        return

    print(f"Downloading {len(needed)} COCO val2017 images into {dest} …")
    for i, fname in enumerate(needed, 1):
        urllib.request.urlretrieve(f"{COCO_IMG_BASE}/{fname}", dest / fname)
        print(f"\r  {i}/{len(needed)}: {fname}", end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# TFLite calibration dataset generator
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
            img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            return np.array(img)

    images = sorted(calib_dir.glob("*.jpg"))[:200]
    if not images:
        raise FileNotFoundError(f"No JPEG images found in {calib_dir}")
    print(f"Calibrating with {len(images)} images from {calib_dir} …")

    def gen():
        for path in images:
            tensor = (_load(path) / 255.0).astype(np.float32)[np.newaxis]  # [1,640,640,3]
            yield [tensor]

    return gen


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(full_int8: bool, calib_dir: Path | None) -> Path:
    if not SAVED_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"SavedModel not found at {SAVED_MODEL_DIR}.\n"
            "Run onnx2tf first to produce the SavedModel."
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if full_int8:
        if calib_dir is None:
            _ensure_coco_images(CALIB_CACHE_DIR, 200)
            calib_dir = CALIB_CACHE_DIR

        converter.representative_dataset    = make_representative_dataset(calib_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Keep float32 I/O so the Android app needs no changes
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32
        suffix = "int8"
    else:
        suffix = "w8a32"

    print(f"Converting ({suffix}) …")
    tflite_model = converter.convert()

    output = OUTPUT_DIR / f"detr_resnet50_640_nchw_{suffix}.tflite"
    output.write_bytes(tflite_model)
    mb = output.stat().st_size / 1_000_000
    print(f"Wrote {output} ({mb:.1f} MB)")
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--full-int8",
        action="store_true",
        help="Full INT8 quantization (W8A8) — NPU-compatible. "
             "COCO val images are downloaded automatically if --calib-dir is omitted.",
    )
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of JPEG images for INT8 calibration "
             "(defaults to auto-downloaded COCO val2017 subset).",
    )
    ap.add_argument(
        "--num-calib-images",
        type=int,
        default=200,
        metavar="N",
        help=f"Number of COCO images to download/use for calibration (default: 200).",
    )
    args = ap.parse_args()

    # global NUM_CALIB_IMAGES
    NUM_CALIB_IMAGES = args.num_calib_images

    convert(full_int8=args.full_int8, calib_dir=args.calib_dir)


if __name__ == "__main__":
    main()
