"""Convert Keras 3 .h5 models to ONNX using keras2onnx / tf2onnx SavedModel path.

Usage:
    pip install tensorflow-macos tensorflow-metal keras tf2onnx
    python scripts/convert_to_onnx_v2.py
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

try:
    import tensorflow as tf
    import keras
    import tf2onnx
    import onnx
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

print(f"TF: {tf.__version__}, Keras: {keras.__version__}, tf2onnx: {tf2onnx.__version__}")

MODELS = {
    "fer2013_model": "models/fer2013_model.h5",
    "ckplus_model":  "models/ckplus_model.h5",
    "affectnet_model": "models/affectnet_model.h5",
}

INPUT_SHAPE = (1, 48, 48, 1)

for name, h5_path in MODELS.items():
    if not os.path.exists(h5_path):
        print(f"Skipping {name}: {h5_path} not found")
        continue

    onnx_path = h5_path.replace(".h5", ".onnx")
    saved_model_path = f"/tmp/{name}_saved_model"
    print(f"\nConverting {name}...")

    try:
        model = keras.models.load_model(h5_path)
        print(f"  Loaded model, output shape: {model.output_shape}")

        # Warm up the model so Keras traces the graph
        dummy = np.zeros(INPUT_SHAPE, dtype=np.float32)
        model(dummy)

        # Use Keras 3 native ONNX export
        model.export(onnx_path, format="onnx")
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  -> {onnx_path}  ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print("\nDone.")
