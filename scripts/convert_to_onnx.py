"""Convert .h5 Keras models to ONNX format.

Usage: python scripts/convert_to_onnx.py

Requires: pip install tf2onnx tensorflow
These are only needed for conversion, not for serving.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tf2onnx

MODELS = {
    "fer2013_model": "models/fer2013_model.h5",
    "ckplus_model": "models/ckplus_model.h5",
    "affectnet_model": "models/affectnet_model.h5",
}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"Skipping {name}: {path} not found")
        continue

    print(f"Converting {name}...")
    model = tf.keras.models.load_model(path)
    spec = (tf.TensorSpec(model.input_shape, tf.float32),)
    output_path = path.replace(".h5", ".onnx")
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"  -> {output_path}")

print("Done! You can now remove tensorflow from requirements-api.txt")
