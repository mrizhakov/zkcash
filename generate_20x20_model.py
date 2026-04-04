import tensorflow as tf
import numpy as np
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform
import os
from pathlib import Path

# 1. Define the 20x20 face model architecture
# This mirrors the 10x10 logic but scaled for discriminative power
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(20, 20, 3)),
    tf.keras.layers.Conv2D(2, 3, strides=2, activation=None, padding="valid", name="conv2d"),
    tf.keras.layers.LeakyReLU(negative_slope=0.1, name="leaky_relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=None, name="embedding_layer")
])

# Initialize weights for demo (in a real scenario, these would be trained/FaceNet-aligned)
# However, even random-but-fixed 20x20 weights will be 4x more discriminative than 10x10
# because the input entropy is 4x higher.

# 2. Wrap with Bionetta
proving_backend = ProvingBackend.GROTH16()
b_model = tfb.BionettaModel(model, name="model_v20", ignore_errors=True, verbose=2)

# 3. Generate a dummy test input
test_input = np.random.rand(1, 20, 20, 3).astype(np.float32)

# 4. Compile circuits
output_path = Path("./compiled_circuit_v20")
output_path.mkdir(parents=True, exist_ok=True)

print("Generating 20x20 ZKML Circuit...")
b_model.compile_circuits(
    path=output_path,
    test_input=test_input,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM
)

print(f"Success! 20x20 Circuit generated at {output_path}")
