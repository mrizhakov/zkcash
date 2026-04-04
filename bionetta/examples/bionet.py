"""
Module for testing ZKBioNet architecture in TensorFlow.
"""

from __future__ import annotations

import tensorflow as tf

import tf_bionetta as tfb
from tf_bionetta.applications.bionet.v1 import BioNetV1
from tf_bionetta import BionettaModel

from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


model = BioNetV1(output_size=92)

proving_backend = ProvingBackend.GROTH16()
model = BionettaModel(model, name='BioNetV1', verbose=1)

# Print the number of constraints
model.constraints_summary(proving_backend)

# Create some random training process
x = tf.random.normal(
    shape=(32, 40, 40, 3), mean=0.0, stddev=0.5
)  # Batch of 32 grayscale images of size 192x192x1
y = tf.random.normal(shape=(32, 92), mean=0.0, stddev=0.5)  # Batch of 32 outputs

model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=1, verbose=0)

test_input = tf.random.normal(shape=(1, 40, 40, 3), mean=0.0, stddev=0.5)
model.compile_circuits(
    path='./examples/bionet_test_model',
    test_input=test_input,
    save_weights=True,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM
)


proof_dir = './examples/bionet_test_model/proof'

proof = model.prove(
    input=test_input,
    target_dir=proof_dir,
)

print('The resultant proof:', proof)

assert model.verify(proof_dir=proof_dir), "model verification failed"
