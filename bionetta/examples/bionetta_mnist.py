"""
Demonstrates how to train the Bionetta-based model on the MNIST dataset and
check the constraints of the model using the BionettaModel class. After the
training, we save the model locally.
"""

import tensorflow as tf
import numpy as np

import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


# 0. Define the hyperparameters for training in the separate structure (if needed)
hyperparams = tfb.hyperparameters.TrainingHyperparameters({
    'epochs': 1,
    'validation_split': 0.9
})

# 1. Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tfb.layers.EDLight2DConv(
        7,
        16,
        hidden_units=16,
        kernel_output_size=3,
        activation=tfb.layers.ShiftReLU(3),
    ),
    tf.keras.layers.Conv2D(32, 3, activation=None, padding="same"),
    tfb.layers.SEHeavyBlock(6, hidden_units=32, activation=tfb.layers.ShiftReLU(2)),
    tf.keras.layers.Conv2D(64, 3, strides=2, activation=None, padding="same"),
    tfb.layers.EDLight2DConv(
        2,
        128,
        hidden_units=16,
        kernel_output_size=3,
        activation=tfb.layers.ShiftReLU(3),
    ),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation=tfb.layers.ShiftReLU(4)),
    tf.keras.layers.Dense(10, activation=None),
])

# 3. Wrap the model architecture with our custom BionettaModel class
# NOTE: LookupLeakyReLU is not support for UltraGroth for now
proving_backend = ProvingBackend.GROTH16()
model = tfb.BionettaModel(model, ignore_errors=False, verbose=2)
model.constraints_summary(proving_backend)

# 4. Create callbacks for checking the training progress: saving the model and print example predictions

# 4.1. Generate a random batch of 4 samples for debugging
DEBUG_BATCH_SIZE = 4
random_idx = np.random.randint(len(X_train), size=DEBUG_BATCH_SIZE)
X_debug = X_train[random_idx]
y_debug = y_train[random_idx]

# 4.2. Create the callbacks
callbacks = [
    tfb.callbacks.BionettaSubmodelCheckpoint(
        model, "./examples/bionetta_mnist_test_model/training"
    ),
    tfb.callbacks.ExamplePredictionsCallback(X_debug, y_debug),
]

model.save_circuit_weights('./bionetta-mnist-temp', compress=False)

# 4.3. Finally, compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, 
    epochs=hyperparams.epochs, 
    validation_data=(X_test, y_test), 
    validation_split=hyperparams.validation_split,
    callbacks=callbacks
)

# 5. Save the resultant model
test_input = X_test[np.random.randint(len(X_test))]

model.compile_circuits(
    path='./examples/bionetta_mnist_circuits',
    test_input=test_input,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM
)

proof_dir = './examples/bionetta_mnist_circuits/proof'

model.prove(
    input=test_input,
    target_dir=proof_dir
)

model.verify(proof_dir=proof_dir)