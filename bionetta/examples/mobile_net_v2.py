"""
Demonstrates how to train a keras MobileNetV2 architecture and
check the constraints of the model using the BionettaModel class. After
the training, we save the model locally.
"""

import numpy as np
import tensorflow as tf
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


# 1. Load the MNIST dataset first
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension

# The size has been changed, as MobileNet only accepts 32x32 or larger parameters
X_train = tf.image.resize(X_train, (32,32)).numpy()
X_test = tf.image.resize(X_test, (32,32)).numpy()

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test, 10) # One-hot encode the labels

# 2. Define the model architecture
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 1),
    classes=10,
    weights=None,
    classifier_activation=None
)

# 3. Wrap the model architecture with our custom BionettaModel class
# Best params for current model
#raise ValueError("Currently in development")

proving_backend = ProvingBackend.ULTRAGROTH(16, 3)

name = 'mobile_net_v2'
model = tfb.BionettaModel(model=model, name=name, verbose=2)
model.constraints_summary(proving_backend)
test_input = X_test[np.random.randint(len(X_test))]

# 4. Compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
    epochs=1,
    validation_data=(X_test, y_test),
    validation_split=0.2
)

folder = 'mobile_net_circuits'
# 5. Compile the model
model.compile_circuits(
    path=f'./examples/{folder}',
    test_input=test_input,
    save_weights=True,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM
)

# 6. Init your architecture from compiled dir
loaded_model = tfb.BionettaModel.load_from_compiled_folder(f'./examples/{folder}', name=name, verbose=2)
test_input = X_test[np.random.randint(len(X_test))]

# 7. Prove and verify your input. Simple mnist should be previousely compiled!!!
proof_dir = f'./examples/{folder}/proof'

proof = loaded_model.prove(
    input=test_input,
    target_dir=proof_dir,
)

print('The resultant proof:', proof)

assert loaded_model.verify(proof_dir=proof_dir), "model verification failed"
