"""
Demonstrates how to train a keras MobileNetV2 architecture and
check the constraints of the model using the BionettaModel class. After
the training, we save the model locally.
"""

from typing import List

import numpy as np
import tensorflow as tf
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


def ResNetBlock(inputs, channels, down_sample=False):
    res = x = inputs  # Save res for skip connection
    strides = [2, 1] if down_sample else [1, 1]

    KERNEL_SIZE = (3, 3)
    INIT_SCHEME = "he_normal"

    x = tf.keras.layers.Conv2D(channels, strides=strides[0],
                               kernel_size=KERNEL_SIZE, padding='same', kernel_initializer=INIT_SCHEME)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels, strides=strides[1],
                               kernel_size=KERNEL_SIZE, padding='same', kernel_initializer=INIT_SCHEME)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual path
    if down_sample:
        res = tf.keras.layers.Conv2D(
            channels, strides=2, kernel_size=(1, 1), padding='same', kernel_initializer=INIT_SCHEME
        )(res)
        res = tf.keras.layers.BatchNormalization()(res)

    # Merge and activate
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.ReLU()(x)
    return x


# The model is taken from https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras
def ResNet18(num_classes: int, input_shape: List[int], name: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Conv2D(
        64, kernel_size=(7,7), strides=2, padding='same', kernel_initializer='he_normal'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # NOTE: Originally MaxPool2D
    x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(x)

    # ResNet blocks
    x = ResNetBlock(x, 64)
    x = ResNetBlock(x, 64)
    x = ResNetBlock(x, 128, down_sample=True)
    x = ResNetBlock(x, 128)
    x = ResNetBlock(x, 256, down_sample=True)
    x = ResNetBlock(x, 256)
    x = ResNetBlock(x, 512, down_sample=True)
    x = ResNetBlock(x, 512)

    # Classifier
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs, name=name)


# 1. Load the MNIST dataset first
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension

X_train = tf.image.resize(X_train, (32, 32))
X_test = tf.image.resize(X_test, (32, 32))

# The size has been changed, as MobileNet only accepts 32x32 or larger parameters
X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train)).numpy()
X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test)).numpy()

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test, 10) # One-hot encode the labels


# 3. Wrap the model architecture with our custom BionettaModel class
# Best params for current model
proving_backend = ProvingBackend.ULTRAGROTH(16, 3)

# NOTE: Also can do it with model = ResNet18()
name = 'ResNet18'
model = ResNet18(
    num_classes=10,
    input_shape=(32, 32, 3),
    name=name
)

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

for layer in model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

folder = 'resnet18_circuits'
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
