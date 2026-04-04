"""
Test of Residual layer. It works as follows:
- It defines the model architecture
- Trains the model on the MNIST dataset
- Compiles the circit for the model
- Saves the model and the circuit
- Generates the input for the circuit
- Checks the output of the circuit
- Verifies that the circuit's output is the same as the model's output
"""
from __future__ import annotations

import unittest

import tensorflow as tf

from tests.model_testing.utils import test_model
import tf_bionetta as tfb
from tf_bionetta.layers import L2UnitNormalizationLayer
from tf_bionetta.layers.custom_objects import EDLight2DConv
from tf_bionetta.layers.se.heavy import SEHeavyBlock
from .utils import test_model
from tf_bionetta.specs.backend_enums import(
    ProvingBackend,
    Groth16,
    UltraGroth
)


class ResidualTest(unittest.TestCase):
    """
    Class for testing the Residual layer of the Bionetta framework
    with a simple MNIST model.
    """
    
    MODEL_NAME = 'residual_model'


    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        self.proving_backend = ProvingBackend.ULTRAGROTH(13, 2)

        if isinstance(self.proving_backend, Groth16):
            activation = tfb.layers.ShiftReLU(5)  # LeakyReLU with alpha=1/(2**5)=1/32
        elif isinstance(self.proving_backend, UltraGroth):
            activation = tf.keras.layers.ReLU()

        # kernel_initializer="he_normal"
        inputs = tf.keras.layers.Input(shape=(28,28,1))
        y = EDLight2DConv(kernel_size=4, kernel_output_size=4, channels=3)(inputs)
        z = SEHeavyBlock(kernel_size=4)(y)

        x = tf.keras.layers.Add()([y, z])
        x = tf.keras.layers.Flatten(input_shape=(28,28,3))(x)
        x = tf.keras.layers.Dense(units=64)(x)
        x = activation(x)
        x = tf.keras.layers.Dense(10)(x)
        outputs = L2UnitNormalizationLayer()(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=self.MODEL_NAME)


    def test_Residual(self) -> None:
        test_model(self, './compiled_residual', self.proving_backend)