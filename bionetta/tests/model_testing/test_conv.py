"""
Test of Conv2D layer. It works as follows:
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
from tf_bionetta.specs.backend_enums import(
    ProvingBackend,
    Groth16,
    UltraGroth
)


class ConvTest(unittest.TestCase):
    """
    Class for testing the of Conv2D correctness of the Bionetta framework
    with a simple MNIST model.
    """
    
    MODEL_NAME = 'conv_model'
    
    
    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        # NOTE: UltraGroth LookupReLU6 in process of implementing...
        self.proving_backend = ProvingBackend.GROTH16()

        inputs = tf.keras.layers.Input(shape=(28,28,1))

        # Can change only use_bias argument
        x = tf.keras.layers.Conv2D(
            kernel_size=4, filters=3, use_bias=True,
            activation=None, padding="same", kernel_initializer="he_normal"
        )(inputs)
        x = tf.keras.layers.Flatten(input_shape=(28,28,3))(x)
        x = tf.keras.layers.Dense(units = 64, activation=tfb.layers.HardSigmoid())(x)
        x = tf.keras.layers.Dense(10)(x)
        outputs = L2UnitNormalizationLayer()(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=self.MODEL_NAME)


    def test_Conv(self) -> None:
        test_model(self, './compiled_conv', self.proving_backend)
