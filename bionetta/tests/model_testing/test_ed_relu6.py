"""
Test of EDConv2D layer. It works as follows:
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
from tf_bionetta.specs.backend_enums import ProvingBackend


class EdTest(unittest.TestCase):
    """
    Class for testing the EDConv2D correctness of the Bionetta framework
    with a simple MNIST model.
    """
    
    MODEL_NAME = 'ed_model'
    
    
    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        inputs = tf.keras.layers.Input(shape=(28,28,1))
        x = EDLight2DConv(kernel_size=4, kernel_output_size=4, channels=1, activation=tfb.layers.ReLU6())(inputs)
        x = tf.keras.layers.Flatten(input_shape=(28,28,3))(x)
        x = tf.keras.layers.Dense(units = 64)(x)
        x = tfb.layers.ShiftReLU(5)(x)  # LeakyReLU with alpha=1/(2**5)=1/32
        x = tf.keras.layers.Dense(10)(x)

        outputs = L2UnitNormalizationLayer()(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=self.MODEL_NAME)

    def test_ED(self) -> None:
        test_model(self, './compiled_ed_relu6', proving_backend=ProvingBackend.GROTH16())