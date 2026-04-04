"""
Test of DepthwiseConv2D layer. It works as follows:
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
from tf_bionetta.layers import L2UnitNormalizationLayer
from tf_bionetta.specs.backend_enums import ProvingBackend


class DWConvTest(unittest.TestCase):
    """
    Class for testing the DepthwiseConv2D layer of the Bionetta framework
    with a simple MNIST model.
    """
    
    MODEL_NAME = 'dwconv_model'
    
    
    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        inputs = tf.keras.layers.Input(shape=(28,28,1))
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=4, depth_multiplier=3, activation=None,
            use_bias=False, padding="same", kernel_initializer="he_normal"
        )(inputs)
        x = tf.keras.layers.Flatten(input_shape=(28,28,3))(x)
        x = tf.keras.layers.Dense(10)(x)

        outputs = L2UnitNormalizationLayer()(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=self.MODEL_NAME)

    def test_Conv(self) -> None:
        test_model(self, './compiled_dwconv_nobias', proving_backend=ProvingBackend.GROTH16())
        #test_model(self, './compiled_dwconv_ug_nobias', proving_backend=ProvingBackend.ULTRAGROTH())