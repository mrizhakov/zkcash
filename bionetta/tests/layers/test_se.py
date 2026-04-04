"""
Module for testing the Squeeze-and-Excitation block implementation,
based on the implementation below:

https://arxiv.org/pdf/1709.01507
"""

from __future__ import annotations

import unittest

import tensorflow as tf
import numpy as np

from tf_bionetta.layers.se.light import SELightBlock
from tf_bionetta.layers.se.heavy import SEHeavyBlock


class SqueezeAndExcitationTestCase(unittest.TestCase):
    """
    Test case for the Squeeze-and-Excitation Block.
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        self.light_layer = SELightBlock(hidden_units=10)
        self.heavy_layer = SEHeavyBlock(kernel_size=8, hidden_units=6)


    def testLaunch(self) -> None:
        """
        Tests whether the encoder decoder works at all
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(32, 64, 64, 16)
        )  # Batch of 32 inputs, input dimensionality (64, 64, 16)
        y = self.light_layer(x)

        assert x.shape == y.shape, f"Expected shape not to change, got {y.shape}"


    def testHeavyLaunch(self) -> None:
        """
        Tests whether the encoder decoder works at all using heavy block
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(32, 56, 56, 16)
        )  # Batch of 32 inputs, input dimensionality (64, 64, 16)
        y = self.heavy_layer(x)

        assert x.shape == y.shape, f"Expected shape not to change, got {y.shape}"


    def testGradientFlow(self) -> None:
        """
        Tests whether the gradient descent is well-defined for the given layer
        """

        x = tf.random.normal(
            shape=(32, 64, 64, 16)
        )  # Batch of 64 inputs, input dimensionality (64, 64, 16)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.light_layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, self.light_layer.trainable_variables)
        check = all(g is not None for g in grads)
        assert check, "Some gradients are None, gradient flow is broken"


    def testModelIntegration(self) -> None:
        """
        Tests how the layer behaves when integrated into a model
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 16)),
                SELightBlock(10),
                tf.keras.layers.GlobalAveragePooling2D(),  # This gives a shape (batch_size, 16)
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 100
        x_train = np.random.randn(TESTS_NUM, 64, 64, 16).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 16).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, verbose=0)


    def testHeavyModelIntegration(self) -> None:
        """
        Tests how the layer behaves when integrated into a model
        with both heavy and light SEBlock's
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 16)),
                SELightBlock(10),
                SEHeavyBlock(8, 6),
                tf.keras.layers.GlobalAveragePooling2D(),  # This gives a shape (batch_size, 16)
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 100
        x_train = np.random.randn(TESTS_NUM, 64, 64, 16).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 16).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, verbose=0)
