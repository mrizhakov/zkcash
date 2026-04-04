"""
Module for testing Encoder-Decoder Layer
"""

from __future__ import annotations

import unittest

import tensorflow as tf
import numpy as np

from tf_bionetta.layers.ed import EncoderDecoderLayer


class EncoderDecoderTestCase(unittest.TestCase):
    """
    Test case for the Encoder Decoder Layer.
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        self.layer = EncoderDecoderLayer(32, 10)

    def testLaunch(self) -> None:
        """
        Tests whether the encoder decoder works at all
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(64, 16)
        )  # Batch of 64 inputs, input dimensionality 16
        y = self.layer(x)

        assert (
            tf.TensorShape([64, 32]) == y.shape
        ), f"Expected shape (64, 32), got {y.shape}"

    def testGradientFlow(self) -> None:
        """
        Tests whether the gradient descent is well-defined for the given layer
        """

        x = tf.random.normal(
            shape=(64, 16)
        )  # Batch of 64 inputs, input dimensionality 16

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, self.layer.trainable_variables)
        check = all(g is not None for g in grads)
        assert check, "Some gradients are None, gradient flow is broken"

    def testModelIntegration(self) -> None:
        """
        Tests how the layer behaves when integrated into a model
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(16,)),
                EncoderDecoderLayer(
                    units=32, hidden_units=10, activation=tf.keras.layers.ReLU()
                ),
                tf.keras.layers.Dense(1),
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 100
        x_train = np.random.randn(TESTS_NUM, 16).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 1).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1)
