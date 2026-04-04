"""
Module for testing Light Encoder-Decoder Convolution Layer
"""

from __future__ import annotations

import os
import unittest

import tensorflow as tf
import numpy as np

from tf_bionetta.layers.conv.edlight import EDLight2DConv
from tf_bionetta.layers.conv.edheavy import EDHeavy2DConv


class EDConvTestCase(unittest.TestCase):
    """
    Test case for the Encoder-Decoder Light+Heavy Convolutional Layers.
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable irritating warnings

        # Simply use the first layer of the ZKBioNetV2 architecture
        self.light_layer = EDLight2DConv(
            kernel_size=16,
            channels=8,
            hidden_units=5,
            kernel_output_size=8,
        )
        self.heavy_layer = EDHeavy2DConv(
            kernel_size=16,
            channels=8,
            hidden_units=5,
            kernel_output_size=8,
        )
        self.dense = tf.keras.layers.Dense(20, activation=None, input_shape=(2048,))


    def testEDLightLaunch(self) -> None:
        """
        Tests whether the encoder decoder works at all
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(32, 32, 32, 1)
        )  # Batch of 32 grayscale images of size 32x32x1
        y = self.light_layer(x)

        assert (
            tf.TensorShape([32, 16, 16, 8]) == y.shape
        ), f"Expected shape (32, 16, 16, 8), got {y.shape}"


    def testEDHeavyLaunch(self) -> None:
        """
        Tests whether the encoder decoder works at all
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(32, 32, 32, 1)
        )  # Batch of 32 grayscale images of size 32x32x1
        y = self.heavy_layer(x)

        assert (
            tf.TensorShape([32, 16, 16, 8]) == y.shape
        ), f"Expected shape (32, 16, 16, 8), got {y.shape}"


    def testEDLightGradientFlow(self) -> None:
        """
        Tests whether the gradient descent is well-defined for the given layer
        """

        x = tf.random.normal(
            shape=(32, 32, 32, 1)
        )  # Batch of 32 inputs, input dimensionality (32, 32, 1)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.light_layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, self.light_layer.trainable_variables)
        check = all(g is not None for g in grads)
        assert check, "Some gradients are None, gradient flow is broken"


    def testEDLightModelIntegration(self) -> None:
        """
        Tests how the layer behaves when integrated into a model
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(16, 16, 1)),
                EDLight2DConv(
                    kernel_size=8,
                    channels=8,
                    hidden_units=16,
                    kernel_output_size=4,
                ),
                tf.keras.layers.GlobalAveragePooling2D(),  # This gives a shape (batch_size, 8)
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 100
        x_train = np.random.randn(TESTS_NUM, 16, 16, 1).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 8).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, verbose=0)


    def testEDLightModelIntegrationTwoConvs(self) -> None:
        """
        Tests how the layer behaves when integrated into a model when two sequential EDLight2DConv layers are used
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(32, 32, 1)),
                EDLight2DConv(
                    kernel_size=4,
                    channels=8,
                    hidden_units=5,
                    kernel_output_size=2,
                    activation=tf.keras.layers.LeakyReLU(alpha=1 / 4),
                ),  # Output: (batch_size, 16, 16, 8)
                EDLight2DConv(
                    kernel_size=8,
                    channels=16,
                    hidden_units=10,
                    kernel_output_size=4,
                ),  # Output: (batch_size, 8, 8, 16)
                tf.keras.layers.GlobalAveragePooling2D(),  # This gives a shape (batch_size, 16)
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 3
        x_train = np.random.randn(TESTS_NUM, 32, 32, 1).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 16).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, verbose=0)


    def testEDLightAndHeavyModelIntegration(self) -> None:
        """
        Tests how the layer behaves when integrated into a model when
        two sequential EDLight2DConv+EDHeavy2DConv layers are used
        """

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(16, 16, 1)),
                EDLight2DConv(
                    kernel_size=8,
                    channels=8,
                    hidden_units=5,
                    kernel_output_size=8,
                ),  # Output: (batch_size, 8, 8, 8)
                EDHeavy2DConv(
                    kernel_size=4,
                    channels=16,
                    hidden_units=10,
                    kernel_output_size=2,
                ),  # Output: (batch_size, 4, 4, 16)
                tf.keras.layers.GlobalAveragePooling2D(),  # This gives a shape (batch_size, 16)
            ]
        )

        # Compile and Train on Dummy Data
        TESTS_NUM = 100
        x_train = np.random.randn(TESTS_NUM, 16, 16, 1).astype(np.float32)
        y_train = np.random.randn(TESTS_NUM, 16).astype(np.float32)

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, verbose=0)
