"""
Module for testing ZKBioNet architectures in TensorFlow.
"""

from __future__ import annotations

import os
import unittest

import tensorflow as tf

from tf_bionetta.layers.normalization.class_projection import ClassProjectionLayer


class ClassProjectionLayerTestCase(unittest.TestCase):
    """
    Test case for the ClassProjectionLayer
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable irritating warnings

        # Create the basic model, consisting of the following:
        # 1. Dense Layer with 10 units
        # 2. Normalization Layer
        # 3. ProjectionLayer with 20 classes
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(30,)),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1)),
                ClassProjectionLayer(20),
            ]
        )

    def testLaunch(self) -> None:
        """
        Launches the model and checks the output shape
        """

        # Generate a random input
        x = tf.random.normal(shape=(32, 30))
        y = self.model(x)

        assert y.shape == tf.TensorShape(
            [32, 20]
        ), f"Expected shape (32, 20), got {y.shape}"

        # Assert that each output neuron is in range [-1, 1]
        assert tf.reduce_all(
            tf.logical_and(y >= -1, y <= 1)
        ), "Values out of range [-1, 1]"

    def testGradientFlow(self) -> None:
        """
        Tests whether the gradient descent is well-defined for the given layer
        """

        x = tf.random.normal(shape=(32, 30))

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.model(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, self.model.trainable_variables)
        check = all(g is not None for g in grads)
        assert check, "Some gradients are None, gradient flow is broken"
