"""
Tests for the Batch Normalization Folding
procedure.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import tensorflow as tf
import numpy as np

from tf_bionetta.applications.bionet import BioNetV1
from tf_bionetta.optimizations.batch_folding import fold_batch_norm
from tf_bionetta.optimizations.optimizer import BionettaModelOptimizer


class TestBatchNormFolding(unittest.TestCase):
    """
    Test case for the Batch Normalization Folding.
    """

    TOLERANCE = 1e-5

    def testDenseFolding(self) -> None:
        """
        Creates the test for the Dense layer batch normalization folding.
        """

        # Create a simple model
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16,)),
                tf.keras.layers.Dense(8),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(16, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )
        model.build((None, 16))

        # Update BatchNorm moving statistics
        print("Updating BatchNorm moving statistics...")
        for _ in range(100):
            model(tf.random.normal(shape=(64, 16), mean=0, stddev=0.5), training=True)
        model.trainable = False  # Fixate the moving parameters
        print("Done updating the BatchNorm moving statistics.")

        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())

        # Fold the BatchNorm layers, consisting of Dense layers
        dense_1, cloned_dense_1, bn_1 = (
            model.layers[0],
            cloned_model.layers[0],
            model.layers[1],
        )
        dense_2, cloned_dense_2, bn_2 = (
            model.layers[3],
            cloned_model.layers[3],
            model.layers[4],
        )
        dense_3, cloned_dense_3, bn_3 = (
            model.layers[6],
            cloned_model.layers[6],
            model.layers[7],
        )

        print("Folding BatchNorm layers...")
        folded_dense_1 = fold_batch_norm(dense_1, cloned_dense_1, bn_1)
        folded_dense_2 = fold_batch_norm(dense_2, cloned_dense_2, bn_2)
        folded_dense_3 = fold_batch_norm(dense_3, cloned_dense_3, bn_3)

        # Change Dense + BN layers to the folded ones
        folded_model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16,)),
                folded_dense_1,
                tf.keras.layers.ReLU(),
                folded_dense_2,
                tf.keras.layers.ReLU(),
                folded_dense_3,
                tf.keras.layers.ReLU(),
            ]
        )
        print("Folding done, saving both models...")

        # Save both models somewhere
        model.save("temp/test_model.h5")
        folded_model.save("temp/test_folded_model.h5")

        # Load both models again
        model = tf.keras.models.load_model("temp/test_model.h5")
        folded_model = tf.keras.models.load_model("temp/test_folded_model.h5")
        print("Models saved and loaded, removing the files...")

        # Remove the models from the disk and the corresponding folder
        Path("temp/test_model.h5").unlink()
        Path("temp/test_folded_model.h5").unlink()
        Path("temp").rmdir()

        # Check whether outputs are the same for the given set of inputs
        print("Generating random inputs and testing the divergence negligibility...")
        x = tf.random.normal(shape=(64, 16), mean=0, stddev=1.0)
        y = model(x, training=False)
        y_folded = folded_model(x, training=False)
        distance = np.linalg.norm(y.numpy() - y_folded.numpy())

        assert (
            distance < TestBatchNormFolding.TOLERANCE
        ), f"Outputs are not the same. Got distance {distance}"
        print("Test passed.")

    def testConv2DFolding(self) -> None:
        """
        Creates the test for the Conv2D layer batch normalization folding.
        """

        # Create a simple model
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16, 16, 3)),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=4, strides=1, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=2, strides=2, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, use_bias=False),
            ]
        )
        model.build((16, 16, 3))

        # Update BatchNorm moving statistics
        for _ in range(50):
            model(
                tf.random.normal(shape=(16, 16, 16, 3), mean=0, stddev=0.5),
                training=True,
            )
        model.trainable = False  # Fixate the moving parameters

        # Duplicate the model
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())

        # Fold the BatchNorm layers, consisting of Conv2D layers
        conv_1, cloned_conv_1, bn_1 = (
            model.layers[0],
            cloned_model.layers[0],
            model.layers[1],
        )
        conv_2, cloned_conv_2, bn_2 = (
            model.layers[3],
            cloned_model.layers[3],
            model.layers[4],
        )
        conv_3, cloned_conv_3, bn_3 = (
            model.layers[6],
            cloned_model.layers[6],
            model.layers[7],
        )
        output_dense = model.layers[-1]

        folded_conv_1 = fold_batch_norm(conv_1, cloned_conv_1, bn_1)
        folded_conv_2 = fold_batch_norm(conv_2, cloned_conv_2, bn_2)
        folded_conv_3 = fold_batch_norm(conv_3, cloned_conv_3, bn_3)

        # Change Dense + BN layers to the folded ones
        folded_model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16, 16, 3)),
                folded_conv_1,
                tf.keras.layers.ReLU(),
                folded_conv_2,
                tf.keras.layers.ReLU(),
                folded_conv_3,
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                output_dense,
            ]
        )

        # Save both models somewhere
        model.save("temp/test_conv_model.h5")
        folded_model.save("temp/test_folded_conv_model.h5")

        # Load both models again
        model = tf.keras.models.load_model("temp/test_conv_model.h5")
        folded_model = tf.keras.models.load_model("temp/test_folded_conv_model.h5")

        # Remove the models from the disk and the corresponding folder
        Path("temp/test_conv_model.h5").unlink()
        Path("temp/test_folded_conv_model.h5").unlink()
        Path("temp").rmdir()

        # Check whether outputs are the same for the given set of inputs
        x = tf.random.normal(shape=(64, 16, 16, 3), mean=0, stddev=1.0)
        y = model(x, training=False)
        y_folded = folded_model(x, training=False)

        distance = np.linalg.norm(y.numpy() - y_folded.numpy())
        assert (
            distance < TestBatchNormFolding.TOLERANCE
        ), f"Outputs are not the same. Got distance {distance}"

    def testDepthwiseConv2DFolding(self) -> None:
        """
        Creates the test for the DepthwiseConv2D layer batch normalization folding.
        """

        # Create a simple model
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16, 16, 3)),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=4, strides=1, depth_multiplier=1, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=2,
                    strides=1,
                    depth_multiplier=2,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=2, strides=2, depth_multiplier=4, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, use_bias=False),
            ]
        )

        # Update BatchNorm moving statistics
        for _ in range(50):
            model(
                tf.random.normal(shape=(16, 16, 16, 3), mean=0, stddev=0.5),
                training=True,
            )
        model.trainable = False  # Fixate the moving parameters

        # Clone the model
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())

        # Fold the BatchNorm layers, consisting of DepthwiseConv2D layers
        depthwise_conv_1, cloned_depthwise_conv_1, bn_1 = (
            model.layers[0],
            cloned_model.layers[0],
            model.layers[1],
        )
        depthwise_conv_2, cloned_depthwise_conv_2, bn_2 = (
            model.layers[3],
            cloned_model.layers[3],
            model.layers[4],
        )
        depthwise_conv_3, cloned_depthwise_conv_3, bn_3 = (
            model.layers[6],
            cloned_model.layers[6],
            model.layers[7],
        )
        output_dense = model.layers[-1]

        folded_depthwise_conv_1 = fold_batch_norm(
            depthwise_conv_1, cloned_depthwise_conv_1, bn_1
        )
        folded_depthwise_conv_2 = fold_batch_norm(
            depthwise_conv_2, cloned_depthwise_conv_2, bn_2
        )
        folded_depthwise_conv_3 = fold_batch_norm(
            depthwise_conv_3, cloned_depthwise_conv_3, bn_3
        )

        # Change Dense + BN layers to the folded ones
        folded_model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(16, 16, 3)),
                folded_depthwise_conv_1,
                tf.keras.layers.ReLU(),
                folded_depthwise_conv_2,
                tf.keras.layers.ReLU(),
                folded_depthwise_conv_3,
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                output_dense,
            ]
        )

        # Save both models somewhere
        model.save("temp/test_depthwise_conv_model.h5")
        folded_model.save("temp/test_folded_depthwise_conv_model.h5")

        # Load both models again
        model = tf.keras.models.load_model("temp/test_depthwise_conv_model.h5")
        folded_model = tf.keras.models.load_model(
            "temp/test_folded_depthwise_conv_model.h5"
        )

        # Remove the models from the disk and the corresponding folder
        Path("temp/test_depthwise_conv_model.h5").unlink()
        Path("temp/test_folded_depthwise_conv_model.h5").unlink()
        Path("temp").rmdir()

        # Check whether outputs are the same for the given set of inputs
        x = tf.random.normal(shape=(64, 16, 16, 3), mean=0, stddev=1.0)
        y = model(x, training=False)
        y_folded = folded_model(x, training=False)

        distance = np.linalg.norm(y.numpy() - y_folded.numpy())
        assert (
            distance < TestBatchNormFolding.TOLERANCE
        ), f"Outputs are not the same. Got distance {distance}"

    def testBioNetFolding(self) -> None:
        """
        Tests the folding of the BatchNorm layers in the BioNetV3 model.
        """

        model = BioNetV1(input_shape=(40, 40, 1), output_size=10)
        optimizer = BionettaModelOptimizer(model)

        # Generate a random input
        x = tf.random.normal(shape=(1, 40, 40, 1), mean=0.0, stddev=0.5)
        # Get the original model's outputs
        y = model(x)

        # Fold the BatchNorm layers
        folded_model = optimizer.fold_batch_norms()

        # Get the folded model's outputs
        y_folded = folded_model(x)

        # Check the outputs
        distance = np.linalg.norm(y - y_folded)

        assert (
            distance < TestBatchNormFolding.TOLERANCE
        ), f"Outputs are not the same. Got distance {distance}"
