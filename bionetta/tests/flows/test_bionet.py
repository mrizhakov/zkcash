"""
Module for testing ZKBioNet architecture in TensorFlow.
"""

from __future__ import annotations

import os
import unittest

import tensorflow as tf

from tf_bionetta.applications.bionet.v1 import BioNetV1


class BioNetV1TestCase(unittest.TestCase):
    """
    Test case for the BioNetV1 architecture.
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable irritating warnings

        # Simply use the first layer of the ZKBioNetV2 architecture
        self.model = BioNetV1(output_size=92)


    def testBioNetV1Launch(self) -> None:
        """
        Tests whether the BioNetV2 model can be launched.
        """

        # Generate a random input
        x = tf.random.normal(
            shape=(32, 40, 40, 3), mean=0.0, stddev=0.5
        )  # Batch of 32 grayscale images of size 192x192x1
        y = self.model.predict(x)

        assert y.shape == tf.TensorShape((32, 92)), "Output shape mismatch"
