"""
Package for testing the constraints calculator package
"""

from __future__ import annotations

import unittest
import tensorflow as tf

from tf_bionetta.constraints import ModelConstraintsCalculator
from tf_bionetta.constraints.types.activations import ActivationOps
from tf_bionetta.specs.backend_enums import ProvingBackend
from tf_bionetta.layers import (
    SEHeavyBlock,
    SELightBlock,
    EDLight2DConv,
    EDHeavy2DConv,
)
from tf_bionetta.layers.experimental import (
    EDConv2D,
    Gated2DConv,
)

from tf_bionetta.logging import create_logger, VerboseMode


class TestConstraintsCalculator(unittest.TestCase):
    """
    Test case for the constraints calculator.
    """

    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        self.logger = create_logger(mode=VerboseMode.WARNING)

    def testInvalidLeakyReLUAlphaGroth16(self) -> None:
        """
        Tests whether the constraint calculator recognizes an invalid LeakyReLU
        alpha value, which is either negative or not the power of two for 
        the UltraGroth backend.
        """

        test_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(40, 40, 3)),
                tf.keras.layers.Conv2D(32, 3, padding="same"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(300),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )
        backend = ProvingBackend.ULTRAGROTH(limb_size=15, precision_multiplicity=1)
        calculator = ModelConstraintsCalculator(
            model=test_model,
            backend=backend,
            logger=self.logger
        )
        calculator.print_constraints_summary()
        
        # Validate that the constraints were validated properly
        layers = test_model.layers
        valid_constraints = {
            layers[0]: (0, False),
            layers[1]: (40 * 40 * 32 * ActivationOps.RELU.cost(backend), False),
            layers[2]: (0, False),
            layers[3]: (0, False),
            layers[4]: (300 * ActivationOps.UNSUPPORTED_LEAKY_RELU.cost(backend), True),
        }
        for layer, constraints in valid_constraints.items():
            self.assertEqual(calculator.layer_constraints[layer], constraints)


    def testCustomLayersConstraintCalculator(self) -> None:
        """
        Tests whether the constraint calculator correctly calculates the count of constraints for custom layers
        """

        test_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(5, 5, 1)),
                SEHeavyBlock(hidden_units=5, kernel_size=5),
                SELightBlock(hidden_units=5),
                EDLight2DConv(
                    hidden_units=5, kernel_size=5, channels=5, kernel_output_size=5
                ),
                EDHeavy2DConv(
                    hidden_units=5, kernel_size=5, channels=5, kernel_output_size=5
                ),
                EDConv2D(
                    hidden_layer_size=5,
                    kernel_size=5,
                    hidden_layer_channels=5,
                    output_layer_size=5,
                    output_layer_channels=5,
                ),
                Gated2DConv(kernel_size=5, channels=5, squeeze_factor=1),
                tf.keras.layers.Dense(10),
            ]
        )

        calculator = ModelConstraintsCalculator(
            model=test_model,
            backend=ProvingBackend.GROTH16(15),
            logger=self.logger
        )
        calculator.print_constraints_summary()

        # Validate that the constraints were validated properly
        layers = test_model.layers
        valid_constraints = {
            layers[0]: 1300,
            layers[1]: 1300,
            layers[2]: 1275,
            layers[3]: 6375,
            layers[4]: 31875,
            layers[5]: 1400,
        }

        for layer, constraints in valid_constraints.items():
            self.assertEqual(calculator.layer_constraints[layer][0], constraints)
