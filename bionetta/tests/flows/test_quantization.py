"""
Unit tests for the quantization module
"""

from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from tf_bionetta.save.quantization import ModelQuantizer


class TestModelQuantizer(unittest.TestCase):
    """
    Unit tests for the ModelQuantizer class
    """

    def setUp(self) -> None:
        """
        Set up the test environment
        """

        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(32,)),
                tf.keras.layers.Dense(10, input_shape=(10,)),
                tf.keras.layers.Dense(1),
            ],
            name="test_quantization_model",
        )
        self.quantizer = ModelQuantizer(self.model)

    def test_form_quantized_model(self) -> None:
        """
        Test the form_quantized_model method
        """

        quantized_model = self.quantizer.form_quantized_model()
        self.assertIsInstance(
            quantized_model, bytes, "The quantized model is not in the bytes format"
        )

    def test_save(self) -> None:
        """
        Test the save method
        """

        path = Path("test.tflite")
        self.quantizer.save(path)
        self.assertTrue(path.exists())
        path.unlink()

    def test_inference_proximity(self) -> None:
        """
        Test the proximity of the inference results of the original and the quantized models
        """

        # Generate the data
        X = np.random.rand(100, 32)
        X = X.astype(np.float32)

        # Generate the predictions
        original_outputs = self.model.predict(X)

        # Quantize the model, save it, and load it
        self.quantizer.save("proximity_test.tflite")
        loaded_model = tf.lite.Interpreter(model_path="proximity_test.tflite")
        loaded_model.allocate_tensors()

        # Get the input and output tensors
        input_details = loaded_model.get_input_details()
        output_details = loaded_model.get_output_details()

        # Check the input and output shapes
        assert tuple(input_details[0]["shape"]) == tuple(
            [1, 32]
        ), "The input shape is incorrect"
        assert tuple(output_details[0]["shape"]) == tuple(
            [1, 1]
        ), "The output shape is incorrect"

        # Check the proximity of the predictions
        quantized_predictions = np.zeros((100, 1))
        for i in range(100):
            loaded_model.set_tensor(input_details[0]["index"], X[i : i + 1])
            loaded_model.invoke()
            quantized_predictions[i] = loaded_model.get_tensor(
                output_details[0]["index"]
            )

        # Remove the test file
        Path('proximity_test.tflite').unlink()
