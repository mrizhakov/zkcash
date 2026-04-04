"""
Module for testing the hyperparameters of the training process.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from tf_bionetta.hyperparameters import TrainingHyperparameters


class TrainingHyperparamsTestCase(unittest.TestCase):
    """
    Test case for the Training Hyperparameters.
    """

    def testInitialize(self) -> None:
        """
        Test the initialization of the hyperparameters.
        """

        test_hyperparams = TrainingHyperparameters(
            {
                "meta": {"name": "test", "version": "1"},
                "learning_rate": 1e-4,
                "batch_size": 32,
            }
        )

        print(test_hyperparams.meta.raw())

        try:
            test_hyperparams.learning_rate
        except AttributeError:
            raise AttributeError("Learning rate not found in the hyperparameters")

        assert (
            test_hyperparams.batch_size == 32
        ), "Batch size not found in the hyperparameters"

        try:
            test_hyperparams.meta.name
        except AttributeError:
            raise AttributeError(
                f"Name not found in the meta section of the hyperparameters. Meta: {test_hyperparams.meta}"
            )

    def testBioNetInitialize(self) -> None:
        """
        Tests the complex initialization of the hyperparameters for the BioNet model.
        """

        hyperparams = TrainingHyperparameters(
            {
                "meta": {"name": "BioNet", "version": "3", "subversion": "7"},
                "input_shape": [56, 56, 1],
                "candidates_num": 412,
                "batch_size": 216,
                "steps_per_epoch": 100,
                "kernel_initializer": "he_normal",
                "epochs": 50,
                "margin": 0.4,
                "learning_rate": 3e-2,
                "final_learning_rate": 1e-4,
                "embedding_size": 92,
            }
        )

        assert (
            hyperparams.meta.name == "BioNet"
        ), "Name not found in the meta section of the hyperparameters"
        assert hyperparams.input_shape == [
            56,
            56,
            1,
        ], "Input shape not found in the hyperparameters"
        assert (
            hyperparams.candidates_num == 412
        ), "Candidates number not found in the hyperparameters"

    def testSaveLoad(self) -> None:
        """
        Test the saving and loading of the hyperparameters from a JSON file.
        """

        temp_folder = Path("temp")
        hyperparams_path: Path = temp_folder / Path("hyperparams.json")

        test_hyperparams = TrainingHyperparameters(
            {"learning_rate": 1e-4, "batch_size": 32}
        )

        test_hyperparams.save(hyperparams_path)
        loaded_hyperparams = TrainingHyperparameters.from_json(hyperparams_path)

        # Remove the file after testing
        hyperparams_path.unlink()
        temp_folder.rmdir()

        assert (
            loaded_hyperparams.batch_size == test_hyperparams.batch_size
        ), "Batch size not found in the loaded hyperparameters"
