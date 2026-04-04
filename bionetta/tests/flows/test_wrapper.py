"""
Test the core functionality of the BionettaModel wrapper.
"""

from __future__ import annotations

from pathlib import Path
import unittest
import shutil

from tf_bionetta import BionettaModel
from tf_bionetta.specs.backend_enums import ProvingBackend
from tf_bionetta.applications.bionet.v1 import BioNetV1
from tf_bionetta.logging import(
    create_logger,
    VerboseMode
)


class TestBionettaWrapper(unittest.TestCase):
    """
    Test case for the Batch Normalization Folding.
    """

    def setUp(self) -> None:
        """
        Tests the saving of the model.
        """

        self.model = BioNetV1(output_size=92)
        self.logger = create_logger(mode=VerboseMode.DEBUG)
        self.wrapped_model = BionettaModel(self.model, verbose=0)

    def testSaver(self) -> None:
        """
        Test the saving of the model.
        """

        temp_folder: Path = Path("./saver_test")
        self.wrapped_model.save(temp_folder)
        # Remove folder after testing
        shutil.rmtree(temp_folder)

    def testConstraints(self) -> None:
        """
        Test the constraints calculation.
        """

        # If this does not raise an error, the test is successful
        self.wrapped_model.constraints_summary(ProvingBackend.GROTH16())
