"""
Full integration test for Bionetta framework. It works as follows:
- It defines the model architecture
- Trains the model on the MNIST dataset
- Compiles the circit for the model
- Saves the model and the circuit
- Generates the input for the circuit
- Checks the output of the circuit
- Verifies that the circuit's output is the same as the model's output
"""
from __future__ import annotations

import os
import json
import subprocess
import unittest

import numpy as np
import tensorflow as tf

import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


class BionettaMNISTFullIntegrationTest(unittest.TestCase):
    """
    Class for testing the full integration of the Bionetta framework
    with a simple MNIST model.
    """
    
    MODEL_NAME = 'simple_mnist_model'
    
    
    def setUp(self) -> None:
        """
        Basic initializations go here
        """

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28,28,1)),
            tf.keras.layers.Flatten(input_shape=(28,28,1)),
            tf.keras.layers.Dense(64),
            tfb.layers.ShiftReLU(5), # LeakyReLU with alpha=1/(2**5)=1/32
            tf.keras.layers.Dense(10)
        ], name=self.MODEL_NAME)
        
    
    def test_fullMNISTIntegration(self) -> None:
        """
        Conducts the whole integration test for the Bionetta framework.
        """
        
        # 1. Load the MNIST dataset first
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Postprocess the data
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = np.expand_dims(X_train, axis=-1) # Add channel dimension
        X_test = np.expand_dims(X_test, axis=-1) # Add channel dimension
        y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode the labels
        y_test = tf.keras.utils.to_categorical(y_test, 10) # One-hot encode the labels
        
        test_input = X_test[np.random.randint(len(X_test))]
        
        # 2. Wrap the model architecture with our custom BionettaModel class
        self.model = tfb.BionettaModel(self.model, ignore_errors=False, verbose=2)
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X_train, y_train, 
            epochs=1, 
            validation_data=(X_test, y_test), 
            validation_split=0.2
        )
        print(self.model.predict(np.expand_dims(test_input,axis=0)))

        # 3. Compile the model
        COMPILED_MODEL_PATH = './compiled_mnist'
        
        try:
            self.model.compile_circuits(
                path=COMPILED_MODEL_PATH,
                test_input=test_input,
                save_weights=True,
                proving_backend=ProvingBackend.GROTH16(15),
                target_platform=TargetPlatform.DESKTOP,
                witness_backend=WitnessGenerator.CUSTOM
            )
        except Exception as e:
            print(f"Error during compilation: {e}")
            raise e
        
        # 4. Generate proof for the test input and verify it
        proof_dir = './compiled_mnist/proof'

        self.model.prove(
            input=test_input,
            target_dir=proof_dir
        )

        assert self.model.verify(proof_dir=proof_dir), "Proof is not valid"

        test_input = X_test[np.random.randint(len(X_test))]

        # 5. Decipher the witness file
        raw_wtns_file_path = os.path.abspath(f"{COMPILED_MODEL_PATH}/{self.MODEL_NAME}.wtns")
        json_wtns_file_path = os.path.abspath(f"{COMPILED_MODEL_PATH}/proof/{self.MODEL_NAME}.json")
        subprocess.run([
            "snarkjs", "wtns", "export", "json",
            os.path.abspath(raw_wtns_file_path), 
            os.path.abspath(json_wtns_file_path),
        ])
        
        # 6. Check that the output of the circuit is the same as the model's output
        CHECK_WTNS_INDEX = 2 # The index of the output in the witness file
        with open(json_wtns_file_path, "r") as file:
            wtns = json.load(file)
            assert wtns[CHECK_WTNS_INDEX] == "1", "Output of the circuit is not 1"
        
        # 7. Init your architecture from compiled dir
        self.model_for_prove = tfb.BionettaModel.load_from_compiled_folder('./compiled_mnist')
        test_input = X_test[np.random.randint(len(X_test))]

        # 8. Prove and verify your input. Simple mnist should be previousely compiled!!!
        proof_dir = './compiled_mnist/proof2'

        self.model_for_prove.prove(
            input=test_input,
            target_dir=proof_dir
        )

        assert self.model_for_prove.verify(proof_dir=proof_dir), "Proof is not valid"
