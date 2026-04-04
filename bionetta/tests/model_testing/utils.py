import os
import json
import unittest

import numpy as np
import tensorflow as tf

import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


def test_model(
    modelClass: unittest.TestCase,
    compiled_model_path: str,
    proving_backend: ProvingBackend
) -> None:
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
    modelClass.model = tfb.BionettaModel(modelClass.model, ignore_errors=False, verbose=2)
    modelClass.model.summary()
    modelClass.model.constraints_summary(proving_backend)
    modelClass.model.compile(loss='mse', optimizer='adam')
    modelClass.model.fit(X_train, y_train, 
        epochs=1, 
        validation_data=(X_test, y_test), 
        validation_split=0.2
    )
    print(modelClass.model.predict(np.expand_dims(test_input,axis=0)))

    try:
        modelClass.model.compile_circuits(
            path=compiled_model_path,
            test_input=test_input,
            save_weights=True,
            proving_backend=proving_backend,
            target_platform=TargetPlatform.DESKTOP,
            witness_backend=WitnessGenerator.CUSTOM
        )
    except Exception as e:
        print(f"Error during compilation: {e}")
        raise e

    # 4. Generate proof for the test input and verify it
    proof_dir = f'{compiled_model_path}/proof'

    modelClass.model.prove(
        input=test_input,
        target_dir=proof_dir
    )

    assert modelClass.model.verify(proof_dir=proof_dir), "Proof is not valid"

    # 5. Decipher the witness file
    # raw_wtns_file_path = os.path.abspath(f"{compiled_model_path}/{modelClass.MODEL_NAME}.wtns")
    json_wtns_file_path = os.path.abspath(f"{proof_dir}/{modelClass.MODEL_NAME}_public.json")
    # subprocess.run([
    #     "snarkjs", "wtns", "export", "json",
    #     os.path.abspath(raw_wtns_file_path), 
    #     os.path.abspath(json_wtns_file_path),
    # ])

    # 6. Check that the output of the circuit is the same as the model's output
    # CHECK_WTNS_INDEX = 2 # The index of the output in the witness file
    with open(json_wtns_file_path, "r") as file:
        wtns = json.load(file)
        assert wtns[1] == "1", "Output of the circuit is not 1"

    # 7. Init your architecture from compiled dir
    modelClass.model_for_prove = tfb.BionettaModel.load_from_compiled_folder(compiled_model_path)
    test_input = X_test[np.random.randint(len(X_test))]

    # 8. Prove and verify your input. Model should be previousely compiled!!!
    proof_dir = f'{compiled_model_path}/proof2'

    modelClass.model_for_prove.prove(
        input=test_input,
        target_dir=proof_dir
    )

    assert modelClass.model_for_prove.verify(proof_dir=proof_dir), "Proof is not valid"