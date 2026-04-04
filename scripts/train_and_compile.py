import os
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# tf_bionetta requires explicitly registered custom layers
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator, OptimizationLevel
from tf_bionetta.specs.target import TargetPlatform

def main():
    print("1. Synthesizing random face dataset to speed up compilation...")
    # Synthesize data for a Micro-FaceNet to bypass slow downloads and PIL errors.
    # We create 100 images of size 10x10x3 across 5 dummy "people/classes".
    n_samples, h, w, c = 100, 10, 10, 3
    num_classes = 5
    
    # Generate random images and labels
    X = np.random.uniform(0.0, 1.0, size=(n_samples, h, w, c)).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(n_samples,))
    y = tf.keras.utils.to_categorical(labels, num_classes)
    
    print(f"Generated {n_samples} dummy images of shape {h}x{w}x{c}. Classes: {num_classes}")

    # 2. Build the Micro-FaceNet model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10, 10, 3)),
        tf.keras.layers.Conv2D(2, (3,3), strides=(2,2), padding='valid', activation='linear'),
        tfb.layers.ShiftReLU(2),
        tf.keras.layers.Flatten(),
        # The embedding layer
        tf.keras.layers.Dense(4, activation='linear', name='embedding_layer'),
        tfb.layers.ShiftReLU(4),
        # Final classification layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='micro_facenet')

    model.summary()

    # 3. Train
    print("\n3. Training model (should take ~2 minutes on M2)...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    # 4. Extract ONLY the Embedding Model
    print("\n4. Pruning model to output pure Face Embeddings for ZK...")
    # Bionetta will compile the mathematical mapping from photo -> 16 numbers
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
    
    # 5. Compile to ZK using Bionetta
    proving_backend = ProvingBackend.GROTH16()
    bionetta_model = tfb.BionettaModel(embedding_model, verbose=1)
    
    # Dry run constraints output to log
    print("Checking constraints count...")
    bionetta_model.constraints_summary(proving_backend, linear_ops=False)

    print("\n5. Generating ZK Circuits (This involves powersOfTau and takes ~15-30 mins)...")
    test_input = X[0]
    gen_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../compiled_circuit'))
    
    # Compile exclusively for the WEB so we get .wasm and .zkey
    bionetta_model.compile_circuits(
        path=gen_folder,
        test_input=test_input,
        save_weights=True,
        proving_backend=proving_backend,
        target_platform=TargetPlatform.WEB,
        witness_backend=WitnessGenerator.DEFAULT,
        optimization_level=OptimizationLevel.O1
    )
    
    print("\n✅ Done! The .wasm, .zkey, and Verifier.sol artifacts are ready in:", gen_folder)

if __name__ == '__main__':
    main()
