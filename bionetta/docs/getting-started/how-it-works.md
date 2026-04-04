# How it works?

1. You create the _Keras_ model and either use it pre-trained or train it from scratch. The former is more desirable, as in this case you can design the model architecture in a more flexible way.
2.  You wrap the model with the `BionettaModel` class, which will

    * Check the model architecture and print the summary of the constraints via the `constraints_summary` method.
    * Compile the model with the `compile_circuits` method, which will generate the proof circuits for the model.
    * Save the model weights and circuits to a specified directory.

    Code example:

```python
# 3. Wrap the model architecture with our custom BionettaModel class
proving_backend = ProvingBackend.GROTH16()

model = tfb.BionettaModel(model, verbose=2)
model.constraints_summary(proving_backend)
test_input = X_test[np.random.randint(len(X_test))]

# 4. Compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
    epochs=1,
    validation_data=(X_test, y_test),
    validation_split=0.2
)

# 5. Compile the model
gen_folder = './examples/simple_mnist_circuits'
model.compile_circuits(
    path=gen_folder,
    test_input=test_input,
    save_weights=True,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM
)
```

That is pretty much it! Nonetheless, under the hood, the library does a lot of heavy lifting, such as:

* **Model Architecture Check**: The library checks the model architecture and prints the summary of the constraints. It also specifies the layers with the most constraints.
* **Circom Code Generation**: The library generates the Circom code for the model, which is then compiled to the `R1CS` format. The Circom code is generated using the `circom` command-line tool, which is a part of the Circom library.
* **Rust Witness Generator Generation**: The library generates the Rust code for the witness generator, which is then compiled to a binary. The Rust code is generated using the `rust` command-line tool, which is a part of the Rust library. Note that our Rust code is specifically designed to be faster than the native Circom solution.
* **Witness Generation and Proving**: The library generates the witness for the model using the Rust binary, which is then used to generate the proof. The proof is generated using the `snarkjs` command-line tool, which is a part of the SnarkJS library.

All the processes are visualized in the following diagram:
<figure><img src="../../../images/architecture.svg" alt=""><figcaption></figcaption></figure>
