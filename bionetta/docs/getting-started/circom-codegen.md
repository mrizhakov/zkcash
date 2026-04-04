# Circom Code Generation

Let's examine the `compile_circom_circuit` function, which is called in the first step. We create a `CircuitGenerator` and generate all circom code inside it. The weights are created inside the `build` method.

<pre class="language-python"><code class="lang-python">CircuitGenerator = getattr(module, "CircuitGenerator")
PROVE_BACKEND = getattr(module, "ProvingBackend")

<strong>generator = CircuitGenerator(
</strong>    architecture=architecture,
    weights=weights,
    circom_folder=self.CIRCOM_REPO,
    proving_backend=PROVE_BACKEND.create(engine.proving_backend),
    verbose=VerboseMode.DEBUG.value
)
generator.build(target_folder=self._circom_dir)
</code></pre>

After executing this part of the code, we get the weights files, `simple_mnist_model.rs` and `weights.circom`. The main file is `simple_mnist_model.circom`.

The main file was compiled using the following command: `circom simple_mnist_model_circom/simple_mnist_model.circom --r1cs --sym --wasm --O1`. The `--wasm` option creates `simple_mnist_model_js` directory, which is needed to run the web application. It is not added in cases described in the `Bionetta Wrapper` section.

<figure><img src="../../images/image (7).png" alt=""><figcaption></figcaption></figure>
