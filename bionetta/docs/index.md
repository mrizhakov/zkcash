# Welcome to Bionetta Docs
<h1 align="center">
  <br>
  <a href="#"><img src="images/logo.svg" alt="Bionetta" width="250"></a>
  <br>
  Bionetta
  <br>
</h1>

<h4 align="center">Ultimate Client-Side ZKML Prover on top of <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a>.</h4>

<h5 align="center"> <a href="report.pdf"> [Read the Technical Report] </a> </h5>

<p align="center">
  <a href="#raising_hand-about">About</a> •
  <a href="#test_tube-minimal-example">Example Usage</a> •
  <a href="#studio_microphone-learning-hub">Learning Hub</a>
</p>

> [!IMPORTANT]
>
> This project is **under heavy development** and is **not fully stable** yet.
> Despite that, we are happy to share the current state of the project with you
> where you can find the core functionality and the basic examples. We are
> constantly working on improving the library and adding new features, so stay tuned for updates.

# :raising_hand: About {: #raising_hand-about }

**Bionetta** :herb: is a client-side ZKML prover, currently built on top of TensorFlow. With **Bionetta** :herb:, you can prove a variety of statements of neural network execution:

- :white_check_mark: Maybe you want your users to prove that they can authenticate with their
_biometrics_ without revealing their _biometric data_?
- :white_check_mark: Maybe your neural network can perform the credit scoring and the user
wants to prove that they are creditworthy without revealing their _credit history_?
- :white_check_mark: Or maybe you want your users to submit the proof that the message or a photo
is neither  (possibly wspam nor fakeith hiding the content of the message/photo)?

In such cases, **Bionetta** :herb: is your perfect choice! With it, you can generate proof of the neural network execution in tens of seconds, even on your mobile phone :iphone:, with easy on-chain verification (e.g., Ethereum). According to our benchmarks, the solution surpasses all current alternatives in the ZKML zoo.

> [!WARNING]
>
> Despite astounding performance, the library is **not** intended for 
> **private model weights** + **public input** proving. The library is designed only for **public model weights** + **private/public input** proving!

## :key: Key Features

- **Fast**: The fastest ZKML client-side prover on the market, with a proof generation time of less than 10 seconds even for a 1M parameter model.
- **Lightweight**: We are using
[_Groth16_](https://eprint.iacr.org/2016/260)-based provers (such as the
original Groth16 and customly written
[_UltraGroth_](https://hackmd.io/@Merlin404/Hy_O2Gi-h)), so the proof sizes are
constant and small (less than 1KB). Besides, the verification keys are typically
insignificant, so you can easily deploy them on-chain.
- **Easy to Use**: The library is built on top of TensorFlow, so you can use it with your existing models. You can also use the custom layers and callbacks to build your own architecture.

> [!NOTE]
>
> In the future releases, we are planning to add support for [PyTorch](https://pytorch.org/) as well.

## :racing_car: Key Benchmarks

Below, we specify the benchmarks for the following MNIST model with 2M
parameters and 6000 ReLU activations:

```python
def MNISTModel(inputs: tf.Tensor) -> tf.Tensor:
    """
    A simple network with ReLUs and Dense layers.
    """

    x = Flatten()(inputs)
    x = Dense(2000, activation="relu")(x)
    x = Dense(60, activation="relu")(x)
    x = Dense(4000, activation="relu")(x)
    outputs = Dense(10)(x)
    return outputs
```

To see the full comparison (including compilation time), you can read our
[Technical Report](report.pdf). We use
[_UltraGroth_](https://hackmd.io/@Merlin404/Hy_O2Gi-h) as the base proving
system but note that Bionetta is compatible with any proving backend, supporting
R1CS.

| Metric                | Bionetta (UltraGroth)  | [ddkang/zkml](https://github.com/ddkang/zkml) | [EZKL](https://ezkl.xyz/) | [deep-prove](https://github.com/Lagrange-Labs/deep-prove) | [zkCNN](https://github.com/TAMUCrypto/zkCNN) |
|---------------------- |------------------------|----------------------|---------------------|------------|------------|
| Proof Size (KB)       | **0.88**               | 5.05                 | 127.0               | 8002.16    | 23.25      |
| Verification Key (MB) | **0.004 MB**           | 2.60                 | 4.10                | 0          | –          |
| Proving Key (GB)      | **0.20**               | 16.10                | 8.30                | 0          | –          |
| Proving Time (s)      | **3.05**               | 1100                 | 1310                | 3.66       | 3.45       |
| Verification Time (s) | **0.010**              | 0.012                | 5.40                | 0.522      | 1.00       |
| RAM Usage (GB)        | **0.27**               | 39.95                | 21.15               | 1.61       | 1.00       |


_For testing, we used an Intel Xeon E5-2665 (16 threads) CPU with 350 GB of RAM running Ubuntu 24.04.2 LTS (x86_64)._

## :teacher: How to Install? {: #teacher-how-to-install }

**Recommended Python version**: 3.11. One might also use 3.10, but other than 3.10 and 3.11, the library has not been tested and might not work properly.

Install `python3.11`. Then, install all the dependencies by running the bash script (do not use a
`sudo` privilege: otherwise, the `homebrew` will reject the installation):

```bash
bash install.sh
```

Or as a Python package:

``` bash
python3.11 -m venv venv
source venv/bin/activate
pip install [path_to_bionetta_repo]

# This one is necessary to load child repositories into the library.
bash "$(find $(python3.11 -c "import site; print(site.getsitepackages()[0])") -name tf_bionetta -type d)/repo-install.sh"
```

> [!TIP]
> If, for some reason, your NPM requires `sudo` privileges, you can address [this Stack Overflow question](https://stackoverflow.com/questions/16151018/how-to-fix-npm-throwing-error-without-sudo) to fix the issue.

You are ready to go. Test the installation by running the following command:

```bash
python3.11 -m examples.simple_mnist
```

This command runs the full compilation process for the basic MNIST model.

## :leafy_green: Supported Layers

| Layer Name | Notes | Implemented |
|------------|-------|-------------|
| `Dense` | Fully connected layer  | :white_check_mark: |
| `Conv2D`, no activation | Convolutional layer _with no activation function_ (using activation is very costly in R1CS) | :white_check_mark: |
| `DepthwiseConv2D`, no activation | Depthwise convolutional layer _with no activation function_ (using activation is very costly in R1CS) | :white_check_mark: |
| `Conv`s (with activations) | Convolutional/Depthwise layers _with activation function_ | :white_check_mark: |
| `EDLightConv2D` | Custom R1CS-friendly convolution | :white_check_mark: |
| `SEHeavyBlock` | Modified [Squeeze-and-Excitation](https://arxiv.org/pdf/1709.01507) Block. Very R1CS-friendly. | :white_check_mark: |
| Skip Connections | Residual Blocks. Might be unstable. | :white_check_mark: |
| `Attention` | Attention Layer | :x: |
| `Batch Normalization` | Batch normalization layer | :white_check_mark: |
| `AvgPooling2D` | Average pooling layer | :white_check_mark: |
| `SumPooling2D` | Sum pooling layer (average pooling without averaging) | :white_check_mark: |
| `GlobalAvgPooling2D` | Global average pooling layer | :white_check_mark: |
| `Flatten` | Flatten layer | :white_check_mark: |
| `L2UnitNormalization` | L2 normalization layer (typically used in feature learning approaches) | :white_check_mark: |
| `ReLU` | Rectified Linear Unit activation function | :white_check_mark: |
| `LeakyReLU` | Leaky Rectified Linear Unit activation function. Supported only  for $\alpha = 2^{-s}$ | :white_check_mark: |
| `ReLU6` | Rectified Linear Unit activation function with upper bound | :white_check_mark: |
| `HardSigmoid` | Hard Sigmoid activation function | :white_check_mark: |
| `HardSwish` | Hard Swish activation function | :white_check_mark: |


# :test_tube: Minimal Example {: #test_tube-minimal-example }

Here is the minimal example of how to use the library:

```python
import numpy as np
import tensorflow as tf
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


# 1. Load the MNIST dataset first
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension
y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test, 10) # One-hot encode the labels

# 2. Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64),
    tfb.layers.ShiftReLU(5), # LeakyReLU with alpha=1/(2**5)=1/32
    tf.keras.layers.Dense(10)
], name='simple_mnist_model')

# 3. Wrap the model architecture with our custom BionettaModel class
proving_backend = ProvingBackend.GROTH16(15)

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
model.compile_circuits(
    path='./examples/simple_mnist_circuits',
    test_input=test_input,
    save_weights=True,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM,
    # Adds the XCFramework generation script, which can be used in iOS projects
    rust_witness_generator_options={
        RustGeneratorOptions.RUST_WITNESS_GENERATOR_XCFRAMEWORK_SCRIPTS_KEY: True
    }
)

# 6. Init your architecture from compiled dir
loaded_model = tfb.BionettaModel.load_from_compiled_folder('./examples/simple_mnist_circuits', verbose=2)
test_input = X_test[np.random.randint(len(X_test))]

# 7. Prove and verify your input. Simple mnist should be previously compiled!!!
proof_dir = './examples/simple_mnist_circuits/proof'

proof = loaded_model.prove(
    input=test_input,
    target_dir=proof_dir,
)

print('The resultant proof:', proof)

assert loaded_model.verify(proof_dir=proof_dir), "model verification failed"
```


# :studio_microphone: Learning Hub {: #studio_microphone-learning-hub }

## :writing_hand: Blogs

- [Bionetta: Ultimate Client-Side ZKML. Technical Overview](https://hackmd.io/@rarimo/SkZB2zUxex)
- [Implementing Lookups in Groth: The second secret ingredient to reduce Bionetta’s constraints](https://hackmd.io/@rarimo/HJjHOUjGge)

## :scroll: Papers

- [Bionetta: Efficient Client-Side Zero-Knowledge Machine Learning Proving. Technical Report](report.pdf)
- [Bionetta: Efficient Client-Size Zero-Knowledge Machine Learning with Constant Verification Complexity. Paper](paper.pdf)
