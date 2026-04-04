# Rust Code Generation

The Rust generation process is more complex than that of Circom and involves more steps.

1. Create a new Rust project if one does not already exist, and then add the `lib.rs` file.
2. Fill the `Cargo.toml` file with the config from `codegen/templates/witness_generator_config.toml`.  
The code is provided below:
``` toml
[package]
name = "{{ model_name }}"
version = "0.1.0"
edition = "2024"

[dependencies]
witness_generator = { path = "{{ witness_gen_path }}/witness_generator" }
macro_utils = { path = "{{ witness_gen_path }}/macro_utils" }
traits = { path = "{{ witness_gen_path }}/traits" }
serde = { version = "1.0", features = ["derive"] }
bnum = "0.12.1"
serde_json = "1.0"
bincode = "2.0"

[[bin]]
name = "precompute"
path = "src/main.rs"

```
3. Initialize the project structure by pulling in the necessary modules as shown in the picture below. <figure><img src="../../images/image.png" alt=""><figcaption></figcaption></figure>
4. Generate the `architecture.rs` file and fill in the `main.rs` file. Then, obtain the `indexes.bin` file.
5. Add `lib.rs` to the `.toml` file and generate the `.(u)wtns` file (depending on the proving backend), which provide indexes from the `.sym` file. Also generate the `.a`, `.so`, `.d` and `.rlib` dynamic libraries needed for mobile devices. Fill in the value of the witness in the weights file.
6. Ultimately, we obtain a number of constraints based on the `R1CS` file and the UltraGroth lookup tables, and we calculate the precise number of constraints.

As a result, we ended up with this project structure and generated the `simple_mnist_model.wtns` file for `Groth16`.

<figure><img src="../../images/image (8).png" alt=""><figcaption></figcaption></figure>
