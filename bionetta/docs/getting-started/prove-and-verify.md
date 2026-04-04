# Prove and Verify

First, we load the required model in the `simple_mnist.py` file to test the `load_from_compiled_folder` function. We can use the `model` variable instead of the `loaded_model`.

```python
loaded_model = tfb.BionettaModel.load_from_compiled_folder(gen_folder, verbose=2)
test_input = X_test[np.random.randint(len(X_test))]

# 7. Prove and verify your input. Simple mnist should be previously compiled!!!
proof_dir = f'{gen_folder}/proof'

proof = loaded_model.prove(
    input=test_input,
    target_dir=proof_dir,
)

print('The resultant proof:', proof)

assert loaded_model.verify(proof_dir=proof_dir), "model verification failed"
```

The next two files are created based on the `.zkey` and `.(u)wtns` files.

<figure><img src="../../images/image (2).png" alt=""><figcaption></figcaption></figure>

Use the build files generated after building the `UltraGroth` repository. Then, start verifying using the `verification_key.json` (or the `_vkey.json` file for `UltraGroth`), the `..._public.json` and the `..._proof.json` files.

That's it! This proves any TensorFlow model whose layers are supported by the Bionetta. Now, you can try to prove your own models!
