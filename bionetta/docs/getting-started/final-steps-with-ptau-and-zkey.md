# Final steps with PTAU and ZKey

Next, a `.h` file is generated for mobile devices. This file contains bindings to the main rust structures because mobile devices interact more easily with C code than with rust.
For example, the following `header.h` file is created for `simple_mnist`:

```c
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct WtnsBytes {
    uint8_t *data;
    uint64_t len;
    char *error;
    uint32_t error_len;
} WtnsBytes;

#ifdef __cplusplus__
extern "C" {
#endif

struct WtnsBytes calc_wtns_simple_mnist_model(const uint8_t *input_ptr, uint64_t input_len);
void wtns_free(struct WtnsBytes *_self);

#ifdef __cplusplus__
}
#endif
```

After that, the `trusted_setup` of the appropriate degree, which is calculated based on the exact number of constraints obtained after rust code generation processing. If a `trusted_setup` of the same degree or higher already exists, it is used instead of downloading another one.

```python
trusted_setup = None if trusted_setup is None else os.path.abspath(Path(trusted_setup))
if trusted_setup is None or not trusted_setup.exists():
    ptau_kwargs = {
        'target_directory': ptau_dir,
        'constraints_number': 1  # Plug
    }
    trusted_setup = PowersOfTauLoader.form_ptau_file_path(**ptau_kwargs)
    steps.update({
        'Downloading powers of tau': lambda progress: self._powers_of_tau.download(**ptau_kwargs, progress=progress)
    })
trusted_setup = os.path.abspath(trusted_setup)
```

The final step generates a `.zkey` file using different methods for `UltraGroth` and `Groth16`.

For `UltraGroth`, the `ultragroth-snarkjs` repository is used with the command `npm run test:ultragroth`, along with `.r1cs` input files, `.sym` indexes in `JSON` format, and `ptau`. This creates the files `simple_mnist_model_final.zkey`, `simple_mnist_model_indexes.json`, and `simple_mnist_model_vkey.json`.&#x20;

For `Groth16`, standard `snarkjs` is used, resulting in `simple_mnist_model.zkey`, `verification_key.json`, and `verifier.sol`.
