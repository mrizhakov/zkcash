# Bionetta Face Compare

Privacy-preserving face comparison powered by the [Bionetta ZKML framework](https://github.com/rarimo/bionetta-js-sdk) from [Rarimo](https://rarimo.com).

## What it does

Upload two face images and compare them client-side — no data leaves your device.

1. **Face Detection** — MediaPipe TF.js detects faces in uploaded images
2. **Feature Extraction** — Extracts 128-dimensional face embedding vectors
3. **Distance Calculation** — Cosine similarity determines if faces match
4. **ZK Proof** _(coming soon)_ — Bionetta generates a Groth16 ZK-SNARK proof verifiable on-chain

## Architecture

This app mirrors the [Bionetta JS SDK](https://github.com/rarimo/bionetta-js-sdk) architecture:

```
Image Upload → FaceDetectorService → getFaceFeatureVectors() → Distance Calc → Result
                 (MediaPipe)           (TF.js Model)           (Cosine Sim)
```

### About the Model

The full Bionetta pipeline uses a proprietary TensorFlow model whose weights are not yet public.
This demo uses a statistical feature extraction approach (color histograms + spatial gradients + LBP-inspired features) as a substitute.

When the Bionetta model becomes available:
1. Replace `generateEmbedding()` in `src/face-compare.ts` with `model.predict()`
2. Load the model via `tf.loadGraphModel(MODEL_URL)`
3. The rest of the pipeline (detection, cropping, comparison) remains identical

### ZK Proof Pipeline

The ZK proof components (`ProverWorker`, `WitnessWorker`) from `@rarimo/bionetta-js-sdk-core` are production-ready but require:
- `.zkey` file (proving key from Bionetta's trusted setup)
- `.wasm` file (compiled Circom circuit)
- Contact the [Rarimo team](https://rarimo.com) for early access

## Getting Started

### Prerequisites

- Node.js 18+
- npm

### Install & Run

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Train A Stronger ZK Model (Recommended)

For production-like accuracy, train with real labeled identities instead of random demo weights.

### Dataset Layout

Create a folder structure like:

```text
datasets/faces/
    person_1/
        img1.jpg
        img2.jpg
    person_2/
        img1.jpg
        img2.jpg
```

Each subfolder is one identity.

### Run Training + Circuit Compilation

```bash
source bionetta_env/bin/activate
python scripts/train_real_20x20_model.py \
    --data-dir ./datasets/faces \
    --output-dir ./compiled_circuit_v20_real \
    --epochs 40 \
    --embedding-dim 16
```

The script will:
1. Train a stronger 20x20 face embedding model on real identities.
2. Calibrate a recommended threshold from same/different validation pairs.
3. Compile the embedding model to Bionetta ZK artifacts.
4. Write threshold stats to `recommended_threshold.json` in the output directory.

## Tech Stack

- **[Vite](https://vitejs.dev/)** — Build tool
- **TypeScript** — Type safety
- **[@tensorflow/tfjs](https://www.tensorflow.org/js)** — ML inference in-browser
- **[@tensorflow-models/face-detection](https://github.com/tensorflow/tfjs-models/tree/master/face-detection)** — MediaPipe face detection
- **[@rarimo/bionetta-js-sdk-core](https://www.npmjs.com/package/@rarimo/bionetta-js-sdk-core)** — ZK proof architecture reference

## License

MIT — Built for [ETHGlobal Cannes 2026](https://ethglobal.com)
