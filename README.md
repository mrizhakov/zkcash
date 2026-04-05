# ZKcash: Privacy-Preserving Payments via Facial Recognition

Send money to anyone instantly — no wallet required. Powered by the [Bionetta ZKML framework](https://github.com/rarimo/bionetta-js-sdk) from [Rarimo](https://rarimo.com).

## What it does

ZKcash is a completely private, decentralized payment app where:

- **Sender** takes a single photo of the recipient and deposits funds into a stealth address
- **Recipient** unlocks the funds by performing a liveness-checked facial scan on their phone — no wallet or identity verification needed
- **Liveness detection** ensures the recipient is present and authentic, preventing deep-fakes or static image attacks
- **Zero data leakage** — facial images never leave devices, embeddings compared only inside a ZK circuit
- **Stealth addresses** — funds deposited to privacy-preserving addresses whose keys are cryptographically tied to the recipient's facial identity
- **Decentralized trust** — no centralized server, no party ever holds both the sender's photo and recipient's live scan

The system cryptographically proves the sender's photo and recipient's face belong to the same person, fully privately. Zero wallet infrastructure needed to receive payments.

## How it works

### Payment Flow

1. **Sender deposits** — Takes photo of recipient, generates face embedding commitment, creates a stealth address, and locks funds to this stealth address on-chain
2. **Recipient onboards** — No wallet setup required. Simply scans their own face with liveness detection (blink, head turn, etc.) to prove they are a real, present human
3. **ZK proof generation** — Recipient's live embedding is compared to sender's photo inside a ZK circuit, generating a Groth16 proof that proves:
   - Same person (facial match)
   - Real and present (liveness check passed)
   - Authorized recipient (proof tied to their unique facial identity)
4. **Fund unlock** — Recipient submits proof on-chain; smart contract verifies proof and releases funds from stealth address to recipient

### Privacy Architecture

```
SENDER:     Photo → Face Detection → Embedding → Commitment Hash → Stealth Address → On-Chain Deposit
                    (MediaPipe)      (Bionetta)   (Poseidon Hash)   (Commitment)

RECIPIENT:  Live Selfie → Liveness Check → Face Detection → Embedding → ZK Proof Gen → Proof Submit
            (Blink, head   (Anti-deepfake)  (MediaPipe)    (Bionetta) (Circom)       (Claim Tx)
             turn, etc.)

CIRCUIT:    Proves distance(sender_embedding, recipient_embedding) ≤ threshold AND liveness_ok = true
            Binds proof to recipient's unique facial commitment (stealth address key derivation)
            Without revealing either embedding or original images
```

### About the Model & Circuit

The full Bionetta pipeline uses a proprietary TensorFlow model for face embeddings. ZKcash implementation:
1. Trains a 20×20 face embedding model on labeled identity datasets
2. Integrates liveness detection (MediaPipe hand landmarks, face geometry checks, motion analysis)
3. Compiles the combined model + liveness checks to a Circom ZK circuit via Bionetta codegen
4. Generates Groth16 proofs verifiable on-chain

The ZK circuit proves:
- ✓ `distance²(sender_embedding, recipient_embedding) ≤ threshold` (same person)
- ✓ `liveness_check_passed == true` (recipient is real, present, and not a deepfake)
- ✓ `recipient_commitment_hash == stealth_address_commitment` (proof ties to correct recipient)
- ✓ `proof_signer == msg.sender` (no front-running or proof replay)
- ✗ Does NOT reveal either embedding, the original facial images, or recipient's biometric data

## Getting Started

### Prerequisites

- Node.js 18+
- npm
- Hardhat (for local testing)

### Install & Run

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser to access the web app.

### Local Contract Testing

```bash
cd contracts
npm install
npx hardhat test
```

## Training a Production Model

By default, the app ships with a random-weight placeholder model. For real payments, train on labeled identity data:

### Dataset Layout

Create a folder of identity-labeled face images:

```text
datasets/faces/
    alice/
        photo1.jpg
        photo2.jpg
    bob/
        photo1.jpg
        photo2.jpg
```

Each subfolder = one identity. Use 3+ photos per identity for best results.

### Train & Compile to Circuit

```bash
source bionetta_env/bin/activate
python scripts/train_real_20x20_model.py \
    --data-dir ./datasets/faces \
    --output-dir ./compiled_circuit_v20_production \
    --epochs 40 \
    --embedding-dim 16 \
    --batch-size 32 \
    --val-split 0.2
```

This script will:
1. **Train** a 20×20 CNN face embedding model on your data
2. **Calibrate** a recommended match threshold from validation pairs (same-person vs. different-person)
3. **Compile** the model to Bionetta ZK artifacts (Circom circuit + WASM)
4. **Output** recommended threshold to `recommended_threshold.json`

Then:
1. Copy the `.zkey` and `.wasm` files to `public/models/`
2. Update the model URL in `src/web3.ts`
3. Redeploy UI and re-generate verification contracts

## Contract Deployment

Deploy the Solidity contracts to your target network:

```bash
cd contracts
export PRIVATE_KEY=0x...
export RPC_URL=https://...

npx hardhat run scripts/deploy.js --network [chainName]
```

This deploys:
- `Groth16Verifier` — on-chain proof verifier (generated by snarkJS)
- `FaceLockedTransfer` — main payment contract with stealth address support (see [FaceLockedTransfer.sol](contracts/contracts/FaceLockedTransfer.sol))
- Stealth address utilities — derives deterministic, private addresses from facial commitment hashes

## Tech Stack

### Frontend
- **[Vite](https://vitejs.dev/)** — Lightning-fast build tool
- **TypeScript** — Type-safe UI code
- **[@tensorflow/tfjs](https://www.tensorflow.org/js)** — In-browser ML inference
- **[@tensorflow-models/face-detection](https://github.com/tensorflow/tfjs-models/tree/master/face-detection)** — MediaPipe face detection

### ZK & Cryptography
- **[Bionetta](https://github.com/rarimo/bionetta-js-sdk)** — ZKML framework for facial embeddings
- **[@rarimo/bionetta-js-sdk-core](https://www.npmjs.com/package/@rarimo/bionetta-js-sdk-core)** — Proof generation & verification
- **[snarkJS](https://github.com/iden3/snarkjs)** — Groth16 proof utilities
- **[Circom](https://github.com/iden3/circom)** — ZK circuit language

### Smart Contracts
- **Solidity ^0.8.24** — Proof verification contracts
- **Hardhat** — Development, testing, deployment
- **Ethers.js** — Web3 interaction from UI

### Privacy & Security
- All face embeddings computed client-side; original images never uploaded
- ZK circuit proofs never reveal embeddings or photos
- On-chain verification ensures no double-claims
- 30-day refund window for unclaimed deposits

## Security & Privacy Guarantees

### Data Privacy
- ✓ **No centralized server** — all facial embeddings computed in-browser; liveness checks run locally
- ✓ **No image storage** — sender's photo and recipient's selfie never persist anywhere
- ✓ **No database of faces** — embeddings/liveness signals only generated during transaction, never logged
- ✓ **Mathematical privacy** — even the smart contract never sees embeddings, only the ZK proof result
- ✓ **Stealth by default** — funds locked to privacy-preserving addresses, not tied to recipient's public identity

### Transaction Security
- ✓ **Groth16 proof verification** — on-chain verification confirms correct matching and liveness
- ✓ **Liveness detection** — prevents deep-fakes, recorded videos, and static images from being used
- ✓ **Replay protection** — per-lock nonce embedded in proof prevents proof reuse across payments
- ✓ **Recipient binding** — facial commitment derived from recipient's unique biometric; no impersonation possible
- ✓ **Threshold calibration** — model trained to minimize false positives (denying rightful recipients) and false negatives (accepting imposters)

### No-Wallet Onboarding
- ✓ **Zero wallet setup** — Recipients do not need MetaMask, hardware wallets, or seed phrases
- ✓ **Face is the key** — Facial identity directly unlocks funds; keys derived from liveness-checked embedding
- ✓ **Mobile-first** — Full experience on a phone camera; no special hardware required
- ✓ **Inclusive** — Anyone with a face can receive crypto; no KYC or pre-registration

### Known Limitations
- Requires good lighting for face detection
- Liveness detection effectiveness depends on camera quality and network latency
- Currently designed for individual-to-individual transfers (no institutional custody)
- Model accuracy depends on training data diversity (encouraged: diverse ethnicity, age, lighting)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         SENDER FLOW                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Open app → Select recipient photo                         │
│ 2. Extract embedding (client-side, TensorFlow)               │
│ 3. Hash embedding → Create commitment                        │
│ 4. Sign Tx: deposit(commitment, threshold, nonce)            │
│ 5. Lock ETH to FaceLockedTransfer contract                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   [Funds locked on-chain]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       RECIPIENT FLOW                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Receive payment link/ID                                   │
│ 2. Open app on their phone → Scan own face                   │
│ 3. Extract embedding (client-side, TensorFlow)               │
│ 4. Generate ZK proof: do embeddings match? ✓                 │
│ 5. Submit claim(lockId, proof) → On-chain verification       │
│ 6. Contract verifies proof → Release funds to recipient      │
└─────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                            SENDER FLOW                            │
├──────────────────────────────────────────────────────────────────┤
│ 1. Open ZKcash app → Select recipient photo                       │
│ 2. Extract embedding (client-side, Bionetta)                      │
│ 3. Hash embedding → Create facial commitment                      │
│ 4. Derive stealth address from commitment                         │
│ 5. Sign Tx: deposit(commitment, stealth_addr, threshold, nonce)   │
│ 6. Lock ETH to stealth address on FaceLockedTransfer contract     │
│ 7. Share payment link/code with recipient                         │
└──────────────────────────────────────────────────────────────────┘
                             ↓
              [Funds locked in stealth address]
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                        RECIPIENT FLOW                             │
├──────────────────────────────────────────────────────────────────┤
│ 1. Receive payment link/code (no wallet needed)                   │
│ 2. Open ZKcash on their phone → Scan own face                     │
│ 3. Liveness detection: blink, head turn, micro-expressions        │
│ 4. Extract live embedding (client-side, Bionetta)                 │
│ 5. Generate Groth16 ZK proof locally:                             │
│    - Does live face match sender's photo? ✓                       │
│    - Is recipient real & present? ✓                               │
│ 6. Submit claim(proof) → On-chain verification                    │
│ 7. Contract verifies ZK proof → Release funds from stealth addr   │
│ 8. Recipient can now spend funds (no wallet setup required)       │
└──────────────────────────────────────────────────────────────────┘
```

## License

MIT — Built for [ETHGlobal Cannes 2026](https://ethglobal.com)
