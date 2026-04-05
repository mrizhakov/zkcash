import { ethers } from "ethers";

// ABI for FaceLockedTransfer (Explicit 256-bit sync for Ethers v6)
const FaceLockedTransferABI = [
  "function deposit(uint256 faceCommitment, uint256 threshold, uint256 nonce) external payable",
  "function claim(uint256 lockId, uint256[2] _pA, uint256[2][2] _pB, uint256[2] _pC, uint256[5] _pubSignals) external",
  "function getLock(uint256 lockId) external view returns (address sender, uint256 amount, uint256 faceCommitment, uint256 threshold, uint256 nonce, bool claimed, uint256 createdAt)",
  "event FundsLocked(uint256 indexed lockId, address indexed sender, uint256 amount, uint256 faceCommitment, uint256 threshold)",
  "event FundsClaimed(uint256 indexed lockId, address indexed recipient, uint256 amount)",
];

const CONTRACT_ADDRESS = "0x06598187677A261d54fd89ECb5497E2295C3A6Fb"; // patched deployment - Base Sepolia
const CIRCUIT_WASM_PATH = "/model_v10.wasm";
const CIRCUIT_ZKEY_PATH = "/model_v10_final.zkey";
const CIRCUIT_ASSET_TAG = "v10";
const V10_FEATURE_OFFSET = 1470;

export function getRuntimeConfig() {
  return {
    contractAddress: CONTRACT_ADDRESS,
    wasmPath: CIRCUIT_WASM_PATH,
    zkeyPath: CIRCUIT_ZKEY_PATH,
    assetTag: CIRCUIT_ASSET_TAG,
    featureOffset: V10_FEATURE_OFFSET,
  };
}

// ─── NeuroSync: 20x20 High-Fidelity CNN Weights ─────────────────────────────

const CONV_W = [
  [
    [
      [-0x1613, 0x3b14],
      [0x23f7, -0x342d],
      [0x3285, 0x318f],
    ],
    [
      [0x1326, -0x384d],
      [0xfc1, 0x419],
      [0x11eb, -0x14e2],
    ],
    [
      [0x3ac2, 0x1342],
      [0x33f4, 0x195d],
      [0x137, -0x30e],
    ],
  ],
  [
    [
      [-0x15a3, 0x3733],
      [-0x2a2, 0x10a7],
      [0x33ab, 0xb68],
    ],
    [
      [-0x3f51, 0x1e79],
      [0xcde, 0x1e8c],
      [0x3552, 0x6ee],
    ],
    [
      [-0x35f5, -0x3b7e],
      [0x1405, 0x1999],
      [-0x80, -0x359e],
    ],
  ],
  [
    [
      [-0x2df, 0x17c8],
      [-0x3579, -0x1c40],
      [-0x1b93, 0x878],
    ],
    [
      [0xf11, 0x3665],
      [-0x397, 0x229d],
      [0x2796, -0x1adc],
    ],
    [
      [0x1a13, -0x1395],
      [0x79b, 0xe64],
      [0x229a, -0x2f9f],
    ],
  ],
];
const CONV_B = [0x73a64, -0x7b188];

const DENSE_W = [
  [
    -0x1aca, -0x115, -0x2bf7, 0x3807, 0x2cfb, -0x3aec, -0x3935, 0x24b9, 0x12cd,
    -0x1b24, 0x194e, 0x15cf, 0x2e3, 0xf2c, -0x3dab, 0x3a40, 0x252f, 0x3e59,
    0x2f4f, 0xb70, 0x606, 0x3521, -0x239d, -0x256c, -0x2a23, 0x30ba, 0x16dc,
    0x17e3, 0x227, 0x1f85, 0x313a, -0x1b3a, -0x185e, 0x1b1e, -0x16b0, 0xcc7,
    0x765, 0x231f, -0x1a2b, 0x38c5, 0x11e3, -0x1b6f, 0x26, 0x19ea, 0x1cc,
    -0x272c, -0x2efe, 0x25e, 0x146f, 0x1ed, -0x7c3, -0x159c, -0xffd, 0x240e,
    -0x9dd, 0x2416, 0x3b08, 0xe5c, 0x34ff, 0x339c, 0x26ac, -0x3c7b, 0x3ddb,
    0x103, 0x14d8, -0x297c, -0x367a, -0x343a, -0x3475, -0xca5, -0x343f, -0x2fed,
    -0x1e98, -0x2a81, 0x89, -0x2bcb, 0xdbb, -0x1f71, 0x34a9, 0x2c7c, -0x194a,
    0x3f86, -0x1510, -0x8cf, -0xe7f, 0x226c, -0x538, -0x4d7, 0xe3, -0x1ac7,
    0x8e2, -0x1858, 0xd9b, 0x3870, -0x254f, -0x3a4c, 0x49e, 0x16b4, 0x69d,
    0x1b95, -0x66e, -0x59b, 0x2547, -0x181a, -0x1b8e, -0x1d59, 0x33e5, -0x35cd,
    0x3f76, -0xc80, 0x1f83, -0x2c5a, -0x3ee9, 0x2cc7, -0x2ea8, 0x234e, 0x182f,
    -0xa5f, 0x1f4, 0x1379, 0x21d7, 0x1f6e, 0x2ae9, -0x2405, -0x1c2c, 0xf8f,
    -0x21af, 0x3cdd, 0x2b9, 0x107, 0x604, 0x321b, 0xdf7, 0x2df, -0x23a6, 0x3234,
    0x911, 0xbce, -0xbf4, 0x2220, -0x346f, -0x4c1, 0x347b, -0xef4, 0xd60,
    0x38c1, 0xbf2, 0x2697, -0x2a5e, -0x1502, -0x3860, -0x3a30, -0x115c, -0xb6c,
    -0x3cfc, 0x1c85, -0x1f81, 0x4cf, 0x1984, 0xeb7, -0x3d70, -0x486,
  ],
  [
    0x20fd, -0x1d46, 0x320, -0x861, 0x2242, -0x391c, 0x2406, 0xe96, 0xfac,
    0x3feb, -0x21b3, -0x1f5a, -0x20d1, 0x15b0, -0xb65, 0x31cb, 0x3877, 0x2ae6,
    0x1c1f, 0x30c5, -0x2f78, 0x3229, 0x318a, -0x10cc, -0x36c9, 0xca1, 0xcfa,
    -0x2216, -0x3ab9, 0xbeb, -0x2a44, -0x3199, 0x14c8, -0x34a1, -0x24ea, 0x753,
    0x28a5, -0x10fb, 0x1bfc, 0x148c, -0x191d, 0xc12, 0x3200, -0x8ac, -0x292e,
    -0x125, 0x1b48, 0x304c, 0x313e, 0x3d67, -0x3696, -0x15af, 0x1836, -0x2d55,
    -0x1c71, 0x2c3e, -0x4eb, -0x340b, 0xa70, -0x3fd7, -0xe18, -0x2c52, 0x1bb1,
    -0x366e, 0x1b0c, -0x10a3, -0x105d, 0x2f64, 0x16b4, -0x31b2, -0x27fe, 0x3603,
    0x35ef, 0x19bf, -0x25ef, 0x2a4c, 0x7a2, -0x24db, -0x3c81, -0x3f92, 0xe72,
    0x3415, 0x3285, 0x2644, 0x27e8, -0x3005, -0x82d, 0x24d9, -0x2f0f, 0x1f3c,
    0x2224, -0xdbd, 0x115c, -0x25e5, -0x1d18, 0x8e5, 0x35f4, 0x21ca, 0x332b,
    0x1f28, 0x3ac0, -0x75, 0x3c6a, 0x2abd, 0x112e, -0x7b8, -0x96, 0x2804,
    -0x343c, 0x1069, 0x2cec, -0x1e61, 0x1334, -0x2662, -0x2587, 0x121b, -0x23f5,
    0x707, 0x696, 0x262f, 0x366f, -0x17aa, -0x13ab, 0x2bf, -0x1754, 0x3f6f,
    -0x33e2, -0x35d3, -0x181d, 0x1efb, 0x3b11, 0x1697, 0xd60, -0x1c05, 0x16f4,
    0x113a, -0x35d1, -0xfb2, -0x462, -0x2c47, -0x2a0c, 0xde1, -0x8ad, 0x15a3,
    0x2031, -0x3b5d, -0x3c5d, 0x2fd9, -0x57d, 0x514, 0x4ce, 0x2a17, -0xa72,
    0x2442, -0x372e, 0x155, 0x3a28, -0x37b8, -0x1d18, 0x26e7, -0x2472, -0x2d59,
  ],
  [
    -0x13a1, -0xd6a, 0x24c9, 0x2c25, 0x3a8c, 0x34b4, 0x323, 0x250, -0x2d1b,
    0x1c5a, 0x2f7f, -0x37e4, 0xef0, -0x314e, -0x682, -0x219d, 0x18a4, -0x3d5c,
    0xded, -0x159d, 0x3cc9, -0x40e, -0x3368, 0x1954, 0x1a44, -0xb89, 0x3e6b,
    -0x1a38, 0x2ddf, 0xa1e, 0x224e, -0x393e, 0x2e1c, 0x2fc5, 0x1f08, 0x3927,
    0x891, 0x78, 0x3237, 0x342f, 0x4cd, 0x1ef2, -0x1aad, -0x12a8, 0x35b3, 0xae1,
    -0x2319, -0x236f, 0x3c69, -0x3982, -0x21ce, -0xd88, 0x5ab, -0x2f0d, -0x1d57,
    -0x28cb, -0x1188, 0x36de, 0x1c39, -0x611, -0x373e, 0x772, 0x19c3, -0x1777,
    -0x2973, -0x3814, -0x269b, -0x36c4, 0x3a2, 0x30c3, 0x2bc7, -0x10b3, -0x2c70,
    -0x3640, 0x2260, -0x270e, 0x2fcc, 0x213c, 0x19fe, -0x27e, 0x1d49, 0x3af3,
    -0xa44, -0x15ce, -0x13cd, -0x80e, -0x397c, 0x935, 0x24bb, 0x3ec7, -0xb5a,
    -0x16cf, -0x1975, -0x2219, -0x1c9d, 0x15f8, 0x3c96, 0x1192, 0x2ff6, 0x2447,
    -0x2ac6, -0x1af4, 0xeb7, 0x3fb1, -0x136f, -0x29e6, 0x345f, 0x162c, -0x10e2,
    0xcdb, 0x264, -0x2e11, 0x3a54, -0xc28, 0x380e, 0x10d6, 0x9f5, -0x2c07,
    -0x1482, -0xbc7, 0x23a0, 0x7ee, 0x3c15, 0x1a50, 0x2d8e, 0x2ca0, -0x2af8,
    -0xdd3, -0x1b3f, 0x1db7, 0x2a71, 0x1ae7, 0x1f31, -0x33a9, 0x33bf, -0x384a,
    0x1b12, 0x115a, 0x1353, 0x315e, 0x1ecd, -0x6fc, 0xaf6, -0xc45, -0x1ce,
    0x2221, -0x37a7, 0x35fc, -0x1e8c, 0x1964, 0x18ea, -0x152d, -0x746, 0x105d,
    -0x1354, -0x21c7, -0x2810, 0x3192, 0x30d2, -0x1ac5, 0x39c2, 0x30ae,
  ],
  [
    -0x2e6b, -0x2e71, -0xeb5, 0x13af, -0x11a1, 0x1c67, 0x213f, -0x1db7, -0x2445,
    0x130f, 0x1e44, 0xef6, -0x29bb, -0x1499, 0x10b0, 0x2c09, -0x12fc, -0x22cf,
    -0x16aa, -0x2d65, 0x3870, 0xa46, -0xafa, 0x1024, 0x2149, -0x82e, 0x1fb9,
    -0x31e8, -0x124b, 0x3e9, -0x3456, 0x866, -0x3e4d, -0x2abb, 0x233b, -0x33dc,
    -0xdaf, -0x2db8, 0x8f6, 0x180, 0x3c2f, -0x2570, -0xabe, 0x107c, 0x2fd0,
    -0xb16, -0x25b2, 0x21e1, -0x87b, -0x3, -0x364e, -0x3ca3, 0x83f, 0x2eca,
    -0x2723, 0x171f, -0x9fe, 0x3526, 0x3e2a, -0xfe2, -0x13f5, -0x282b, -0x3d67,
    -0x3fc7, 0xa51, -0x2fd4, 0x385f, -0x2548, -0x2cea, 0x1ea0, 0x26e0, -0x23c3,
    -0x30aa, 0x2dd, 0x3778, 0xc34, -0x3a94, -0x22bd, -0x3098, -0x3572, 0x37f,
    0x17fe, 0x1ea, -0x38e0, 0x367e, -0x3461, -0x38a5, 0x1940, 0x2e31, 0x3ed2,
    -0x370f, -0x1e56, 0x1f33, 0x1d2c, 0x2ac6, 0x26fb, -0x12dc, -0x1c3d, 0x26b9,
    0xa3e, 0x160b, -0x357b, -0x10d, -0x955, 0x1ec0, 0x2b7f, 0x3374, -0x70e,
    0x351b, 0x20d4, 0x16fd, 0x144f, -0x322e, 0x181c, 0x647, -0x8be, 0x977,
    0x113d, 0x2feb, -0x3c01, -0x2161, 0xcf7, 0xaa2, 0xa8f, -0x2a4d, -0x599,
    -0x18b9, -0x12bc, 0x1ecb, -0xec5, 0x42e, -0x3493, -0x386f, -0x1014, 0x1d8c,
    -0x287c, -0x60c, -0x3f6a, 0x6a, 0x364e, 0x1726, -0x145c, -0x3018, 0x330e,
    0x1d9a, 0x1171, 0x1115, -0x3569, 0x15b3, -0x18bb, -0x2f2, -0x1022, 0x18a6,
    -0x33e4, -0x3de4, -0x29e0, -0x19b2, -0x3af8, 0x1289, 0xc54, 0x17a0, 0x3495,
  ],
];
const DENSE_B = [0xaeb690, 0xb0cdb8, -0x7c5fd8, -0xa125e8];

const BN254_PRIME = BigInt(
  "21888242871839275222246405745257275088548364400416034343698204186575808495617",
);

const MIN_LOCK_THRESHOLD = 100000n;
const DEFAULT_LOCK_THRESHOLD = 900000n;
const IMPOSTOR_GUARD_CAP = 20000000n;
const SOFT_MAX_LOCK_THRESHOLD = 35000000n;
const HARD_MAX_LOCK_THRESHOLD = 100000000n;
const SELF_CHECK_THRESHOLD = "1";

export interface EnrollmentCalibrationInfo {
  calibratedOffset: number;
  offsetsTried: number;
  calibrationMs: number;
  variantsSampled: number;
  maxVariantDistance: string;
  p90VariantDistance: string;
  normBasedMinThreshold: string;
}

/**
 * Sync-Pass: Predict 10x10 embedding using JS to match the ZK-circuit logic perfectly.
 */
export function predict20x20(image: number[][][], p: number = 15): number[] {
  // Fallback preview path for non-20x20 inputs (v10 runtime uses 10x10).
  if (!image?.[0]?.length || !image?.[0]?.[0]?.length) return [0, 0, 0, 0];
  if (image[0].length !== 20 || image[0][0].length !== 20) {
    const h = image[0].length;
    const w = image[0][0].length;
    const q = [0, 0, 0, 0];
    const counts = [0, 0, 0, 0];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const bucket = (y < h / 2 ? 0 : 2) + (x < w / 2 ? 0 : 1);
        q[bucket] += image[0][y][x] + image[1][y][x] + image[2][y][x];
        counts[bucket] += 3;
      }
    }
    return q.map((sum, i) => Math.trunc(sum / Math.max(1, counts[i])));
  }

  // 1. Conv2D (Aligned with ZKcash 20x20 Architecture)
  const outSize = 9;
  const numFilters = 2;
  const convOut = Array(numFilters)
    .fill(0)
    .map(() =>
      Array(outSize)
        .fill(0)
        .map(() => Array(outSize).fill(0)),
    );
  const stride = 2;
  const padding = 0;
  const kernelSize = 3;

  for (let f = 0; f < numFilters; f++) {
    for (let i = 0; i < outSize; i++) {
      for (let j = 0; j < outSize; j++) {
        let sum = CONV_B[f];
        for (let c = 0; c < 3; c++) {
          for (let ki = 0; ki < kernelSize; ki++) {
            for (let kj = 0; kj < kernelSize; kj++) {
              const r = i * stride + ki - padding;
              const col = j * stride + kj - padding;
              if (r >= 0 && r < 20 && col >= 0 && col < 20) {
                sum += image[c][r][col] * CONV_W[ki][kj][c][f];
              }
            }
          }
        }
        // LeakyReLU (Bit-Perfect slope shift=2) & CutPrecision (p=15)
        // In model.circom, negative values are divided by 2^(p+SHIFT) = 2^17.
        // Relative to positive division (2^15), this is a division by 4.
        if (sum < 0) sum = Math.trunc(sum / 4);
        convOut[f][i][j] = Math.trunc(sum / Math.pow(2, p));
      }
    }
  }

  // 2. Interleaved Flattening (Match Circom index: i + j*F*W + k*F)
  const flattened: number[] = new Array(162);
  for (let f = 0; f < numFilters; f++) {
    for (let i = 0; i < outSize; i++) {
      for (let j = 0; j < outSize; j++) {
        const index = f + i * numFilters * outSize + j * numFilters;
        flattened[index] = convOut[f][i][j];
      }
    }
  }

  const out: number[] = [];
  for (let i = 0; i < 4; i++) {
    let sum = DENSE_B[i];
    for (let j = 0; j < 162; j++) {
      sum += flattened[j] * DENSE_W[i][j];
    }
    out.push(Math.trunc(sum / Math.pow(2, p)));
  }

  return out;
}

export interface Web3State {
  provider: ethers.BrowserProvider | null;
  signer: ethers.Signer | null;
  address: string | null;
  contract: ethers.Contract | null;
}

const web3State: Web3State = {
  provider: null,
  signer: null,
  address: null,
  contract: null,
};

// 1. Connect Wallet (MetaMask Optimized Provider Shielding)
export async function connectWallet(): Promise<string> {
  const ethereum = (window as any).ethereum;
  if (!ethereum) {
    throw new Error("No crypto wallet found. Please install MetaMask.");
  }

  // Choose MetaMask specifically if multiple providers exist (prevents window.ethereum corruption)
  let providerToUse = ethereum;
  if (ethereum.providers) {
    providerToUse =
      ethereum.providers.find((p: any) => p.isMetaMask) || ethereum;
  }

  web3State.provider = new ethers.BrowserProvider(providerToUse);
  const accounts = await web3State.provider.send("eth_requestAccounts", []);
  web3State.signer = await web3State.provider.getSigner();
  web3State.address = accounts[0];
  web3State.contract = new ethers.Contract(
    CONTRACT_ADDRESS,
    FaceLockedTransferABI,
    web3State.signer,
  );

  return accounts[0];
}

// Ensure the user is on the right network (Base Sepolia Chain ID: 84532)
export async function ensureBaseSepolia(): Promise<void> {
  if (!web3State.provider) throw new Error("Wallet not connected");

  const network = await web3State.provider.getNetwork();
  if (network.chainId !== 84532n) {
    try {
      await (window as any).ethereum.request({
        method: "wallet_switchEthereumChain",
        params: [{ chainId: "0x14a34" }], // 84532 in hex
      });

      // Ethers v6 throws if the underlying network changes. We must re-build the provider context.
      web3State.provider = new ethers.BrowserProvider((window as any).ethereum);
      web3State.signer = await web3State.provider.getSigner();
      web3State.contract = new ethers.Contract(
        CONTRACT_ADDRESS,
        FaceLockedTransferABI,
        web3State.signer,
      );
    } catch (e) {
      // If the chain hasn't been added to MetaMask, we should try adding it
      if ((e as any).code === 4902) {
        await (window as any).ethereum.request({
          method: "wallet_addEthereumChain",
          params: [
            {
              chainId: "0x14a34",
              chainName: "Base Sepolia",
              rpcUrls: ["https://sepolia.base.org"],
              nativeCurrency: { name: "Ether", symbol: "ETH", decimals: 18 },
              blockExplorerUrls: ["https://sepolia.basescan.org/"],
            },
          ],
        });
        web3State.provider = new ethers.BrowserProvider(
          (window as any).ethereum,
        );
        web3State.signer = await web3State.provider.getSigner();
        web3State.contract = new ethers.Contract(
          CONTRACT_ADDRESS,
          FaceLockedTransferABI,
          web3State.signer,
        );
      } else {
        throw new Error(
          "Please switch to Base Sepolia Network in your wallet.",
        );
      }
    }
  }
}
// Helper to hash Float32Array to uint256 (mocking client-side ZK embedding hash logic)
export function hashEmbedding(embedding: Float32Array): bigint {
  const bytes = new Uint8Array(embedding.buffer);
  const hash = ethers.keccak256(bytes);
  return BigInt(hash);
}

// 2. Deposit Funds
export async function depositFunds(
  imagePixels: number[][][],
  _embedding: Float32Array,
): Promise<{
  lockId: bigint;
  commitment: bigint;
  enrolledFeatures: number[];
  threshold: bigint;
  nonce: bigint;
  calibration: EnrollmentCalibrationInfo;
}> {
  if (!web3State.contract) throw new Error("Contract not initialized");
  await ensureBaseSepolia();

  const nonce = BigInt(Math.floor(Math.random() * 1000000));
  const timestamp = Date.now();
  const wasmUrl = `${CIRCUIT_WASM_PATH}?${CIRCUIT_ASSET_TAG}&t=${timestamp}`;
  const zkeyUrl = `${CIRCUIT_ZKEY_PATH}?${CIRCUIT_ASSET_TAG}&t=${timestamp}`;

  // ═══════════════════════════════════════════════════════════════════
  // PASS 1: Run the circuit with DUMMY features to compute a witness,
  // then auto-calibrate a valid 4-value feature window via strict
  // self-checks (distance^2 <= 1 for the same image).
  // ═══════════════════════════════════════════════════════════════════
  const response1 = await fetch("/test_input.json?t=" + Date.now());
  const probeParams = await response1.json();
  probeParams.image = imagePixels;
  probeParams.features = ["0", "0", "0", "0"]; // dummy — we just need the witness
  probeParams.address = probeParams.address.toString();
  probeParams.threshold =
    "14474011154664524427946373126085988481573677491474835889066385825310705051647"; // max so probe doesn't fail
  probeParams.nonce = nonce.toString();

  // Load WASM and compute full witness
  const wasmResp = await fetch(wasmUrl);
  const wasmBuffer = await wasmResp.arrayBuffer();
  const wasmModule = await (window as any).WebAssembly.compile(wasmBuffer);
  const wc = await createWitnessCalculator(wasmModule);
  const witness = await wc.calculateWitness(probeParams);

  const toFieldFeatureInput = (features: number[]): string[] =>
    features.map((v) => {
      let b = BigInt(Math.floor(v)) % BN254_PRIME;
      if (b < 0n) b += BN254_PRIME;
      return b.toString();
    });

  const isCompletedWitness = (w: bigint[]): boolean => {
    return w[2] === 1n;
  };

  const extractSignedFeaturesAt = (
    sourceWitness: bigint[],
    start: number,
  ): number[] | null => {
    const out: number[] = [];
    for (let i = 0; i < 4; i++) {
      const raw = sourceWitness[start + i];
      const signed = raw > BN254_PRIME / 2n ? raw - BN254_PRIME : raw;
      if (
        signed > BigInt(Number.MAX_SAFE_INTEGER) ||
        signed < BigInt(Number.MIN_SAFE_INTEGER)
      ) {
        return null;
      }
      out.push(Number(signed));
    }
    return out;
  };

  const squaredDistance = (a: number[], b: number[]): bigint => {
    let sum = 0n;
    for (let i = 0; i < 4; i++) {
      const d = BigInt(a[i]) - BigInt(b[i]);
      sum += d * d;
    }
    return sum;
  };

  const clampQ15 = (v: number): number => {
    if (v < 0) return 0;
    if (v > 32768) return 32768;
    return Math.round(v);
  };

  const mapImage = (
    source: number[][][],
    fn: (v: number) => number,
  ): number[][][] => {
    return source.map((channel) =>
      channel.map((row) => row.map((px) => clampQ15(fn(px)))),
    );
  };

  const shiftImage = (
    source: number[][][],
    dx: number,
    dy: number,
  ): number[][][] => {
    const h = source[0].length;
    const w = source[0][0].length;
    const out = source.map((channel) => channel.map((row) => row.slice()));

    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const sy = Math.max(0, Math.min(h - 1, y - dy));
          const sx = Math.max(0, Math.min(w - 1, x - dx));
          out[c][y][x] = source[c][sy][sx];
        }
      }
    }

    return out;
  };

  const generateCalibrationVariants = (
    source: number[][][],
  ): number[][][][] => {
    const mid = 16384;
    return [
      // Illumination variants
      mapImage(source, (v) => v * 0.85),
      mapImage(source, (v) => v * 1.15),
      mapImage(source, (v) => v * 0.94),
      mapImage(source, (v) => v * 1.06),
      // Contrast variants
      mapImage(source, (v) => (v - mid) * 0.8 + mid),
      mapImage(source, (v) => (v - mid) * 1.2 + mid),
      mapImage(source, (v) => (v - mid) * 0.92 + mid),
      mapImage(source, (v) => (v - mid) * 1.08 + mid),
      // Alignment jitter variants
      shiftImage(source, -2, 0),
      shiftImage(source, 2, 0),
      shiftImage(source, 0, -2),
      shiftImage(source, 0, 2),
      shiftImage(source, 1, 0),
      shiftImage(source, 0, 1),
      shiftImage(source, -1, 0),
      shiftImage(source, 0, -1),
    ];
  };

  const calibrateFeatureWindow = async (): Promise<{
    features: number[];
    calibratedOffset: number;
    offsetsTried: number;
  }> => {
    const checkWithWitness = async (features: number[]): Promise<boolean> => {
      const checkParams = {
        image: imagePixels,
        address: BigInt(web3State.address || "0").toString(),
        threshold: SELF_CHECK_THRESHOLD,
        nonce: nonce.toString(),
        features: toFieldFeatureInput(features),
      };
      const checkWitness = await wc.calculateWitness(checkParams);
      return isCompletedWitness(checkWitness);
    };

    const checkWithProof = async (features: number[]): Promise<boolean> => {
      const response = await fetch("/test_input.json?t=" + Date.now());
      const proofParams = await response.json();
      proofParams.image = imagePixels;
      proofParams.features = toFieldFeatureInput(features);
      proofParams.address = BigInt(web3State.address || "0").toString();
      proofParams.threshold = SELF_CHECK_THRESHOLD;
      proofParams.nonce = nonce.toString();

      const { publicSignals } = await (window as any).snarkjs.groth16.fullProve(
        proofParams,
        wasmUrl,
        zkeyUrl,
      );
      return publicSignals[1] === "1";
    };

    // v10 uses a stable post-cut embedding signal at witness offset 1470..1473.
    const seedCandidates = [
      V10_FEATURE_OFFSET,
      V10_FEATURE_OFFSET - 1,
      V10_FEATURE_OFFSET + 1,
      V10_FEATURE_OFFSET - 2,
      V10_FEATURE_OFFSET + 2,
      1391,
      1389,
      1382,
      1376,
    ];
    const rangeCandidates: number[] = [];
    for (
      let idx = V10_FEATURE_OFFSET - 24;
      idx <= V10_FEATURE_OFFSET + 24;
      idx++
    ) {
      rangeCandidates.push(idx);
    }
    const candidates = Array.from(
      new Set([...seedCandidates, ...rangeCandidates]),
    );

    let offsetsTried = 0;
    const proofFallbackStarts: number[] = [];

    for (const start of candidates) {
      offsetsTried += 1;
      const maybe = extractSignedFeaturesAt(witness, start);
      if (!maybe) continue;

      // Ignore windows that are clearly outside the model's embedding magnitude.
      if (maybe.some((v) => Math.abs(v) > 5_000_000)) continue;

      if (proofFallbackStarts.length < 8) {
        proofFallbackStarts.push(start);
      }

      if (await checkWithWitness(maybe)) {
        console.log("[ZKcash WitnessSync] Calibrated feature offset:", start);
        return {
          features: maybe,
          calibratedOffset: start,
          offsetsTried,
        };
      }
    }

    // Slow but robust fallback: check top candidates with full proof verification.
    for (const start of proofFallbackStarts) {
      const maybe = extractSignedFeaturesAt(witness, start);
      if (!maybe) continue;
      if (await checkWithProof(maybe)) {
        console.log(
          "[ZKcash WitnessSync] Calibrated feature offset via proof fallback:",
          start,
        );
        return {
          features: maybe,
          calibratedOffset: start,
          offsetsTried,
        };
      }
    }

    throw new Error(
      `Unable to calibrate feature offset from witness. Tried ${offsetsTried} candidates (+ proof fallback). Circuit artifacts appear inconsistent with current extraction logic.`,
    );
  };

  const calibrationStart = performance.now();
  const calibration = await calibrateFeatureWindow();
  const calibrationMs = Math.round(performance.now() - calibrationStart);

  const realFeatures = calibration.features;
  if (realFeatures.length !== 4) {
    throw new Error("Failed to extract 4-dimensional enrolled features.");
  }
  console.log(
    "[ZKcash WitnessSync] Circuit's TRUE CNN features:",
    realFeatures,
  );
  console.log("[ZKcash WitnessSync] Calibration telemetry:", {
    offset: calibration.calibratedOffset,
    offsetsTried: calibration.offsetsTried,
    calibrationMs,
  });

  const deriveAdaptiveThreshold = async (): Promise<{
    threshold: bigint;
    variantsSampled: number;
    maxVariantDistance: bigint;
    p90VariantDistance: bigint;
    normBasedMinThreshold: bigint;
  }> => {
    const variants = generateCalibrationVariants(imagePixels);
    let maxVariantDistance = 0n;
    const distances: bigint[] = [];
    const featureNormSq = realFeatures.reduce((acc, v) => {
      const b = BigInt(v);
      return acc + b * b;
    }, 0n);

    for (const variant of variants) {
      const variantProbe = {
        image: variant,
        address: probeParams.address,
        threshold: probeParams.threshold,
        nonce: probeParams.nonce,
        features: probeParams.features,
      };

      const variantWitness = await wc.calculateWitness(variantProbe);
      const variantFeatures = extractSignedFeaturesAt(
        variantWitness,
        calibration.calibratedOffset,
      );
      if (!variantFeatures) continue;

      const dist = squaredDistance(realFeatures, variantFeatures);
      distances.push(dist);
      if (dist > maxVariantDistance) maxVariantDistance = dist;
    }

    if (distances.length === 0) {
      // Secure fallback for rare artifact/layout inconsistencies.
      let threshold = DEFAULT_LOCK_THRESHOLD;
      const normBasedMinThreshold = (featureNormSq * 55n) / 1000n;
      if (threshold < normBasedMinThreshold) threshold = normBasedMinThreshold;
      if (threshold < MIN_LOCK_THRESHOLD) threshold = MIN_LOCK_THRESHOLD;
      if (threshold > IMPOSTOR_GUARD_CAP) threshold = IMPOSTOR_GUARD_CAP;

      return {
        threshold,
        variantsSampled: 0,
        maxVariantDistance: 0n,
        p90VariantDistance: 0n,
        normBasedMinThreshold,
      };
    }

    distances.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    const p90Index = Math.floor((distances.length - 1) * 0.9);
    const p90VariantDistance = distances[p90Index];

    // Robust thresholding: use P90 to avoid single outlier variants causing over-lenient locks.
    const margin = (p90VariantDistance * 25n) / 100n + 50000n;
    let threshold = p90VariantDistance + margin;

    // Model-scale floor: small synthetic perturbations can underestimate real same-person variance.
    // Use a conservative fraction of embedding norm to reduce false negatives.
    const normBasedMinThreshold = (featureNormSq * 55n) / 1000n; // 5.5% of ||f||^2

    if (threshold < normBasedMinThreshold) threshold = normBasedMinThreshold;
    if (threshold < MIN_LOCK_THRESHOLD) threshold = MIN_LOCK_THRESHOLD;

    // Hard quality gate: reject enrollments that require very large thresholds.
    if (threshold > HARD_MAX_LOCK_THRESHOLD) {
      throw new Error(
        "Enrollment quality is too unstable (required threshold too high). Please retry with a clearer, front-facing image and stable lighting.",
      );
    }

    // Soft cap to reduce false-positive risk while preserving genuine matches.
    if (threshold > SOFT_MAX_LOCK_THRESHOLD)
      threshold = SOFT_MAX_LOCK_THRESHOLD;

    // If even the estimated genuine floor is above the impostor guard cap,
    // there is no safe single-threshold operating point for this model/input.
    if (normBasedMinThreshold > IMPOSTOR_GUARD_CAP) {
      throw new Error(
        "No safe threshold available for this enrollment (identity distributions overlap under current model). Please retry with clearer images or use a stronger trained model.",
      );
    }

    // Security-first cap: prevent over-lenient thresholds that can accept impostors.
    if (threshold > IMPOSTOR_GUARD_CAP) threshold = IMPOSTOR_GUARD_CAP;

    return {
      threshold,
      variantsSampled: variants.length,
      maxVariantDistance,
      p90VariantDistance,
      normBasedMinThreshold,
    };
  };

  const adaptive = await deriveAdaptiveThreshold();
  const finalThreshold = adaptive.threshold;
  console.log("[ZKcash WitnessSync] Adaptive threshold telemetry:", {
    finalThreshold: finalThreshold.toString(),
    maxVariantDistance: adaptive.maxVariantDistance.toString(),
    p90VariantDistance: adaptive.p90VariantDistance.toString(),
    normBasedMinThreshold: adaptive.normBasedMinThreshold.toString(),
    variantsSampled: adaptive.variantsSampled,
  });

  // ═══════════════════════════════════════════════════════════════════
  // PASS 2: Strict self-check (threshold=1).
  // If this fails, extracted features are not the real circuit embedding
  // and enrolling would create an invalid commitment.
  // ═══════════════════════════════════════════════════════════════════
  const response2 = await fetch("/test_input.json?t=" + Date.now());
  const inputParams = await response2.json();
  inputParams.image = imagePixels;

  inputParams.features = toFieldFeatureInput(realFeatures);

  // SECURE BINDING: Use the real connected wallet address for the circuit input
  inputParams.address = BigInt(web3State.address || "0").toString();
  inputParams.threshold = SELF_CHECK_THRESHOLD;
  inputParams.nonce = nonce.toString();

  const { publicSignals } = await (window as any).snarkjs.groth16.fullProve(
    inputParams,
    wasmUrl,
    zkeyUrl,
  );

  if (publicSignals[1] !== "1") {
    throw new Error(
      "Enrollment self-check failed (distance is not zero). Circuit artifacts and witness index mapping are out of sync. Regenerate artifacts or update feature extraction before depositing.",
    );
  }

  const commitment = BigInt(publicSignals[0]);
  const isCompleted = publicSignals[1];
  console.log("WitnessSync Commitment:", commitment.toString());
  console.log(
    "Deposit isCompleted:",
    isCompleted === "1" ? "✅ Perfect Alignment" : "⚠️ Misaligned",
  );

  const tx = await web3State.contract.deposit(
    commitment,
    finalThreshold,
    nonce,
    {
      value: ethers.parseEther("0.00001"),
      gasLimit: 3000000,
    },
  );
  const receipt = await tx.wait();

  // Parse FundsLocked event
  const event = receipt.logs
    .map((log: any) => {
      try {
        return web3State.contract?.interface.parseLog(log);
      } catch (e) {
        return null;
      }
    })
    .find((e: any) => e && e.name === "FundsLocked");

  if (!event || !event.args)
    throw new Error("Failed to find FundsLocked event");

  console.log(
    "FundsLocked Event Caught! Lock ID:",
    event.args.lockId.toString(),
  );

  return {
    lockId: event.args.lockId,
    commitment,
    enrolledFeatures: realFeatures,
    threshold: finalThreshold,
    nonce,
    calibration: {
      calibratedOffset: calibration.calibratedOffset,
      offsetsTried: calibration.offsetsTried,
      calibrationMs,
      variantsSampled: adaptive.variantsSampled,
      maxVariantDistance: adaptive.maxVariantDistance.toString(),
      p90VariantDistance: adaptive.p90VariantDistance.toString(),
      normBasedMinThreshold: adaptive.normBasedMinThreshold.toString(),
    },
  };
}

// ─── Browser Witness Calculator ──────────────────────────────────────────────
// Minimal port of witness_calculator.js for in-browser use
async function createWitnessCalculator(wasmModule: WebAssembly.Module) {
  let errStr = "";
  let msgStr = "";

  function getMessage(instance: any) {
    let message = "";
    let c = instance.exports.getMessageChar();
    while (c !== 0) {
      message += String.fromCharCode(c);
      c = instance.exports.getMessageChar();
    }
    return message;
  }

  const instance = await WebAssembly.instantiate(wasmModule, {
    runtime: {
      exceptionHandler: function (code: number) {
        const errors: Record<number, string> = {
          1: "Signal not found",
          2: "Too many signals",
          3: "Signal already set",
          4: "Assert Failed",
          5: "Not enough memory",
          6: "Input signal array access exceeds size",
        };
        throw new Error((errors[code] || "Unknown error") + "\n" + errStr);
      },
      printErrorMessage: function () {
        errStr += getMessage(instance) + "\n";
      },
      writeBufferMessage: function () {
        const msg = getMessage(instance);
        if (msg === "\n") {
          console.log(msgStr);
          msgStr = "";
        } else {
          if (msgStr) msgStr += " ";
          msgStr += msg;
        }
      },
      showSharedRWMemory: function () {},
    },
  });
  const exports = instance.exports as any;

  const n32 = exports.getFieldNumLen32();
  const witnessSize = exports.getWitnessSize();

  // Get prime
  exports.getRawPrime();
  const primeArr = new Uint32Array(n32);
  for (let j = 0; j < n32; j++) {
    primeArr[n32 - 1 - j] = exports.readSharedRWMemory(j);
  }
  let prime = BigInt(0);
  for (let i = 0; i < primeArr.length; i++) {
    prime = prime * BigInt(0x100000000) + BigInt(primeArr[i]);
  }

  return {
    calculateWitness: async function (
      input: Record<string, any>,
    ): Promise<bigint[]> {
      (exports.init as Function)(0);

      const keys = Object.keys(input);
      for (const k of keys) {
        const h = fnvHash(k);
        const hMSB = parseInt(h.slice(0, 8), 16);
        const hLSB = parseInt(h.slice(8, 16), 16);
        const fArr = flatArray(input[k]);

        for (let i = 0; i < fArr.length; i++) {
          let val = BigInt(fArr[i]) % prime;
          if (val < 0n) val += prime;

          const arr32 = new Array(n32).fill(0);
          let rem = val;
          for (let j = arr32.length - 1; j >= 0; j--) {
            arr32[j] = Number(rem % BigInt(0x100000000));
            rem = rem / BigInt(0x100000000);
          }

          for (let j = 0; j < n32; j++) {
            exports.writeSharedRWMemory(j, arr32[n32 - 1 - j]);
          }
          exports.setInputSignal(hMSB, hLSB, i);
        }
      }

      const w: bigint[] = [];
      for (let i = 0; i < witnessSize; i++) {
        exports.getWitness(i);
        const arr = new Uint32Array(n32);
        for (let j = 0; j < n32; j++) {
          arr[n32 - 1 - j] = exports.readSharedRWMemory(j);
        }
        let val = BigInt(0);
        for (let x = 0; x < arr.length; x++) {
          val = val * BigInt(0x100000000) + BigInt(arr[x]);
        }
        w.push(val);
      }
      return w;
    },
  };

  function fnvHash(str: string): string {
    const uint64_max = BigInt(2) ** BigInt(64);
    let hash = BigInt("0xCBF29CE484222325");
    for (let i = 0; i < str.length; i++) {
      hash ^= BigInt(str.charCodeAt(i));
      hash *= BigInt(0x100000001b3);
      hash %= uint64_max;
    }
    let shash = hash.toString(16);
    return "0".repeat(16 - shash.length) + shash;
  }

  function flatArray(a: any): any[] {
    const res: any[] = [];
    function fill(x: any) {
      if (Array.isArray(x)) {
        for (const item of x) fill(item);
      } else {
        res.push(x);
      }
    }
    fill(a);
    return res;
  }
}

// 3. Claim Funds (using Real ZK-SNARKs!)
export async function claimFunds(
  lockId: bigint,
  imagePixels: number[][][],
  enrolledFeatures: number[],
): Promise<void> {
  if (!web3State.contract) throw new Error("Contract not initialized");
  await ensureBaseSepolia();

  // Load the test input structure with cache-buster
  const response = await fetch("/test_input.json?t=" + Date.now());
  const inputParams = await response.json();

  // SECURE BINDING: Fetch the lock from the blockchain to get the original nonce and threshold
  const lock = await web3State.contract.getLock(lockId);

  // Align inputs: CURRENT pixels vs ORIGINAL embedding
  inputParams.image = imagePixels;
  const PRIME = BigInt(
    "21888242871839275222246405745257275088548364400416034343698204186575808495617",
  );
  inputParams.features = enrolledFeatures.map((v) => {
    let b = BigInt(Math.floor(v)) % PRIME;
    if (b < 0n) b += PRIME;
    return b.toString();
  });

  // Use the same address, threshold and nonce stored on-chain
  inputParams.address = BigInt(web3State.address || "0").toString();
  inputParams.threshold = lock.threshold.toString();
  inputParams.nonce = lock.nonce.toString();

  console.log(
    "[ZKcash Debug] Current Quantized Features (Target):",
    enrolledFeatures,
  );
  console.log(
    `Generating Scientific-Sync ZK Proof (${lock.threshold} Limit)...`,
  );

  // Generate the real ZK-SNARK proof right in the browser!
  const timestamp = Date.now();
  const { proof, publicSignals } = await (
    window as any
  ).snarkjs.groth16.fullProve(
    inputParams,
    `${CIRCUIT_WASM_PATH}?${CIRCUIT_ASSET_TAG}&t=${timestamp}`,
    `${CIRCUIT_ZKEY_PATH}?${CIRCUIT_ASSET_TAG}&t=${timestamp}`,
  );

  console.log(
    "Proof result:",
    publicSignals[1] === "1" ? "✅ Match Found" : "❌ Match Failed",
  );
  console.log("Full Public Signals:", publicSignals);
  console.log("Proof successfully generated by SnarkJS!");

  if (publicSignals[1] !== "1") {
    throw new Error(
      "Biometric Mismatch. The ZK Circuit successfully generated a proof of rejection. Distance exceeded threshold.",
    );
  }

  // Parse Groth16 Proof parameters
  const calldataStr = await (
    window as any
  ).snarkjs.groth16.exportSolidityCallData(proof, publicSignals);
  const args = JSON.parse("[" + calldataStr + "]");

  const pA = args[0];
  const pB = args[1];
  const pC = args[2];
  const pubSignals = args[3];

  const tx = await web3State.contract.claim(lockId, pA, pB, pC, pubSignals, {
    gasLimit: 3000000,
  });
  await tx.wait();
}

export function getAddress(): string | null {
  return web3State.address;
}
