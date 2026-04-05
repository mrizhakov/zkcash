/**
 * main.ts — Entry point for ZKcash
 */

import "./style.css";
import { loadModels } from "./face-compare";
import type { Box } from "./face-compare";
import {
  processImageFile,
  setupDropZone,
  type LoadedImage,
} from "./image-loader";
import {
  connectWallet,
  depositFunds,
  claimFunds,
  predict20x20,
  getRuntimeConfig,
  type EnrollmentCalibrationInfo,
} from "./web3";

console.log("🚀 ZKCASH VERSION 2.0.0 - WITNESS-SYNC PASS");

// ─── State ──────────────────────────────────────────────────────────────────

interface AppState {
  imageA: LoadedImage | null;
  imageB: LoadedImage | null;
  facePixelsA: number[][][] | null;
  facePixelsB: number[][][] | null;
  walletAddress: string | null;
  currentLockId: bigint | null;
  currentCommitment: bigint | null;
  currentEnrolledFeatures: number[] | null;
  currentThreshold: bigint | null;
  currentNonce: bigint | null;
  currentCalibration: EnrollmentCalibrationInfo | null;
}

const CLAIM_PRECHECK_FULL_DISTANCE = 0.55;

const state: AppState = {
  imageA: null,
  imageB: null,
  facePixelsA: null,
  facePixelsB: null,
  walletAddress: null,
  currentLockId: null,
  currentCommitment: null,
  currentEnrolledFeatures: null,
  currentThreshold: null,
  currentNonce: null,
  currentCalibration: null,
};

// ─── DOM References ─────────────────────────────────────────────────────────

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const loadingOverlay = $<HTMLDivElement>("loading-overlay");
const loadingStatus = $<HTMLParagraphElement>("loading-status");

const connectWalletBtn = $<HTMLButtonElement>("connect-wallet-btn");

const canvasA = $<HTMLCanvasElement>("canvas-a");
const canvasB = $<HTMLCanvasElement>("canvas-b");
const placeholderA = $<HTMLDivElement>("placeholder-a");
const placeholderB = $<HTMLDivElement>("placeholder-b");
const faceStatusA = $<HTMLSpanElement>("face-status-a");
const faceStatusB = $<HTMLSpanElement>("face-status-b");
const dropZoneA = $<HTMLDivElement>("drop-zone-a");
const dropZoneB = $<HTMLDivElement>("drop-zone-b");
const fileInputA = $<HTMLInputElement>("file-a");
const fileInputB = $<HTMLInputElement>("file-b");

// Web3 Stepper panels
const stepConnect = $<HTMLDivElement>("step-connect");
const stepDeposit = $<HTMLDivElement>("step-deposit");
const stepClaim = $<HTMLDivElement>("step-claim");
const resultLoading = $<HTMLDivElement>("result-loading");
const resultMatch = $<HTMLDivElement>("result-match");

const loadingTxt = $<HTMLParagraphElement>("loading-txt");
const depositBtn = $<HTMLButtonElement>("deposit-btn");
const claimBtn = $<HTMLButtonElement>("claim-btn");
const resetBtn = $<HTMLButtonElement>("reset-btn");
const lockIdDisplay = $<HTMLElement>("lock-id-display");

// Debug elements
const debugCalcStatus = $("debug-calc-status");
const debugDistFull = $("debug-dist-full");
const debugDistCNN = $("debug-dist-cnn");
const debugLikenessBar = $("debug-likeness-bar");
const debugLikenessPercent = $("debug-likeness-percent");
const debugZkThreshold = $("debug-zk-threshold");
const debugCalibOffset = $("debug-calib-offset");
const debugCalibScan = $("debug-calib-scan");
const debugRuntimeConfig = $("debug-runtime-config");

// ─── Initialization ─────────────────────────────────────────────────────────

async function init(): Promise<void> {
  try {
    await loadModels((status) => {
      loadingStatus.textContent = status;
    });

    setTimeout(() => {
      loadingOverlay.classList.add("hidden");
    }, 500);

    setupUI();
  } catch (err) {
    loadingStatus.textContent = `Error: ${(err as Error).message}`;
    console.error("[ZKcash] Init failed:", err);
  }
}

// ─── UI Setup ───────────────────────────────────────────────────────────────

function setupUI(): void {
  // Set up drop zones
  setupDropZone(dropZoneA, fileInputA, (file) => handleImage("a", file));
  setupDropZone(dropZoneB, fileInputB, (file) => handleImage("b", file));

  connectWalletBtn.addEventListener("click", handleConnectWallet);
  depositBtn.addEventListener("click", handleDeposit);
  claimBtn.addEventListener("click", handleClaim);
  resetBtn.addEventListener("click", resetApp);

  updateStepView();
}

async function handleConnectWallet(): Promise<void> {
  try {
    connectWalletBtn.textContent = "Connecting...";
    connectWalletBtn.disabled = true;
    const address = await connectWallet();
    state.walletAddress = address;

    // Dispay shortened address
    connectWalletBtn.textContent = `${address.slice(0, 6)}...${address.slice(-4)}`;
    updateStepView();
  } catch (err) {
    alert((err as Error).message);
    connectWalletBtn.textContent = "Connect Wallet";
    connectWalletBtn.disabled = false;
  }
}

function updateStepView(): void {
  // Hide all panels
  stepConnect.hidden = true;
  stepDeposit.hidden = true;
  stepClaim.hidden = true;
  resultLoading.hidden = true;
  resultMatch.hidden = true;

  if (!state.walletAddress) {
    stepConnect.hidden = false;
  } else if (state.currentLockId === null) {
    stepDeposit.hidden = false;
    depositBtn.disabled = !state.imageA || !state.imageA.embedding;
  } else {
    stepClaim.hidden = false;
    lockIdDisplay.textContent = state.currentLockId.toString();
    claimBtn.disabled = !state.imageB || !state.imageB.embedding;
  }

  updateDebugPanel();
}

function calculateDistance(v1: Float32Array, v2: Float32Array): number {
  if (v1.length !== v2.length) return 999;
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
}

function updateDebugPanel(): void {
  const cfg = getRuntimeConfig();
  const shortAddr = `${cfg.contractAddress.slice(0, 8)}...${cfg.contractAddress.slice(-6)}`;
  debugRuntimeConfig.textContent = `${shortAddr} | ${cfg.wasmPath} | ${cfg.zkeyPath} | off ${cfg.featureOffset}`;

  debugZkThreshold.textContent = state.currentThreshold
    ? state.currentThreshold.toString()
    : "--";

  debugCalibOffset.textContent = state.currentCalibration
    ? state.currentCalibration.calibratedOffset.toString()
    : "--";

  debugCalibScan.textContent = state.currentCalibration
    ? `${state.currentCalibration.offsetsTried} windows / ${state.currentCalibration.calibrationMs} ms / p90 ${state.currentCalibration.p90VariantDistance} / max ${state.currentCalibration.maxVariantDistance} / normMin ${state.currentCalibration.normBasedMinThreshold}`
    : "--";

  if (!state.imageA || !state.facePixelsA) {
    debugCalcStatus.textContent = "Waiting for Image A...";
    return;
  }

  if (!state.imageB || !state.facePixelsB) {
    debugCalcStatus.textContent = "Waiting for Image B...";
    debugDistFull.textContent = "--";
    debugDistCNN.textContent = "--";
    debugLikenessBar.style.width = "0%";
    debugLikenessPercent.textContent = "0%";
    return;
  }

  debugCalcStatus.textContent =
    "🧠 NEUORSYNC: 10x10 Active Hardware-Aligned Analysis";

  // 1. Full-Resolution distance (Ground Truth from face-api.js)
  const distFull = calculateDistance(
    state.imageA!.embedding!.embedding,
    state.imageB!.embedding!.embedding,
  );
  debugDistFull.textContent = distFull.toFixed(4);

  // 2. CNN Prediction distance (as seen by the ZK-circuit)
  const featuresA = predict20x20(state.facePixelsA);
  const featuresB = predict20x20(state.facePixelsB);

  let squaredDistCNN = 0;
  for (let i = 0; i < 4; i++) {
    squaredDistCNN += Math.pow(featuresA[i] - featuresB[i], 2);
  }
  debugDistCNN.textContent = (squaredDistCNN / 1e6).toFixed(1) + "M";

  // Likeness score calculation against active on-chain threshold.
  const activeThreshold = Number(state.currentThreshold ?? 900000n);

  // Likeness score normalized to the active lock threshold.
  const likeness = Math.max(0, 100 - squaredDistCNN / (activeThreshold / 100));
  debugLikenessBar.style.width = `${likeness}%`;
  debugLikenessPercent.textContent = `${Math.round(likeness)}%`;

  // Update status color against active on-chain threshold.
  if (squaredDistCNN < activeThreshold) {
    debugLikenessPercent.style.color = "var(--accent-green)";
  } else if (squaredDistCNN < activeThreshold * 1.5) {
    debugLikenessPercent.style.color = "var(--accent-yellow)";
  } else {
    debugLikenessPercent.style.color = "var(--accent-red)";
  }

  console.log("[ZKcash ScientificSync] Live Connection:", {
    fullDist: distFull,
    cnnSquaredDist: squaredDistCNN,
    passLimit: activeThreshold,
  });
}

// ─── Image Handling ─────────────────────────────────────────────────────────

async function handleImage(side: "a" | "b", file: File): Promise<void> {
  const canvas = side === "a" ? canvasA : canvasB;
  const placeholder = side === "a" ? placeholderA : placeholderB;
  const statusEl = side === "a" ? faceStatusA : faceStatusB;

  // Show loading state
  statusEl.textContent = "Detecting face…";
  statusEl.className = "face-status";

  try {
    const loaded = await processImageFile(file, canvas);

    canvas.classList.add("visible");
    placeholder.classList.add("hidden");

    if (loaded.embedding) {
      statusEl.textContent = "✓ Face detected";
      statusEl.className = "face-status detected";
    } else {
      statusEl.textContent = "No face detected";
      statusEl.className = "face-status error";
    }

    const box = loaded.embedding?.box;
    if (side === "a") {
      state.imageA = loaded;
      state.facePixelsA =
        box && loaded.image
          ? getFacePixels(loaded.image, 10, box, loaded.landmarks)
          : [];
    } else {
      state.imageB = loaded;
      state.facePixelsB =
        box && loaded.image
          ? getFacePixels(loaded.image, 10, box, loaded.landmarks)
          : [];
    }

    updateStepView();
  } catch (err) {
    statusEl.textContent = "Error loading image";
    statusEl.className = "face-status error";
    console.error(`[ZKcash] Image ${side} error:`, err);
  }
}

/**
 * Extracts raw pixel data from the canvas and resizes it to the target dimensions
 * required by the ZK-ML model (e.g., 10x10x3).
 */
function getFacePixels(
  image: HTMLImageElement,
  size: number,
  box: Box,
  landmarks: any,
): number[][][] {
  // 1. Create a specialized Professional Alignment Canvas (200x200)
  const alignCanvas = document.createElement("canvas");
  alignCanvas.width = 200;
  alignCanvas.height = 200;
  const alignCtx = alignCanvas.getContext("2d", { willReadFrequently: true });
  if (!alignCtx) return [];

  if (landmarks) {
    // ─── Professional 3-Point Alignment ───────────────────────────────────────
    const getAvgPoint = (pts: any[]) => {
      let x = 0,
        y = 0;
      pts.forEach((p) => {
        x += p.x;
        y += p.y;
      });
      return { x: x / pts.length, y: y / pts.length };
    };

    const LC = getAvgPoint(landmarks.getLeftEye());
    const RC = getAvgPoint(landmarks.getRightEye());
    const MC = getAvgPoint(landmarks.getMouth());

    // Eye line angle
    const angle = Math.atan2(RC.y - LC.y, RC.x - LC.x);
    // Face center (weighted eyes + mouth)
    const faceCenter = {
      x: (LC.x + RC.x + MC.x) / 3,
      y: (LC.y + RC.y + MC.y) / 3,
    };

    // Scale so eyes are 80px apart
    const eyeDist = Math.sqrt(
      Math.pow(RC.x - LC.x, 2) + Math.pow(RC.y - LC.y, 2),
    );
    const scale = 80 / eyeDist;

    alignCtx.save();
    alignCtx.translate(100, 100); // Center the face in the 200x200 grid
    alignCtx.rotate(-angle);
    alignCtx.scale(scale, scale);
    alignCtx.drawImage(image, -faceCenter.x, -faceCenter.y);
    alignCtx.restore();

    // ─── Biometric Contrast Normalization (Min-Max Stretching) ───────────────
    let imageData = alignCtx.getImageData(0, 0, 200, 200);
    const data = imageData.data;
    let min = 255,
      max = 0;
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const val = data[i + c];
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }
    if (max > min) {
      for (let i = 0; i < data.length; i += 4) {
        for (let c = 0; c < 3; c++) {
          data[i + c] = Math.round(((data[i + c] - min) / (max - min)) * 255);
        }
      }
      alignCtx.putImageData(imageData, 0, 0);
    }
  } else {
    alignCtx.drawImage(
      image,
      -box.x,
      -box.y,
      image.naturalWidth,
      image.naturalHeight,
    );
  }

  // 2. Extract Professional-Grade Pixels (aligned + equalized)
  const imageData = alignCtx.getImageData(0, 0, 200, 200);
  const data = imageData.data;
  const roiWidth = 200,
    roiHeight = 200;

  // 3. Deterministic manual box-averaging
  const pixels: number[][][] = [[], [], []];

  const hStep = roiHeight / size;
  const wStep = roiWidth / size;

  for (let c = 0; c < 3; c++) {
    for (let sy = 0; sy < size; sy++) {
      const row: number[] = [];
      for (let sx = 0; sx < size; sx++) {
        let sum = 0;
        let count = 0;

        const startY = Math.floor(sy * hStep);
        const endY = Math.floor((sy + 1) * hStep);
        const startX = Math.floor(sx * wStep);
        const endX = Math.floor((sx + 1) * wStep);

        for (let y = startY; y < endY; y++) {
          if (y < 0 || y >= roiHeight) continue;
          for (let x = startX; x < endX; x++) {
            if (x < 0 || x >= roiWidth) continue;
            const idx = (y * roiWidth + x) * 4;
            sum += data[idx + c];
            count++;
          }
        }

        const avg = count > 0 ? sum / count : 0;
        // Quantize to 15-bit (0..32768)
        row.push(Math.round((avg / 255.0) * 32768));
      }
      pixels[c].push(row);
    }
  }

  return pixels;
}

// ─── Web3 Actions ───────────────────────────────────────────────────────────

async function handleDeposit() {
  if (!state.imageA?.embedding) return;

  stepDeposit.hidden = true;
  resultLoading.hidden = false;
  loadingTxt.textContent = "Please confirm the lock transaction in MetaMask...";

  try {
    const res = await depositFunds(
      state.facePixelsA!,
      state.imageA!.embedding!.embedding,
    );
    state.currentLockId = res.lockId;
    state.currentCommitment = res.commitment;
    state.currentEnrolledFeatures = res.enrolledFeatures;
    state.currentThreshold = res.threshold;
    state.currentNonce = res.nonce;
    state.currentCalibration = res.calibration;
    updateStepView();
  } catch (err) {
    console.error(err);
    alert("Deposit transaction failed: " + (err as Error).message);
    updateStepView(); // Revert back to deposit step
  }
}

async function handleClaim() {
  if (!state.imageB?.embedding || state.currentLockId === null) return;

  stepClaim.hidden = true;
  resultLoading.hidden = false;

  try {
    const fullDistance = calculateDistance(
      state.imageA!.embedding!.embedding,
      state.imageB!.embedding!.embedding,
    );

    if (fullDistance > CLAIM_PRECHECK_FULL_DISTANCE) {
      throw new Error(
        `Biometric pre-check failed: face distance ${fullDistance.toFixed(4)} exceeds safety limit ${CLAIM_PRECHECK_FULL_DISTANCE.toFixed(2)}.`,
      );
    }

    // 1. ZK proof generation
    loadingTxt.textContent = "Generating ZK Proof locally...";

    // 2. Submit transaction
    loadingTxt.textContent =
      "Please confirm the claim transaction in MetaMask...";
    await claimFunds(
      state.currentLockId,
      state.facePixelsB!,
      state.currentEnrolledFeatures!,
    );

    // 3. Success
    resultLoading.hidden = true;
    resultMatch.hidden = false;
    resultMatch.classList.add("animate-in");
  } catch (err) {
    console.error(err);
    alert("Claim transaction failed: " + (err as Error).message);
    updateStepView();
  }
}

function resetApp() {
  state.imageA = null;
  state.imageB = null;
  state.currentLockId = null;
  state.currentCommitment = null;
  state.currentThreshold = null;
  state.currentNonce = null;
  state.currentCalibration = null;

  // Reset UI canvasses
  canvasA.classList.remove("visible");
  placeholderA.classList.remove("hidden");
  faceStatusA.textContent = "No face detected";
  faceStatusA.className = "face-status";

  canvasB.classList.remove("visible");
  placeholderB.classList.remove("hidden");
  faceStatusB.textContent = "No face detected";
  faceStatusB.className = "face-status";

  updateStepView();
}

// ─── Start ──────────────────────────────────────────────────────────────────

init();
