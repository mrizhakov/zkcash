/**
 * face-compare.ts
 *
 * Core face comparison logic using @vladmandic/face-api which provides:
 * - SSD MobileNet v1 face detection
 * - 68-point face landmarks
 * - 128-dimensional face recognition embeddings (FaceNet-style)
 *
 * The architecture mirrors the ZKcash face-match pipeline:
 *   FaceDetectorService → getFaceFeatureVectors() → distance calc
 *
 * In the full ZKcash ZKML pipeline, the distance comparison
 * happens inside a Circom circuit producing a Groth16 ZK-SNARK proof.
 */

import * as faceapi from "@vladmandic/face-api";

// ─── Types ──────────────────────────────────────────────────────────────────

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceEmbedding {
  /** 128-dimensional face descriptor from the recognition model */
  embedding: Float32Array;
  /** Bounding box of the detected face */
  box: Box;
  /** Face landmarks (68 points) */
  landmarks: faceapi.FaceLandmarks68 | null;
}

export interface ComparisonResult {
  cosineSimilarity: number;
  euclideanDistance: number;
  matchPercentage: number;
  isMatch: boolean;
  isUncertain: boolean;
  label: string;
}

// ─── Constants ──────────────────────────────────────────────────────────────

/**
 * Euclidean distance thresholds for face-api.js embeddings.
 * face-api.js uses 128-dim descriptors where:
 * - < 0.4 = very likely same person
 * - 0.4-0.6 = uncertain
 * - > 0.6 = different persons
 *
 * In ZKcash's ZK circuit, this threshold check happens in R1CS constraints.
 */
const MATCH_THRESHOLD = 0.45;
const UNCERTAIN_UPPER = 0.6;

// ─── Model Loading ──────────────────────────────────────────────────────────

const MODEL_URL = "/models";

export async function loadModels(
  onProgress?: (status: string) => void,
): Promise<void> {
  onProgress?.("Loading face detection model…");
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

  onProgress?.("Loading face landmark model…");
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

  onProgress?.("Loading face recognition model…");
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

  onProgress?.("Models loaded!");
  console.log("[ZKcash] All face-api.js models loaded");
}

// ─── Face Detection + Embedding ─────────────────────────────────────────────

/**
 * Detect the most prominent face in an image and extract its 128-dim embedding.
 *
 * This mirrors the ZKcash pipeline:
 *   FaceDetectorService.detectFaces() → getFaceFeatureVectors()
 *
 * face-api.js gives us a production-quality 128-dim face descriptor
 * (FaceNet architecture), which is the same dimensionality that
 * ZKcash model outputs.
 */
export async function extractFaceEmbedding(
  input: HTMLCanvasElement | HTMLImageElement,
): Promise<FaceEmbedding | null> {
  const detection = await faceapi
    .detectSingleFace(input)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) return null;

  const box = detection.detection.box;

  return {
    embedding: detection.descriptor,
    box: {
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
    },
    landmarks: detection.landmarks,
  };
}

// ─── Comparison ─────────────────────────────────────────────────────────────

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Compute Euclidean distance between two vectors.
 * This is what ZKcash's ZK circuit actually proves:
 *   "The Euclidean distance between embedding A and B is ≤ threshold"
 */
function euclideanDistance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

/**
 * Compare two face embeddings and return a detailed result.
 *
 * Uses Euclidean distance as the primary metric (same as ZKcash's ZK circuit),
 * with cosine similarity as a secondary signal.
 */
export function compareFaces(
  a: FaceEmbedding,
  b: FaceEmbedding,
): ComparisonResult {
  const distance = euclideanDistance(a.embedding, b.embedding);
  const cosine = cosineSimilarity(a.embedding, b.embedding);

  // Convert distance to a 0-100 match percentage
  // Distance of 0 = 100% match, distance of 1.0+ = 0% match
  const matchPercentage = Math.max(
    0,
    Math.min(100, Math.round((1 - distance / 1.0) * 100)),
  );

  const isMatch = distance < MATCH_THRESHOLD;
  const isUncertain = !isMatch && distance < UNCERTAIN_UPPER;

  let label: string;
  if (isMatch) {
    label = "✓ Same Person";
  } else if (isUncertain) {
    label = "? Uncertain";
  } else {
    label = "✗ Different Person";
  }

  return {
    cosineSimilarity: cosine,
    euclideanDistance: distance,
    matchPercentage,
    isMatch,
    isUncertain,
    label,
  };
}
