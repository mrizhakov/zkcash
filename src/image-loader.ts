/**
 * image-loader.ts
 *
 * Handles image upload, drag-and-drop, canvas rendering,
 * and face cropping using face-api.js detection.
 */

import { extractFaceEmbedding, type Box, type FaceEmbedding } from './face-compare';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface LoadedImage {
  /** The original image element */
  image: HTMLImageElement;
  /** The face embedding (null if no face detected) */
  embedding: FaceEmbedding | null;
  /** Face landmarks for alignment */
  landmarks: any | null;
}

// ─── Image Loading ──────────────────────────────────────────────────────────

/**
 * Load an image from a File object and return an HTMLImageElement.
 */
function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (!file.type.startsWith('image/')) {
      reject(new Error('Not an image file'));
      return;
    }

    const img = new Image();
    const reader = new FileReader();

    reader.onload = () => {
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = reader.result as string;
    };

    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

// ─── Canvas Rendering ───────────────────────────────────────────────────────

/**
 * Draw the loaded image onto the display canvas with face box overlay.
 */
function renderToCanvas(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  faceBox: Box | null,
): void {
  // Compute canvas size to maintain aspect ratio
  const maxSize = 400;
  const ratio = Math.min(maxSize / image.naturalWidth, maxSize / image.naturalHeight);
  canvas.width = Math.round(image.naturalWidth * ratio);
  canvas.height = Math.round(image.naturalHeight * ratio);

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  // Draw face detection box
  if (faceBox) {
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(
      faceBox.x * ratio,
      faceBox.y * ratio,
      faceBox.width * ratio,
      faceBox.height * ratio,
    );
    ctx.setLineDash([]);

    // Corner accents
    const cornerLen = 8;
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 3;

    const bx = faceBox.x * ratio;
    const by = faceBox.y * ratio;
    const bw = faceBox.width * ratio;
    const bh = faceBox.height * ratio;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(bx, by + cornerLen);
    ctx.lineTo(bx, by);
    ctx.lineTo(bx + cornerLen, by);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(bx + bw - cornerLen, by);
    ctx.lineTo(bx + bw, by);
    ctx.lineTo(bx + bw, by + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(bx, by + bh - cornerLen);
    ctx.lineTo(bx, by + bh);
    ctx.lineTo(bx + cornerLen, by + bh);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(bx + bw - cornerLen, by + bh);
    ctx.lineTo(bx + bw, by + bh);
    ctx.lineTo(bx + bw, by + bh - cornerLen);
    ctx.stroke();
  }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Full pipeline: load image file → detect face → extract embedding → render display.
 */
export async function processImageFile(
  file: File,
  displayCanvas: HTMLCanvasElement,
): Promise<LoadedImage> {
  const image = await loadImageFromFile(file);

  // face-api.js can work directly on HTMLImageElement
  const embedding = await extractFaceEmbedding(image);

  // Render the image with face box to display canvas
  renderToCanvas(displayCanvas, image, embedding?.box ?? null);

  return {
    image,
    embedding,
    landmarks: embedding ? embedding.landmarks : null,
  };
}

// ─── Event Helpers ──────────────────────────────────────────────────────────

/**
 * Set up drag-and-drop and click-to-upload on a drop zone element.
 */
export function setupDropZone(
  dropZone: HTMLElement,
  fileInput: HTMLInputElement,
  onFile: (file: File) => void,
): void {
  // Click to open file picker
  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) {
      onFile(file);
      fileInput.value = '';
    }
  });

  // Drag-and-drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) onFile(file);
  });
}
