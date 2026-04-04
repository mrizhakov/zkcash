const fs = require('fs');
const path = require('path');
const faceapi = require('@vladmandic/face-api');
const { Canvas, Image, ImageData, loadImage } = require('canvas');

// Patch face-api for Node
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MODEL_DIR = path.join(__dirname, '../public/models');
const IMAGES_DIR = path.join(__dirname, '../test_images');
const WASM_PATH = path.join(__dirname, '../compiled_circuit/model_circom/model_js/model.wasm');
const WC_PATH = '/tmp/witness_calculator.cjs';

// Emulate getFacePixels behavior from main.ts
function getFacePixels(canvas, size) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const width = canvas.width;
  const height = canvas.height;
  const pixels = [[], [], []]; 
  const hStep = height / size;
  const wStep = width / size;
  
  for (let c = 0; c < 3; c++) {
    for (let sy = 0; sy < size; sy++) {
      const row = [];
      for (let sx = 0; sx < size; sx++) {
        let sum = 0;
        let count = 0;
        const startY = Math.floor(sy * hStep);
        const endY = Math.floor((sy + 1) * hStep);
        const startX = Math.floor(sx * wStep);
        const endX = Math.floor((sx + 1) * wStep);
        for (let y = startY; y < endY; y++) {
          for (let x = startX; x < endX; x++) {
            const idx = (y * width + x) * 4;
            sum += data[idx + c];
            count++;
          }
        }
        const avg = count > 0 ? sum / count : 0;
        row.push(Math.round((avg / 255.0) * 32768).toString()); 
      }
      pixels[c].push(row);
    }
  }
  return pixels;
}

async function main() {
  console.log("Loading face-api models...");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_DIR);
  
  console.log("Loading witness calculator...");
  const wasmBuffer = fs.readFileSync(WASM_PATH);
  const wc = await require(WC_PATH)(wasmBuffer);
  const prime = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');

  const files = fs.readdirSync(IMAGES_DIR).filter(f => f.match(/\.(png|jpg|jpeg)$/i));
  const data = {};

  for (const file of files) {
    console.log(`Processing ${file}...`);
    const imgPath = path.join(IMAGES_DIR, file);
    const img = await loadImage(imgPath);
    
    // Convert canvas image to HTMLImageElement format
    const detection = await faceapi.detectSingleFace(img);
    if (!detection) {
      console.log(`  -> No face detected in ${file}. Skipping.`);
      continue;
    }
    
    const box = detection.box;
    const canvas = new Canvas(Math.round(box.width), Math.round(box.height));
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, box.x, box.y, box.width, box.height, 0, 0, canvas.width, canvas.height);
    
    const pixels = getFacePixels(canvas, 10);
    
    // Pass to circuit
    const wit = await wc.calculateWitness({ image: pixels, address: '1', threshold: '1000000', nonce: '1', features: ['0','0','0','0'] });
    const feat = [];
    for(let i=0; i<4; i++){ 
      let v = wit[1470+i]; 
      feat.push(v > prime / 2n ? Number(v - prime) : Number(v)); 
    }
    
    data[file] = feat;
    console.log(`  -> Features: ${feat}`);
  }

  console.log('\n--- Distance Matrix ---');
  const fileKeys = Object.keys(data);
  for(let i=0; i<fileKeys.length; i++) {
    for(let j=i+1; j<fileKeys.length; j++) {
      const f1 = fileKeys[i];
      const f2 = fileKeys[j];
      let dist = 0n;
      for(let k=0; k<4; k++) {
         const diff = BigInt(data[f1][k]) - BigInt(data[f2][k]);
         dist += diff * diff;
      }
      console.log(`${f1} <-> ${f2} = ${dist.toString()}`);
    }
  }
}

main().catch(console.error);
