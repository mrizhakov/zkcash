import * as faceapi from "@vladmandic/face-api";

function getFacePixels(canvas: HTMLCanvasElement, size: number): number[][][] {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const width = canvas.width;
  const height = canvas.height;
  const pixels: number[][][] = [[], [], []];
  const hStep = height / size;
  const wStep = width / size;

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
          for (let x = startX; x < endX; x++) {
            const idx = (y * width + x) * 4;
            sum += data[idx + c];
            count++;
          }
        }
        const avg = count > 0 ? sum / count : 0;
        row.push(Math.round((avg / 255.0) * 32768));
      }
      pixels[c].push(row);
    }
  }
  return pixels;
}

export async function createWitnessCalculator(wasmModule: WebAssembly.Module) {
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
        throw new Error("Exception " + code + "\n" + errStr);
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

async function runBenchmark() {
  const logDiv = document.getElementById("log-container")!;
  const log = (msg: string) => {
    logDiv.textContent += msg + "\n";
    console.log(msg);
  };

  try {
    log("Loading face-api models...");
    await faceapi.nets.ssdMobilenetv1.loadFromUri("/models");

    log("Loading WebAssembly circuit...");
    const wasmResp = await fetch(`/model_v10.wasm?v=10&t=${Date.now()}`);
    const wasmBuffer = await wasmResp.arrayBuffer();
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const wc = await createWitnessCalculator(wasmModule);

    const imageFiles = [
      "User_A_1.jpg",
      "User_A_2.jpg",
      "User_A_4.jpg",
      "User_B_1.png",
    ];
    const data: Record<string, number[]> = {};

    const PRIME = BigInt(
      "21888242871839275222246405745257275088548364400416034343698204186575808495617",
    );

    for (const file of imageFiles) {
      log(`Processing /test_images/${file}...`);

      const img = new Image();
      img.src = `/test_images/${file}`;
      await new Promise((r) => (img.onload = r));

      const detection = await faceapi.detectSingleFace(img);
      if (!detection) {
        log(`  -> No face detected. Skipping.`);
        continue;
      }

      const box = detection.box;
      const canvas = document.createElement("canvas");
      canvas.width = box.width;
      canvas.height = box.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(
        img,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        canvas.width,
        canvas.height,
      );

      const pixels = getFacePixels(canvas, 10);

      const input = {
        image: pixels,
        address: "1",
        threshold: "1000000",
        nonce: "1",
        features: ["0", "0", "0", "0"],
      };
      const wit = await wc.calculateWitness(input);

      const feat: number[] = [];
      for (let i = 0; i < 4; i++) {
        const val = wit[1470 + i];
        feat.push(val > PRIME / 2n ? Number(val - PRIME) : Number(val));
      }

      data[file] = feat;
      log(`  -> Features: [${feat.join(", ")}]`);
    }

    log("\n=== DISTANCE MATRIX ===");
    const files = Object.keys(data);

    // Setup JSON export variable
    const resultObj: any = { features: data, distances: [] };

    for (let i = 0; i < files.length; i++) {
      for (let j = i; j < files.length; j++) {
        const f1 = files[i];
        const f2 = files[j];
        let dist = 0n;
        for (let k = 0; k < 4; k++) {
          const diff = BigInt(data[f1][k]) - BigInt(data[f2][k]);
          dist += diff * diff;
        }
        log(`${f1} <-> ${f2} = ${dist.toString()}`);
        resultObj.distances.push({ from: f1, to: f2, dist: Number(dist) });
      }
    }

    log("\nDONE_TESTING");
    (window as any).__BENCHMARK_RESULTS = resultObj;
  } catch (err) {
    log(`ERROR: ${(err as Error).message}`);
    (window as any).__BENCHMARK_RESULTS = { error: (err as Error).message };
  }
}

runBenchmark();
