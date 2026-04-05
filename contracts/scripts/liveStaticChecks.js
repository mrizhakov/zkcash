import fs from "node:fs";
import path from "node:path";
import { ethers } from "ethers";
import * as snarkjs from "snarkjs";

const FACE_LOCKED_TRANSFER = "0x06598187677A261d54fd89ECb5497E2295C3A6Fb";

function readEnv(filePath) {
  const envRaw = fs.readFileSync(filePath, "utf8");
  return Object.fromEntries(
    envRaw
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#") && line.includes("="))
      .map((line) => {
        const idx = line.indexOf("=");
        return [line.slice(0, idx), line.slice(idx + 1)];
      }),
  );
}

function toProofArgs(proof) {
  return {
    pA: [proof.pi_a[0], proof.pi_a[1]],
    pB: [
      [proof.pi_b[0][1], proof.pi_b[0][0]],
      [proof.pi_b[1][1], proof.pi_b[1][0]],
    ],
    pC: [proof.pi_c[0], proof.pi_c[1]],
  };
}

async function main() {
  const contractsDir = process.cwd();
  const repoRoot = path.resolve(contractsDir, "..");
  const env = readEnv(path.resolve(contractsDir, ".env"));

  const provider = new ethers.JsonRpcProvider(
    env.RPC_URL || "https://sepolia.base.org",
  );
  const wallet = new ethers.Wallet(env.PRIVATE_KEY, provider);

  const faceLock = new ethers.Contract(
    FACE_LOCKED_TRANSFER,
    [
      "function claim(uint256 lockId,uint[2] calldata _pA,uint[2][2] calldata _pB,uint[2] calldata _pC,uint[5] calldata _pubSignals) external",
      "function getLock(uint256 lockId) external view returns (address sender,uint256 amount,uint256 faceCommitment,uint256 threshold,uint256 nonce,bool claimed,uint256 createdAt)",
    ],
    wallet,
  );

  const wasmPath = path.resolve(repoRoot, "public/model_v10.wasm");
  const zkeyPath = path.resolve(repoRoot, "public/model_v10_final.zkey");

  const failInput = JSON.parse(
    fs.readFileSync(
      path.resolve(repoRoot, "compiled_circuit_v10_demo/test_input.json"),
      "utf8",
    ),
  );
  failInput.address = BigInt(wallet.address).toString();

  const passInput = JSON.parse(
    fs.readFileSync(
      path.resolve(repoRoot, "compiled_circuit_v10_demo/test_input_pass.json"),
      "utf8",
    ),
  );
  passInput.address = BigInt(wallet.address).toString();

  console.log("Signer:", wallet.address);
  console.log("Contract:", FACE_LOCKED_TRANSFER);

  const lock0 = await faceLock.getLock(0n);
  const lock1 = await faceLock.getLock(1n);
  console.log("Lock 0 threshold:", lock0.threshold.toString());
  console.log("Lock 1 threshold:", lock1.threshold.toString());

  const failRes = await snarkjs.groth16.fullProve(failInput, wasmPath, zkeyPath);
  const failArgs = toProofArgs(failRes.proof);

  let failPassed = false;
  try {
    await faceLock.claim.staticCall(
      0n,
      failArgs.pA,
      failArgs.pB,
      failArgs.pC,
      failRes.publicSignals,
    );
    failPassed = true;
  } catch {
    console.log("Negative check: lock 0 claim reverts (expected)");
  }
  if (failPassed) {
    throw new Error("Negative check failed: fail-case claim unexpectedly passed");
  }

  const passRes = await snarkjs.groth16.fullProve(passInput, wasmPath, zkeyPath);
  const passArgs = toProofArgs(passRes.proof);

  await faceLock.claim.staticCall(
    1n,
    passArgs.pA,
    passArgs.pB,
    passArgs.pC,
    passRes.publicSignals,
  );
  console.log("Positive check: lock 1 claim simulation passes");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
