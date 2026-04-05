import hre from "hardhat";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as snarkjs from "snarkjs";

const FACE_LOCKED_TRANSFER = "0x06598187677A261d54fd89ECb5497E2295C3A6Fb";

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
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

async function generateProof(inputJsonPath, wasmPath, zkeyPath, signerAddress) {
  const input = readJson(inputJsonPath);
  input.address = BigInt(signerAddress).toString();

  const { proof, publicSignals } = await snarkjs.groth16.fullProve(
    input,
    wasmPath,
    zkeyPath,
  );

  return { proof, publicSignals };
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const repoRoot = path.resolve(__dirname, "../..");

  const wasmPath = path.resolve(repoRoot, "public/model_v10.wasm");
  const zkeyPath = path.resolve(repoRoot, "public/model_v10_final.zkey");
  const failInputPath = path.resolve(
    repoRoot,
    "compiled_circuit_v10_demo/test_input.json",
  );
  const passInputPath = path.resolve(
    repoRoot,
    "compiled_circuit_v10_demo/test_input_pass.json",
  );

  if (!fs.existsSync(wasmPath) || !fs.existsSync(zkeyPath)) {
    throw new Error(
      "Circuit assets are missing: public/model_v10.wasm or public/model_v10_final.zkey",
    );
  }

  const [signer] = await hre.ethers.getSigners();
  const signerAddress = await signer.getAddress();
  let nextNonce = await signer.getNonce("pending");

  const txOverrides = async (extra = {}) => {
    const feeData = await hre.ethers.provider.getFeeData();
    const maxFeePerGas = feeData.maxFeePerGas ?? undefined;
    const maxPriorityFeePerGas = feeData.maxPriorityFeePerGas ?? undefined;

    return {
      nonce: nextNonce++,
      ...(maxFeePerGas ? { maxFeePerGas } : {}),
      ...(maxPriorityFeePerGas ? { maxPriorityFeePerGas } : {}),
      ...extra,
    };
  };

  console.log("Network:", hre.network.name);
  console.log("Signer:", signerAddress);
  console.log("FaceLockedTransfer:", FACE_LOCKED_TRANSFER);

  const faceLock = await hre.ethers.getContractAt(
    "FaceLockedTransfer",
    FACE_LOCKED_TRANSFER,
    signer,
  );

  console.log("\n[1/4] Generating fail-case proof...");
  const failRes = await generateProof(
    failInputPath,
    wasmPath,
    zkeyPath,
    signerAddress,
  );
  console.log("Fail publicSignals:", failRes.publicSignals);
  if (failRes.publicSignals[1] !== "0") {
    throw new Error(
      `Expected fail-case proof (isCompleted=0), got ${failRes.publicSignals[1]}`,
    );
  }

  console.log(
    "\n[2/4] Depositing fail-case lock and validating claim rejection...",
  );
  const failLockId = await faceLock.nextLockId();
  const failCommitment = BigInt(failRes.publicSignals[0]);
  const failThreshold = BigInt(failRes.publicSignals[3]);
  const failNonce = BigInt(failRes.publicSignals[4]);

  const failDepositTx = await faceLock.deposit(
    failCommitment,
    failThreshold,
    failNonce,
    await txOverrides({ value: 1n }),
  );
  await failDepositTx.wait();

  const failArgs = toProofArgs(failRes.proof);
  let failRejected = false;
  try {
    const failClaimTx = await faceLock.claim(
      failLockId,
      failArgs.pA,
      failArgs.pB,
      failArgs.pC,
      failRes.publicSignals,
      await txOverrides(),
    );
    await failClaimTx.wait();
  } catch (error) {
    failRejected = true;
    console.log("Fail-case claim reverted as expected.");
  }

  if (!failRejected) {
    throw new Error(
      "Security regression: fail-case claim unexpectedly succeeded.",
    );
  }

  console.log("\n[3/4] Generating pass-case proof...");
  const passRes = await generateProof(
    passInputPath,
    wasmPath,
    zkeyPath,
    signerAddress,
  );
  console.log("Pass publicSignals:", passRes.publicSignals);
  if (passRes.publicSignals[1] !== "1") {
    throw new Error(
      `Expected pass-case proof (isCompleted=1), got ${passRes.publicSignals[1]}`,
    );
  }

  console.log(
    "\n[4/4] Depositing pass-case lock and validating successful claim...",
  );
  const passLockId = await faceLock.nextLockId();
  const passCommitment = BigInt(passRes.publicSignals[0]);
  const passThreshold = BigInt(passRes.publicSignals[3]);
  const passNonce = BigInt(passRes.publicSignals[4]);

  const passDepositTx = await faceLock.deposit(
    passCommitment,
    passThreshold,
    passNonce,
    await txOverrides({ value: 1n }),
  );
  await passDepositTx.wait();

  const passArgs = toProofArgs(passRes.proof);
  const passClaimTx = await faceLock.claim(
    passLockId,
    passArgs.pA,
    passArgs.pB,
    passArgs.pC,
    passRes.publicSignals,
    await txOverrides(),
  );
  await passClaimTx.wait();

  console.log("Pass-case claim succeeded.");
  console.log("\nE2E verification complete: fail rejected, pass accepted.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
