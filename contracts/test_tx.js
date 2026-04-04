import hre from "hardhat";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function signalToAddress(signalValue) {
  const hex = BigInt(signalValue).toString(16).padStart(40, "0");
  return `0x${hex}`;
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  const proofDir = path.resolve(
    __dirname,
    "../compiled_circuit_v10_demo/face_embedding_backbone_circom/groth16",
  );
  const proof = loadJson(path.join(proofDir, "proof_pass.json"));
  const publicSignals = loadJson(path.join(proofDir, "public_pass.json"));

  const Verifier = await hre.ethers.getContractFactory("Groth16Verifier");
  const FaceLockedTransfer = await hre.ethers.getContractFactory(
    "FaceLockedTransfer",
  );
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();

  const faceLocked = await FaceLockedTransfer.deploy(
    await verifier.getAddress(),
  );
  await faceLocked.waitForDeployment();

  const commitment = BigInt(publicSignals[0]);
  const threshold = BigInt(publicSignals[3]);
  const nonce = BigInt(publicSignals[4]);
  await faceLocked.deposit(commitment, threshold, nonce, { value: 100n });

  const pA = [proof.pi_a[0], proof.pi_a[1]];
  const pB = [
    [proof.pi_b[0][1], proof.pi_b[0][0]],
    [proof.pi_b[1][1], proof.pi_b[1][0]],
  ];
  const pC = [proof.pi_c[0], proof.pi_c[1]];

  const verifierResult = await verifier.verifyProof(pA, pB, pC, publicSignals);
  console.log("Verifier result:", verifierResult);

  const claimantAddress = signalToAddress(publicSignals[2]);
  await hre.network.provider.request({
    method: "hardhat_setBalance",
    params: [claimantAddress, "0x3635C9ADC5DEA00000"],
  });
  await hre.network.provider.request({
    method: "hardhat_impersonateAccount",
    params: [claimantAddress],
  });

  const claimant = await hre.ethers.getSigner(claimantAddress);

  console.log("Claiming from address:", claimantAddress);
  console.log("Using public signals:", publicSignals);

  const tx = await faceLocked
    .connect(claimant)
    .claim(0, pA, pB, pC, publicSignals);
  await tx.wait();
  console.log("Claim succeeded.");

  await hre.network.provider.request({
    method: "hardhat_stopImpersonatingAccount",
    params: [claimantAddress],
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
