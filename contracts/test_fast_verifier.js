import hre from "hardhat";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
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
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();

  const pA = [proof.pi_a[0], proof.pi_a[1]];
  const pB = [
    [proof.pi_b[0][1], proof.pi_b[0][0]],
    [proof.pi_b[1][1], proof.pi_b[1][0]],
  ];
  const pC = [proof.pi_c[0], proof.pi_c[1]];

  const isValid = await verifier.verifyProof(pA, pB, pC, publicSignals);
  console.log("Fast proof valid:", isValid);

  if (!isValid) {
    throw new Error("Verifier returned false for generated fast proof.");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
