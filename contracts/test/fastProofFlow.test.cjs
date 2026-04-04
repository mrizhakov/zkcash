const { expect } = require("chai");
const hre = require("hardhat");
const fs = require("node:fs");
const path = require("node:path");

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function signalToAddress(signalValue) {
  const hex = BigInt(signalValue).toString(16).padStart(40, "0");
  return `0x${hex}`;
}

describe("Fast proof flow", function () {
  let verifier;
  let faceLocked;
  let passProof;
  let passSignals;
  let passA;
  let passB;
  let passC;
  let failProof;
  let failSignals;
  let failA;
  let failB;
  let failC;

  beforeEach(async function () {
    const proofDir = path.resolve(
      __dirname,
      "../../compiled_circuit_v10_demo/face_embedding_backbone_circom/groth16",
    );

    passProof = loadJson(path.join(proofDir, "proof_pass.json"));
    passSignals = loadJson(path.join(proofDir, "public_pass.json"));
    failProof = loadJson(path.join(proofDir, "proof.json"));
    failSignals = loadJson(path.join(proofDir, "public.json"));

    passA = [passProof.pi_a[0], passProof.pi_a[1]];
    passB = [
      [passProof.pi_b[0][1], passProof.pi_b[0][0]],
      [passProof.pi_b[1][1], passProof.pi_b[1][0]],
    ];
    passC = [passProof.pi_c[0], passProof.pi_c[1]];

    failA = [failProof.pi_a[0], failProof.pi_a[1]];
    failB = [
      [failProof.pi_b[0][1], failProof.pi_b[0][0]],
      [failProof.pi_b[1][1], failProof.pi_b[1][0]],
    ];
    failC = [failProof.pi_c[0], failProof.pi_c[1]];

    const Verifier = await hre.ethers.getContractFactory("Groth16Verifier");
    const FaceLockedTransfer = await hre.ethers.getContractFactory(
      "FaceLockedTransfer",
    );

    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();

    faceLocked = await FaceLockedTransfer.deploy(await verifier.getAddress());
    await faceLocked.waitForDeployment();
  });

  it("accepts the pass-case fast proof in Groth16Verifier", async function () {
    const isValid = await verifier.verifyProof(
      passA,
      passB,
      passC,
      passSignals,
    );
    expect(isValid).to.equal(true);
  });

  it("allows claim with pass-case signals", async function () {
    const commitment = BigInt(passSignals[0]);
    const threshold = BigInt(passSignals[3]);
    const nonce = BigInt(passSignals[4]);

    await faceLocked.deposit(commitment, threshold, nonce, { value: 100n });

    const claimantAddress = signalToAddress(passSignals[2]);

    await hre.network.provider.request({
      method: "hardhat_setBalance",
      params: [claimantAddress, "0x3635C9ADC5DEA00000"],
    });
    await hre.network.provider.request({
      method: "hardhat_impersonateAccount",
      params: [claimantAddress],
    });

    const claimant = await hre.ethers.getSigner(claimantAddress);
    await expect(
      faceLocked.connect(claimant).claim(0, passA, passB, passC, passSignals),
    ).to.not.be.reverted;

    await hre.network.provider.request({
      method: "hardhat_stopImpersonatingAccount",
      params: [claimantAddress],
    });
  });

  it("reverts claim for fail-case sample due business-level signal checks", async function () {
    const commitment = BigInt(failSignals[0]);
    const threshold = BigInt(failSignals[3]);
    const nonce = BigInt(failSignals[4]);

    await faceLocked.deposit(commitment, threshold, nonce, { value: 100n });

    const claimantAddress = signalToAddress(failSignals[2]);

    await hre.network.provider.request({
      method: "hardhat_setBalance",
      params: [claimantAddress, "0x3635C9ADC5DEA00000"],
    });
    await hre.network.provider.request({
      method: "hardhat_impersonateAccount",
      params: [claimantAddress],
    });

    const claimant = await hre.ethers.getSigner(claimantAddress);

    await expect(
      faceLocked.connect(claimant).claim(0, failA, failB, failC, failSignals),
    ).to.be.revertedWithCustomError(faceLocked, "InvalidProof");

    await hre.network.provider.request({
      method: "hardhat_stopImpersonatingAccount",
      params: [claimantAddress],
    });
  });
});
