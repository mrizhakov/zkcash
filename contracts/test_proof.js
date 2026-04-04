const { ethers } = require("hardhat");

async function main() {
  const Verifier = await ethers.getContractFactory("Groth16Verifier");
  const verifier = await Verifier.deploy();
  
  const snarkjs = require("snarkjs");
  const { proof, publicSignals } = await snarkjs.groth16.fullProve(
    require("../public/test_input.json"), 
    "../public/model.wasm", 
    "../public/model_final.zkey"
  );
  
  console.log("publicSignals:", publicSignals);
  
  const pA = [proof.pi_a[0], proof.pi_a[1]];
  const pB = [
    [proof.pi_b[0][1], proof.pi_b[0][0]],
    [proof.pi_b[1][1], proof.pi_b[1][0]],
  ];
  const pC = [proof.pi_c[0], proof.pi_c[1]];
  
  console.log("Verifying proof directly...");
  const isValid = await verifier.verifyProof(pA, pB, pC, publicSignals);
  console.log("Is valid?", isValid);
}

main().catch(console.error);
