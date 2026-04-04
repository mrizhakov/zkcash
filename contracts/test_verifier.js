const { ethers } = require("ethers");
async function main() {
  const snarkjsModule = require("../node_modules/snarkjs");
  const { proof, publicSignals } = await snarkjsModule.groth16.fullProve(
    require("../public/test_input.json"), 
    "../public/model.wasm", 
    "../public/model_final.zkey"
  );
  
  console.log("Proof publicSignals:", publicSignals);
  
  const provider = new ethers.JsonRpcProvider("https://sepolia.base.org");
  const contract = new ethers.Contract("0x5589Ee2575E03C6d31c8B886E1ddC6f06E3c8ce7", [
    "function verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC, uint[5] calldata _pubSignals) public view returns (bool)"
  ], provider);
  
  const pA = [proof.pi_a[0], proof.pi_a[1]];
  const pB = [
    [proof.pi_b[0][1], proof.pi_b[0][0]],
    [proof.pi_b[1][1], proof.pi_b[1][0]],
  ];
  const pC = [proof.pi_c[0], proof.pi_c[1]];
  
  const isValid = await contract.verifyProof(pA, pB, pC, publicSignals);
  console.log("Is valid?", isValid);
}
main().catch(console.error);
