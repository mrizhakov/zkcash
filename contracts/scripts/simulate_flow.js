import hre from "hardhat";

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Using account:", deployer.address);
  
  const faceLockAddress = "0xCD866A7aC80f0c3C6DF8A7f6708ad0D4427c2226"; // Our deployed FaceLockedTransfer
  const FaceLockedTransfer = await hre.ethers.getContractAt("FaceLockedTransfer", faceLockAddress);

  console.log("\n--- SENDER FLOW ---");
  // 1. Simulate the hashing of the face embedding
  // In reality, this is done by hashing the exact ZK-extracted embeddings
  const simulatedCommitment = "12345678901234567890"; // Dummy poseidon hash integer
  const threshold = 327680; // Example threshold
  const nonce = Math.floor(Math.random() * 1000000); // Random nonce
  const depositAmount = hre.ethers.parseEther("0.00001"); // Very small amount of ETH

  console.log("1. Depositing funds locked to face commitment...");
  console.log(`   Commitment Hash: ${simulatedCommitment}`);
  console.log(`   Amount:          0.00001 ETH`);

  const baseNonce = await deployer.getNonce("latest");

  const depositTx = await FaceLockedTransfer.deposit(simulatedCommitment, threshold, nonce, { value: depositAmount, nonce: baseNonce });
  console.log("   Waiting for 'deposit' transaction to be mined...");
  const depositReceipt = await depositTx.wait();
  
  // Extract lockId from the event logs (the first event should be FundsLocked)
  const lockId = hre.ethers.getBigInt(depositReceipt.logs[0].topics[1]);
  console.log(`✅ Success! Funds locked. Lock ID is: ${lockId.toString()}`);

  console.log("\n--- RECIPIENT FLOW ---");
  console.log("2. Simulating ZK Proof generation (client-side)...");
  // The placeholder verifier accepts anything, so we pass dummy data for the Groth16 proof
  const pA = [0, 0];
  const pB = [[0, 0], [0, 0]];
  const pC = [0, 0];
  
  // But the smart contract checks the public signals to ensure you are proving against the RIGHT face
  // The structure expected (based on Bionetta's output):
  // pubSignals[0] = address
  // pubSignals[1] = threshold
  // pubSignals[2] = nonce
  // pubSignals[3] = faceCommitment
  const pubSignals = [
    0,                  // Address (dummy for this basic placeholder)
    threshold,          // must match exactly
    nonce,              // must match exactly
    simulatedCommitment // must match exactly what was deposited!
  ];

  console.log(`   Submitting ZK proof for Lock ID: ${lockId.toString()}`);
  console.log("   Waiting for 'claim' transaction to be mined...");
  
  try {
     const claimTx = await FaceLockedTransfer.claim(lockId, pA, pB, pC, pubSignals, { nonce: baseNonce + 1 });
     await claimTx.wait();
     console.log("✅ Success! Funds were unlocked and claimed by the recipient using their face proof.");
  } catch (error) {
     console.error("❌ Claim failed:", error.message);
  }

  // Let's verify the lock is now 'claimed'
  const lockData = await FaceLockedTransfer.getLock(lockId);
  console.log("\n--- RESULT ---");
  console.log(`Lock ID ${lockId} status => Claimed: ${lockData.claimed}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
