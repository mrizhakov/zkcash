import { ethers } from "ethers";
import fs from "fs";
import path from "path";

const envPath = path.resolve(process.cwd(), ".env");

if (!fs.existsSync(envPath)) {
  const wallet = ethers.Wallet.createRandom();
  fs.writeFileSync(envPath, `PRIVATE_KEY=${wallet.privateKey}\n`);
  console.log(`Generated new deployer wallet: ${wallet.address}`);
} else {
  // Read existing wallet if possible
  const envContent = fs.readFileSync(envPath, "utf-8");
  const match = envContent.match(/PRIVATE_KEY=(0x[a-fA-F0-9]{64})/);
  if (match) {
     const wallet = new ethers.Wallet(match[1]);
     console.log(`Found existing deployer wallet: ${wallet.address}`);
  } else {
     console.log("No valid PRIVATE_KEY found in .env");
  }
}
