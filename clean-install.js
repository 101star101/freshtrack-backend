#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🧹 Cleaning and reinstalling FreshTrack Backend dependencies...\n');

try {
  // Remove node_modules and package-lock.json
  const nodeModulesPath = path.join(__dirname, 'node_modules');
  const packageLockPath = path.join(__dirname, 'package-lock.json');
  
  if (fs.existsSync(nodeModulesPath)) {
    console.log('🗑️  Removing node_modules...');
    fs.rmSync(nodeModulesPath, { recursive: true, force: true });
  }
  
  if (fs.existsSync(packageLockPath)) {
    console.log('🗑️  Removing package-lock.json...');
    fs.unlinkSync(packageLockPath);
  }
  
  // Clear npm cache
  console.log('🧹 Clearing npm cache...');
  execSync('npm cache clean --force', { stdio: 'inherit' });
  
  // Install dependencies
  console.log('📦 Installing dependencies...');
  execSync('npm install', { stdio: 'inherit' });
  
  console.log('\n✅ Installation complete!');
  console.log('🚀 You can now run: npm run dev');
  
} catch (error) {
  console.error('❌ Error during installation:', error.message);
  console.log('\n💡 If you continue to have issues, try:');
  console.log('   1. Update Node.js to a stable version (LTS)');
  console.log('   2. Use yarn instead: npm install -g yarn && yarn install');
  console.log('   3. Install Visual Studio Build Tools for Windows');
}
