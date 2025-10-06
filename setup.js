#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🚀 Setting up FreshTrack Backend...\n');

// Create necessary directories
const directories = [
  'uploads',
  'data',
  'models'
];

directories.forEach(dir => {
  const dirPath = path.join(__dirname, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`✅ Created directory: ${dir}`);
  } else {
    console.log(`📁 Directory already exists: ${dir}`);
  }
});

// Create .env file if it doesn't exist
const envPath = path.join(__dirname, '.env');
const envExamplePath = path.join(__dirname, 'env.example');

if (!fs.existsSync(envPath) && fs.existsSync(envExamplePath)) {
  fs.copyFileSync(envExamplePath, envPath);
  console.log('✅ Created .env file from template');
} else if (fs.existsSync(envPath)) {
  console.log('📄 .env file already exists');
} else {
  console.log('⚠️  No env.example found, please create .env manually');
}

// Check if model exists
const modelPath = path.join(__dirname, 'models', 'best.onnx');
if (fs.existsSync(modelPath)) {
  console.log('🤖 YOLO model found');
} else {
  console.log('⚠️  No YOLO model found. The system will use mock detections for development.');
  console.log('   To add your model:');
  console.log('   1. Convert your YOLO model to ONNX format');
  console.log('   2. Place it as "best.onnx" in the models/ directory');
}

console.log('\n📋 Next steps:');
console.log('   1. Run: npm install');
console.log('   2. Run: npm run dev');
console.log('   3. Update API_BASE_URL in your React Native app to point to this backend');
console.log('   4. Test the API at: http://localhost:3000/health');

console.log('\n🎉 Setup complete!');
