const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 FreshTrack Backend - GitHub Deployment Helper\n');

// Check if we're in the backend directory
if (!fs.existsSync('package.json')) {
  console.error('❌ Error: Please run this script from the backend directory');
  process.exit(1);
}

// Check if git is initialized
if (!fs.existsSync('.git')) {
  console.log('📁 Initializing git repository...');
  execSync('git init', { stdio: 'inherit' });
}

// Check git status
try {
  console.log('📊 Checking git status...');
  execSync('git status', { stdio: 'inherit' });
} catch (error) {
  console.log('Git not configured yet. Let\'s set it up!\n');
}

console.log('\n📋 Next Steps to Deploy to GitHub:\n');

console.log('1. 🆕 Create a new GitHub repository:');
console.log('   • Go to github.com');
console.log('   • Click "New repository"');
console.log('   • Name it: freshtrack-backend');
console.log('   • Make it Public');
console.log('   • Don\'t initialize with README\n');

console.log('2. 🔗 Add your GitHub remote (replace YOUR_USERNAME):');
console.log('   git remote add origin https://github.com/YOUR_USERNAME/freshtrack-backend.git\n');

console.log('3. 📤 Push your code:');
console.log('   git add .');
console.log('   git commit -m "Initial commit: FreshTrack backend"');
console.log('   git branch -M main');
console.log('   git push -u origin main\n');

console.log('4. 🚀 Deploy to Railway:');
console.log('   • Go to railway.app');
console.log('   • Connect your GitHub repo');
console.log('   • Deploy automatically!\n');

console.log('💡 Pro tip: After setting up the remote, you can run:');
console.log('   npm run deploy:github\n');

// Check if .env exists and warn about it
if (fs.existsSync('.env')) {
  console.log('⚠️  Warning: .env file detected!');
  console.log('   Make sure to add environment variables in Railway dashboard\n');
}

console.log('🎯 Your backend will be live in minutes! 🚀');
