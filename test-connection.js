const http = require('http');

console.log('🔍 Testing backend connection...');

const options = {
  hostname: '192.168.1.6',
  port: 3000,
  path: '/health',
  method: 'GET'
};

const req = http.request(options, (res) => {
  console.log(`✅ Backend is accessible! Status: ${res.statusCode}`);
  
  let data = '';
  res.on('data', (chunk) => {
    data += chunk;
  });
  
  res.on('end', () => {
    console.log('📄 Response:', data);
    console.log('\n🎉 Backend is ready for mobile app connection!');
  });
});

req.on('error', (error) => {
  console.error('❌ Connection failed:', error.message);
  console.log('\n💡 Troubleshooting:');
  console.log('   1. Make sure backend is running: npm run dev');
  console.log('   2. Check Windows Firewall settings');
  console.log('   3. Ensure both devices are on the same network');
});

req.end();
