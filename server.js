// server.js
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const { processImage } = require('./services/imageProcessor');
const { getStorageData } = require('./services/storageService');

const app = express();

// Use Railway or local port
const PORT = process.env.PORT || 3000;

// ===================
// Middleware setup
// ===================
app.use(helmet());
app.use(morgan('combined'));
app.use(cors({
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// ===================
// File upload config
// ===================
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) cb(null, true);
    else cb(new Error('Only image files are allowed!'), false);
  }
});

// ===================
// Routes
// ===================

// Root info route
app.get('/', (req, res) => {
  res.json({
    message: '🍏 FreshTrack Backend API',
    version: '2.0.0',
    status: 'Running',
    endpoints: {
      '/api/detect': 'POST - Detect food freshness in an uploaded image',
      '/api/storage/:item': 'GET - Get storage info for a specific item',
      '/api/storage': 'GET - Get all storage data',
      '/health': 'GET - Check API health'
    }
  });
});

// Health check (for Railway monitoring)
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Debug info
app.get('/api/test', (req, res) => {
  try {
    res.json({
      message: '✅ Test endpoint working',
      timestamp: new Date().toISOString(),
      env: {
        NODE_ENV: process.env.NODE_ENV,
        PORT: process.env.PORT,
        MODEL_PATH: process.env.MODEL_PATH,
        FORCE_MOCK_MODE: process.env.FORCE_MOCK_MODE
      },
      platform: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Test endpoint failed', details: error.message });
  }
});

// ===================
// Image detection route
// ===================
app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image file provided' });

    console.log('🖼️ Received image:', req.file.filename);

    // Process image with model or mock
    const detections = await processImage(req.file.path);

    // Enrich with storage info
    const enriched = detections.map(det => ({
      ...det,
      storage: getStorageData(det.label)
    }));

    res.json({
      success: true,
      detections: enriched,
      count: enriched.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('❌ Detection error:', error);
    res.status(500).json({
      error: 'Failed to process image',
      details: error.message
    });
  }
});

// ===================
// Storage info routes
// ===================
app.get('/api/storage/:item', (req, res) => {
  try {
    const item = req.params.item.toLowerCase();
    const data = getStorageData(item);
    if (!data) return res.status(404).json({ error: 'Item not found' });
    res.json({ success: true, item, storage: data });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get storage data' });
  }
});

app.get('/api/storage', (req, res) => {
  try {
    const all = require('./data/storage_data.json');
    res.json({ success: true, data: all });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load storage data' });
  }
});

// ===================
// Error handling
// ===================
app.use((err, req, res, next) => {
  console.error('💥 Internal error:', err);
  res.status(500).json({ error: 'Internal server error', details: err.message });
});

app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// ===================
// Start server
// ===================
app.listen(PORT, () => {
  console.log(`🚀 FreshTrack Backend running on port ${PORT}`);
  console.log(`✅ Health: http://localhost:${PORT}/health`);
});
