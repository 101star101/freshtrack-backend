// server.js
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const { processImage, preloadModel } = require('./services/imageProcessor');
const { getStorageData } = require('./services/storageService');

const app = express();
const PORT = process.env.PORT || 3000;

// ===================
// Middleware setup
// ===================
app.use(helmet());
app.use(morgan('combined'));
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
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
  limits: { fileSize: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) cb(null, true);
    else cb(new Error('Only image files are allowed!'), false);
  }
});

// ===================
// Routes
// ===================
app.get('/', (req, res) => {
  res.json({
    message: '🍏 FreshTrack Backend API',
    version: '2.1.0',
    status: 'Running',
    endpoints: {
      '/api/detect': 'POST - Detect food freshness in an uploaded image',
      '/api/storage/:item': 'GET - Get storage info for a specific item',
      '/api/storage': 'GET - Get all storage data',
      '/health': 'GET - Check API health'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image file provided' });
    console.log('🖼️ Received image:', req.file.filename);

    const detections = await processImage(req.file.path);
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
    res.status(500).json({ error: 'Failed to process image', details: error.message });
  }
});

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
// Startup
// ===================
async function startServer() {
  try {
    console.log('🧠 Preloading YOLO model...');
    await preloadModel();
    console.log('✅ Model ready.');

    const server = app.listen(PORT, () => {
      console.log(`🚀 FreshTrack Backend running on port ${PORT}`);
      console.log(`✅ Health: http://localhost:${PORT}/health`);
    });

    process.on('SIGTERM', () => {
      console.log('🛑 SIGTERM received, shutting down gracefully...');
      server.close(() => {
        console.log('👋 Server closed.');
        process.exit(0);
      });
    });
  } catch (err) {
    console.error('❌ Startup failed:', err);
    process.exit(1);
  }
}

startServer();
