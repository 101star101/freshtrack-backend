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
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(morgan('combined'));
app.use(cors({
  origin: true, // Allow all origins in development
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'FreshTrack Backend API',
    version: '1.0.0',
    endpoints: {
      '/api/detect': 'POST - Detect food items in image',
      '/api/storage/:item': 'GET - Get storage info for specific item',
      '/api/storage': 'GET - Get all storage data'
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Test endpoint for debugging
app.get('/api/test', (req, res) => {
  try {
    const testData = {
      message: 'Test endpoint working',
      timestamp: new Date().toISOString(),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        memory: process.memoryUsage(),
        env: {
          NODE_ENV: process.env.NODE_ENV,
          PORT: process.env.PORT,
          FORCE_MOCK_MODE: process.env.FORCE_MOCK_MODE
        }
      }
    };
    res.json(testData);
  } catch (error) {
    res.status(500).json({ error: 'Test endpoint failed', details: error.message });
  }
});

// Main detection endpoint
app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    console.log('Processing image:', req.file.filename);
    
    // Process the image and get detections
    const detections = await processImage(req.file.path);
    
    // Add storage information to each detection
    const enrichedDetections = detections.map(detection => {
      const storageInfo = getStorageData(detection.label);
      return {
        ...detection,
        storage: storageInfo
      };
    });

    res.json({
      success: true,
      detections: enrichedDetections,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ 
      error: 'Failed to process image',
      details: error.message 
    });
  }
});

// Get storage data for specific item
app.get('/api/storage/:item', (req, res) => {
  try {
    const item = req.params.item.toLowerCase();
    const storageInfo = getStorageData(item);
    
    if (!storageInfo) {
      return res.status(404).json({ error: 'Item not found in storage database' });
    }
    
    res.json({ success: true, item, storage: storageInfo });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get storage data' });
  }
});

// Get all storage data
app.get('/api/storage', (req, res) => {
  try {
    const allStorageData = require('./data/storage_data.json');
    res.json({ success: true, data: allStorageData });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get storage data' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

app.listen(PORT, () => {
  console.log(`FreshTrack Backend running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
