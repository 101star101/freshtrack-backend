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
const PORT = process.env.PORT || 10000;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(helmet());
app.use(morgan('dev'));

// File upload setup
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

// Health check
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    uptime: process.uptime(),
    timestamp: new Date(),
  });
});

// ✅ Food detection route (with /api prefix)
app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No image uploaded' });
    }

    console.log('🖼️ Received image:', req.file.filename);
    const imagePath = req.file.path;

    // Process image for detections
    console.log('🖼️ Preprocessing image:', imagePath);
    const detections = await processImage(imagePath);

    console.log(`✅ Detection complete: ${detections.length} objects found`);

    // Delete image after processing (optional)
    try {
      fs.unlinkSync(imagePath);
    } catch (err) {
      console.warn('⚠️ Failed to delete uploaded file:', err.message);
    }

    return res.json({
      success: true,
      detections,
      timestamp: new Date(),
    });
  } catch (error) {
    console.error('❌ Detection error:', error);
    return res.status(500).json({
      success: false,
      message: 'Error processing image',
      error: error.message,
    });
  }
});

// ✅ Get storage info for one item
app.get('/api/storage/:itemName', async (req, res) => {
  try {
    const { itemName } = req.params;
    if (!itemName) {
      return res.status(400).json({ success: false, message: 'Item name required' });
    }

    console.log(`📦 Fetching storage info for: ${itemName}`);
    const storageData = await getStorageData(itemName);

    if (!storageData) {
      return res.status(404).json({ success: false, message: 'Item not found' });
    }

    return res.json({
      success: true,
      storage: storageData,
    });
  } catch (error) {
    console.error('❌ Storage info error:', error);
    return res.status(500).json({
      success: false,
      message: 'Error fetching storage info',
      error: error.message,
    });
  }
});

// ✅ Get all storage data
app.get('/api/storage', async (req, res) => {
  try {
    console.log('📦 Fetching all storage data');
    const allData = await getStorageData();
    res.json({
      success: true,
      data: allData,
    });
  } catch (error) {
    console.error('❌ Error fetching all storage data:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching all storage data',
    });
  }
});

// Default route
app.get('/', (req, res) => {
  res.send('🍏 FreshTrack backend is running successfully!');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
