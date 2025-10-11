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

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// Uploads folder setup
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// Multer setup for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}.jpg`;
    cb(null, uniqueName);
  },
});
const upload = multer({ storage });

// ✅ Health check route
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});

// ✅ Root route
app.get('/', (req, res) => {
  res.send('🚀 FreshTrack Backend is running!');
});

// ✅ Detection route
app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      console.error('⚠️ No image uploaded.');
      return res.status(400).json({ success: false, message: 'No image uploaded.' });
    }

    const imagePath = path.join(uploadDir, req.file.filename);
    console.log(`🖼️ Preprocessing image: ${imagePath}`);

    // Run model detection
    const detections = await processImage(imagePath);

    // Attach storage info
    for (const detection of detections) {
      detection.storage = getStorageData(detection.label);
    }

    // ✅ Response to client
    res.status(200).json({
      success: true,
      detections,
      timestamp: new Date().toISOString(),
    });

    // 🧹 Safe cleanup of uploaded file
    if (fs.existsSync(imagePath)) {
      try {
        fs.unlinkSync(imagePath);
        console.log(`🧼 Deleted uploaded file: ${imagePath}`);
      } catch (err) {
        console.warn('⚠️ Could not delete uploaded file:', err.message);
      }
    } else {
      console.log(`ℹ️ Uploaded file already deleted or missing: ${imagePath}`);
    }
  } catch (error) {
    console.error('❌ Detection error:', error);
    res.status(500).json({ success: false, message: 'Error processing image', error: error.message });
  }
});

// ✅ Storage data routes
app.get('/api/storage/:itemName', (req, res) => {
  const { itemName } = req.params;
  const data = getStorageData(itemName);
  if (!data) {
    console.warn(`⚠️ No storage data found for: ${itemName}`);
    return res.status(404).json({ success: false, message: 'Item not found' });
  }
  res.json({ success: true, storage: data });
});

app.get('/api/storage', (req, res) => {
  try {
    const data = getStorageData();
    res.json({ success: true, data });
  } catch (error) {
    console.error('❌ Error fetching storage data:', error);
    res.status(500).json({ success: false, message: 'Error fetching storage data' });
  }
});

// 🕓 Auto cleanup of old uploads (every 24h)
setInterval(() => {
  fs.readdir(uploadDir, (err, files) => {
    if (err) {
      console.error('Error reading uploads directory:', err);
      return;
    }

    const now = Date.now();
    const DAY_MS = 24 * 60 * 60 * 1000;

    files.forEach((file) => {
      const filePath = path.join(uploadDir, file);
      fs.stat(filePath, (err, stats) => {
        if (err) return;
        if (now - stats.mtimeMs > DAY_MS) {
          fs.unlink(filePath, (err) => {
            if (!err) console.log(`🧽 Deleted old upload: ${file}`);
          });
        }
      });
    });
  });
}, 24 * 60 * 60 * 1000);

// ✅ Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
