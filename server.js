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

// 🧩 Load environment variables with fallbacks
const PORT = process.env.PORT || 3000;
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, 'uploads');
const LOG_FORMAT = process.env.LOG_FORMAT || 'dev';
const CORS_ORIGIN = process.env.CORS_ORIGIN || '*';

// ==============================
// 🌐 Middleware
// ==============================
app.use(helmet());
app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json({ limit: process.env.MAX_UPLOAD_SIZE || '10mb' }));
app.use(morgan(LOG_FORMAT));

// ==============================
// 📁 Uploads folder setup
// ==============================
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ==============================
// 📸 Multer setup
// ==============================
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}.jpg`;
    cb(null, uniqueName);
  },
});
const upload = multer({ storage });

// ==============================
// 💓 Health & Root routes
// ==============================
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.get('/', (req, res) => {
  res.send('🚀 FreshTrack Backend is running!');
});

// ==============================
// 🧠 Detection route
// ==============================
app.post('/api/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      console.error('⚠️ No image uploaded.');
      return res.status(400).json({ success: false, message: 'No image uploaded.' });
    }

    const imagePath = path.join(UPLOAD_DIR, req.file.filename);
    console.log(`🖼️ Preprocessing image: ${imagePath}`);

    // Run model detection
    const detections = await processImage(imagePath);

    // Attach storage info
    for (const detection of detections) {
      detection.storage = getStorageData(detection.label);
    }

    res.status(200).json({
      success: true,
      detections,
      timestamp: new Date().toISOString(),
    });

    // 🧹 Safe cleanup
    if (fs.existsSync(imagePath)) {
      fs.unlink(imagePath, (err) => {
        if (err) console.warn('⚠️ Could not delete uploaded file:', err.message);
        else console.log(`🧼 Deleted uploaded file: ${imagePath}`);
      });
    } else {
      console.log(`ℹ️ Uploaded file already deleted or missing: ${imagePath}`);
    }
  } catch (error) {
    console.error('❌ Detection error:', error);
    res.status(500).json({ success: false, message: 'Error processing image', error: error.message });
  }
});

// ==============================
// 🗂️ Storage routes
// ==============================
app.get('/api/storage/:itemName', (req, res) => {
  const { itemName } = req.params;
  const data = getStorageData(itemName);
  if (!data) return res.status(404).json({ success: false, message: 'Item not found' });
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

// ==============================
// 🕓 Daily upload cleanup
// ==============================
setInterval(() => {
  fs.readdir(UPLOAD_DIR, (err, files) => {
    if (err) return console.error('Error reading uploads directory:', err);
    const now = Date.now();
    const DAY_MS = 24 * 60 * 60 * 1000;
    files.forEach((file) => {
      const filePath = path.join(UPLOAD_DIR, file);
      fs.stat(filePath, (err, stats) => {
        if (!err && now - stats.mtimeMs > DAY_MS) {
          fs.unlink(filePath, () => console.log(`🧽 Deleted old upload: ${file}`));
        }
      });
    });
  });
}, 24 * 60 * 60 * 1000);

// ==============================
// 🚀 Start server
// ==============================
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
