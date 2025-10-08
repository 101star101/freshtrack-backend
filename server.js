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

// Middleware
app.use(cors());
app.use(helmet());
app.use(morgan('dev'));
app.use(express.json());

// Multer setup
const upload = multer({ dest: 'uploads/' });

// Routes
app.get('/', (req, res) => {
  res.json({ message: '🟢 FreshTrack Backend is running!' });
});

app.post('/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded.' });
    }

    const imagePath = path.resolve(req.file.path);
    const results = await processImage(imagePath);

    // ✅ Clean up uploaded image after processing
    fs.unlink(imagePath, (err) => {
      if (err) console.warn('⚠️ Failed to delete uploaded file:', err.message);
    });

    // ✅ Add fallback if storage data is missing
    const enhancedResults = results.map((item) => {
      const storageInfo = getStorageData(item.class);

      if (!storageInfo) {
        console.warn(`⚠️ No storage data found for: ${item.class}`);

        return {
          ...item,
          storage_info: {
            storage: 'Unknown - please verify manually',
            shelf_life: 'N/A',
            tips: 'No data available for this item',
            status: 'Unknown',
            waste_disposal: 'Dispose safely if unsure'
          }
        };
      }

      return { ...item, storage_info: storageInfo };
    });

    res.json({
      success: true,
      message: 'Image analyzed successfully.',
      detections: enhancedResults
    });
  } catch (error) {
    console.error('❌ Error processing image:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error during image processing.',
      error: error.message
    });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Server listen
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
