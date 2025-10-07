const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');

// Food classes (match your training)
const CLASS_NAMES = [
  'Fresh_Apple', 'Fresh_Banana', 'Fresh_Potato', 'Fresh_Carrot', 'Fresh_Orange',
  'Fresh_Beef', 'Fresh_Chicken', 'Fresh_Pork', 'Fresh_Manggo', 'Fresh_Pepper',
  'Fresh_Cucumber', 'Fresh_Strawberry', 'Fresh_Okra',
  'Rotten_Apple', 'Rotten_Banana', 'Rotten_Potato', 'Rotten_Carrot', 'Rotten_Orange',
  'Rotten_Beef', 'Rotten_Chicken', 'Rotten_Pork', 'Rotten_Manggo', 'Rotten_Pepper',
  'Rotten_Cucumber', 'Rotten_Strawberry', 'Rotten_Okra'
];

class ImageProcessor {
  constructor() {
    this.modelPath = process.env.MODEL_PATH || path.join(__dirname, '../models/best.onnx');
    this.confidenceThreshold = 0.4;
    this.nmsThreshold = 0.45;
    this.session = null;
  }

  async initialize() {
    try {
      console.log('Loading YOLOv8 model from:', this.modelPath);
      if (!fs.existsSync(this.modelPath)) throw new Error('Model file not found');
      this.session = await ort.InferenceSession.create(this.modelPath);
      console.log('✅ YOLOv8 ONNX model loaded successfully.');
      return true;
    } catch (err) {
      console.error('❌ Failed to load model:', err);
      return false;
    }
  }

  async preprocessImage(imagePath) {
    const image = await Jimp.read(imagePath);
    image.resize(640, 640);
    const imgData = Float32Array.from(image.bitmap.data)
      .filter((_, i) => i % 4 !== 3)
      .map(v => v / 255.0);
    return new ort.Tensor('float32', imgData, [1, 3, 640, 640]);
  }

  async detectObjects(imagePath) {
    try {
      if (!this.session) {
        const ok = await this.initialize();
        if (!ok) throw new Error('Model not initialized');
      }

      const inputTensor = await this.preprocessImage(imagePath);
      const feeds = { images: inputTensor };
      const results = await this.session.run(feeds);
      const output = results[Object.keys(results)[0]];

      const detections = this.processYOLOOutput(output.data);
      return this.applyNMS(detections);
    } catch (err) {
      console.error('Error during YOLO detection:', err);
      return [];
    }
  }

  processYOLOOutput(output) {
    const detections = [];
    const numElements = output.length / (5 + CLASS_NAMES.length);
    for (let i = 0; i < numElements; i++) {
      const offset = i * (5 + CLASS_NAMES.length);
      const x = output[offset];
      const y = output[offset + 1];
      const w = output[offset + 2];
      const h = output[offset + 3];
      const conf = output[offset + 4];
      if (conf < this.confidenceThreshold) continue;
      const classScores = output.slice(offset + 5, offset + 5 + CLASS_NAMES.length);
      const classId = classScores.indexOf(Math.max(...classScores));
      detections.push({
        label: CLASS_NAMES[classId],
        confidence: conf,
        bbox: { x, y, width: w, height: h }
      });
    }
    return detections;
  }

  applyNMS(detections) {
    const result = [];
    detections.sort((a, b) => b.confidence - a.confidence);
    for (const det of detections) {
      if (!result.some(r => this.calculateIoU(r.bbox, det.bbox) > this.nmsThreshold)) {
        result.push(det);
      }
    }
    return result;
  }

  calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const unionArea = a.width * a.height + b.width * b.height - interArea;
    return interArea / unionArea;
  }
}

const processor = new ImageProcessor();

async function processImage(imagePath) {
  const detections = await processor.detectObjects(imagePath);
  try {
    fs.unlinkSync(imagePath);
  } catch {}
  return detections;
}

module.exports = { processImage };
