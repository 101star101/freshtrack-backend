// services/imageProcessor.js
const fs = require("fs");
const ort = require("onnxruntime-node");
const path = require("path");

const MODEL_PATH = process.env.MODEL_PATH || path.join(__dirname, "../models/best.onnx");
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.5;
const NMS_THRESHOLD = parseFloat(process.env.NMS_THRESHOLD) || 0.4;

class ImageProcessor {
  constructor() {
    this.model = null;
  }

  // ✅ Load ONNX model once
  async loadModel() {
    if (!this.model) {
      console.log(`📦 Loading YOLO model from: ${MODEL_PATH}`);
      try {
        this.model = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ["cpuExecutionProvider"], // safe for Render
        });
        console.log("✅ YOLO model loaded successfully!");
      } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
        throw new Error(`Cannot load ONNX model at ${MODEL_PATH}`);
      }
    }
  }

  // ✅ Prepare image into Float32 tensor
  async preprocessImage(imagePath) {
    const jimpModule = await import("jimp");
    const Jimp = jimpModule.Jimp;

    if (!fs.existsSync(imagePath)) {
      throw new Error(`Image file not found: ${imagePath}`);
    }

    console.log("🖼️ Preprocessing image:", imagePath);
    const image = await Jimp.read(imagePath);

    // Resize safely for YOLO
    await image.resize({ w: 640, h: 640 });

    // Convert image to Float32 tensor [1, 3, 640, 640]
    const input = new Float32Array(3 * 640 * 640);
    let i = 0;

    for (let y = 0; y < 640; y++) {
      for (let x = 0; x < 640; x++) {
        const { r, g, b } = Jimp.intToRGBA(image.getPixelColor(x, y));
        input[i++] = r / 255;
        input[i++] = g / 255;
        input[i++] = b / 255;
      }
    }

    console.log("✅ Image preprocessing complete.");
    return new ort.Tensor("float32", input, [1, 3, 640, 640]);
  }

  // ✅ Run inference and postprocess
  async detectObjects(imagePath) {
    await this.loadModel();
    const inputTensor = await this.preprocessImage(imagePath);

    let results;
    try {
      results = await this.model.run({ images: inputTensor });
    } catch (err) {
      console.error("❌ Inference failed:", err);
      throw new Error("ONNX inference failed");
    }

    const output = results[Object.keys(results)[0]];
    const detections = this.postprocess(output);
    return detections;
  }

  // ✅ Basic YOLO postprocessing
  postprocess(output) {
    const data = output.data;
    const numPredictions = output.dims[1];
    const numAttributes = output.dims[2];
    const detections = [];

    for (let i = 0; i < numPredictions; i++) {
      const offset = i * numAttributes;
      const x = data[offset];
      const y = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];
      const conf = data[offset + 4];

      if (conf >= CONFIDENCE_THRESHOLD) {
        detections.push({ x, y, width: w, height: h, confidence: conf });
      }
    }

    return this.nonMaxSuppression(detections, NMS_THRESHOLD);
  }

  // ✅ Non-Max Suppression (filter overlaps)
  nonMaxSuppression(boxes, threshold) {
    if (boxes.length === 0) return [];

    boxes.sort((a, b) => b.confidence - a.confidence);
    const selected = [];

    const iou = (a, b) => {
      const xA = Math.max(a.x, b.x);
      const yA = Math.max(a.y, b.y);
      const xB = Math.min(a.x + a.width, b.x + b.width);
      const yB = Math.min(a.y + a.height, b.y + b.height);
      const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
      const boxAArea = a.width * a.height;
      const boxBArea = b.width * b.height;
      return interArea / (boxAArea + boxBArea - interArea);
    };

    while (boxes.length > 0) {
      const current = boxes.shift();
      selected.push(current);
      boxes = boxes.filter((b) => iou(current, b) < threshold);
    }

    return selected;
  }
}

const imageProcessor = new ImageProcessor();

// ✅ Preload model for Render startup
async function preloadModel() {
  await imageProcessor.loadModel();
  console.log("🧠 Model preloaded successfully (Render ready).");
}

// ✅ Main export
async function processImage(filePath) {
  try {
    const detections = await imageProcessor.detectObjects(filePath);
    console.log(`✅ Detection complete: ${detections.length} objects found`);
    return detections;
  } catch (error) {
    console.error("❌ Error during YOLO detection:", error);
    throw error;
  } finally {
    try {
      fs.unlinkSync(filePath);
    } catch {
      // ignore cleanup errors
    }
  }
}

module.exports = { processImage, preloadModel };
