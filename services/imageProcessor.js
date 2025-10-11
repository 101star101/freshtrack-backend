const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node"); // ✅ Use node version for backend
let sharp;

// Optional sharp load
try {
  sharp = require("sharp");
} catch {
  console.warn("⚠️ sharp not available — will use Jimp fallback");
}

// Paths and constants
const MODEL_PATH = process.env.MODEL_PATH || path.join(__dirname, "../models/best.onnx");
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.5;
const NMS_THRESHOLD = parseFloat(process.env.NMS_THRESHOLD) || 0.4;

// ✅ Class names (match your training order)
const CLASS_NAMES = [
  "Fresh_Apple", "Fresh_Banana", "Fresh_Beef", "Fresh_Carrot", "Fresh_Chicken",
  "Fresh_Cucumber", "Fresh_Manggo", "Fresh_Okra", "Fresh_Orange", "Fresh_Pepper",
  "Fresh_Pork", "Fresh_Potato", "Fresh_Strawberry",
  "Rotten_Apple", "Rotten_Banana", "Rotten_Beef", "Rotten_Carrot", "Rotten_Chicken",
  "Rotten_Cucumber", "Rotten_Manggo", "Rotten_Okra", "Rotten_Orange", "Rotten_Pepper",
  "Rotten_Pork", "Rotten_Potato", "Rotten_Strawberry"
];

class ImageProcessor {
  constructor() {
    this.model = null;
    this.modelLoaded = false;
  }

  // ✅ Load YOLO model only once
  async loadModel() {
    if (this.modelLoaded && this.model) return this.model;

    console.log(`📦 Loading YOLO model from: ${MODEL_PATH}`);
    try {
      this.model = await ort.InferenceSession.create(MODEL_PATH);
      this.modelLoaded = true;
      console.log("✅ YOLO model loaded successfully!");
    } catch (err) {
      console.error("❌ Failed to load ONNX model:", err);
      throw new Error(`Cannot load ONNX model at ${MODEL_PATH}`);
    }
    return this.model;
  }

  // ✅ Preprocess image to tensor
  async preprocessImage(imagePath) {
    console.log("🖼️ Preprocessing image:", imagePath);
    const width = 416;
    const height = 416;
    let imageData;

    if (sharp) {
      try {
        const img = await sharp(imagePath)
          .resize(width, height, { fit: "fill" })
          .ensureAlpha()
          .removeAlpha()
          .toColorspace("rgb")
          .raw()
          .toBuffer();

        imageData = new Float32Array(width * height * 3);
        for (let i = 0; i < img.length; i++) {
          imageData[i] = img[i] / 255.0;
        }
      } catch (err) {
        console.error("❌ sharp processing failed, fallback to Jimp:", err);
        imageData = await this.jimpFallback(imagePath, width, height);
      }
    } else {
      imageData = await this.jimpFallback(imagePath, width, height);
    }

    return new ort.Tensor("float32", imageData, [1, 3, height, width]);
  }

  async jimpFallback(imagePath, width, height) {
    const jimpModule = await import("jimp");
    const Jimp = jimpModule.default;
    const image = await Jimp.read(imagePath);
    await image.resize(width, height);

    const data = new Float32Array(3 * width * height);
    let i = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const { r, g, b } = Jimp.intToRGBA(image.getPixelColor(x, y));
        data[i++] = r / 255;
        data[i++] = g / 255;
        data[i++] = b / 255;
      }
    }
    return data;
  }

  // ✅ Run inference
  async detectObjects(imagePath) {
    const model = await this.loadModel();
    const inputTensor = await this.preprocessImage(imagePath);

    let results;
    try {
      results = await model.run({ images: inputTensor });
    } catch (err) {
      console.error("❌ Inference failed:", err);
      throw new Error("ONNX inference failed");
    }

    const output = results[Object.keys(results)[0]];
    return this.postprocess(output);
  }

  // ✅ Postprocess
  postprocess(output) {
    const data = output.data;
    const [batch, numPredictions, numAttributes] = output.dims;
    const detections = [];

    for (let i = 0; i < numPredictions; i++) {
      const offset = i * numAttributes;
      const x = data[offset];
      const y = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];
      const conf = data[offset + 4];

      if (conf >= CONFIDENCE_THRESHOLD) {
        let bestClass = null;
        let bestScore = 0;
        for (let j = 5; j < numAttributes; j++) {
          if (data[offset + j] > bestScore) {
            bestScore = data[offset + j];
            bestClass = j - 5;
          }
        }
        const label = CLASS_NAMES[bestClass] || `Unknown_${bestClass}`;
        detections.push({ x, y, width: w, height: h, confidence: conf, class_id: bestClass, label });
      }
    }

    return this.nonMaxSuppression(detections, NMS_THRESHOLD);
  }

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

async function processImage(filePath) {
  try {
    const detections = await imageProcessor.detectObjects(filePath);
    console.log(`✅ Detection complete: ${detections.length} objects found`);
    return detections;
  } catch (error) {
    console.error("❌ Detection error:", error);
    throw error;
  } finally {
    try {
      fs.unlinkSync(filePath);
    } catch {
      console.warn("ℹ️ Uploaded file already deleted or missing:", filePath);
    }
  }
}

async function preloadModel() {
  await imageProcessor.loadModel();
}

module.exports = { processImage, preloadModel };
