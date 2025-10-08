const ort = require('onnxruntime-web');
const Jimp = require("jimp");
const path = require("path");
const fs = require("fs");

class ImageProcessor {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
  }

  // Load the ONNX model
  async loadModel() {
    console.log("📦 Loading YOLO model from:", this.modelPath);
    try {
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ["cpuExecutionProvider"],
      });
      console.log("✅ Model loaded successfully");
    } catch (error) {
      console.error("❌ Failed to load ONNX model:", error);
      throw new Error(`Cannot load ONNX model at ${this.modelPath}`);
    }
  }

  // Preprocess image to tensor
  async preprocessImage(imagePath) {
    console.log("🖼️ Preprocessing image:", imagePath);

    const image = await Jimp.read(imagePath);
    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const inputWidth = 640;
    const inputHeight = 640;

    // Resize and normalize image
    image.resize(inputWidth, inputHeight);
    const imageData = new Float32Array(3 * inputWidth * inputHeight);

    let idx = 0;
    for (let y = 0; y < inputHeight; y++) {
      for (let x = 0; x < inputWidth; x++) {
        const pixel = image.getPixelColor(x, y);

        // Compatible fix for intToRGBA (works for both old/new Jimp)
        const rgba = Jimp.intToRGBA
          ? Jimp.intToRGBA(pixel)
          : {
              r: (pixel >> 24) & 255,
              g: (pixel >> 16) & 255,
              b: (pixel >> 8) & 255,
              a: pixel & 255,
            };

        imageData[idx++] = rgba.r / 255.0;
        imageData[idx++] = rgba.g / 255.0;
        imageData[idx++] = rgba.b / 255.0;
      }
    }

    const tensor = new ort.Tensor("float32", imageData, [1, 3, inputHeight, inputWidth]);
    return { tensor, width, height };
  }

  // Run detection
  async detectObjects(imagePath) {
    if (!this.session) throw new Error("ONNX model not loaded yet.");

    const { tensor, width, height } = await this.preprocessImage(imagePath);

    const feeds = { images: tensor };
    const results = await this.session.run(feeds);

    const output = results[Object.keys(results)[0]];
    console.log("✅ YOLO detection completed");

    // Convert detections to readable boxes
    return this.postprocess(output, width, height);
  }

  // Basic postprocess (you can adjust as needed)
  postprocess(output, width, height) {
    const boxes = [];

    // Example: parse YOLOv5-like output
    const [batch, channels, numDetections] = output.dims;
    const data = output.data;

    for (let i = 0; i < numDetections; i++) {
      const x = data[i * channels + 0];
      const y = data[i * channels + 1];
      const w = data[i * channels + 2];
      const h = data[i * channels + 3];
      const conf = data[i * channels + 4];

      if (conf > 0.3) {
        boxes.push({
          x: x * width,
          y: y * height,
          w: w * width,
          h: h * height,
          confidence: conf,
        });
      }
    }
    return boxes;
  }
}

// Create a shared instance
const modelPath = path.join(__dirname, "../models/best.onnx");
const processor = new ImageProcessor(modelPath);

// Called before starting the server (preload model)
async function preloadModel() {
  await processor.loadModel();
  return processor;
}

async function processImage(imagePath) {
  try {
    return await processor.detectObjects(imagePath);
  } catch (error) {
    console.error("❌ Error during YOLO detection:", error);
    throw error;
  }
}

module.exports = { processImage, preloadModel };
