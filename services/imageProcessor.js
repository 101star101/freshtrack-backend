const fs = require("fs");
const ort = require("onnxruntime-node");
const Jimp = require("jimp");

const MODEL_PATH = process.env.MODEL_PATH || "./models/best.onnx";
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.5;
const NMS_THRESHOLD = parseFloat(process.env.NMS_THRESHOLD) || 0.4;

class ImageProcessor {
  constructor() {
    this.model = null;
  }

  async loadModel() {
    if (!this.model) {
      console.log(`Loading YOLO model from ${MODEL_PATH}...`);
      this.model = await ort.InferenceSession.create(MODEL_PATH);
      console.log("✅ YOLO model loaded successfully!");
    }
  }

  // ✅ Fixed and stable image pre-processing
  async preprocessImage(imagePath) {
    const buffer = fs.readFileSync(imagePath);
    const image = await Jimp.read(buffer);

    // Resize image to 640x640 and normalize to [0,1]
    image.resize(640, 640);
    const input = new Float32Array(3 * 640 * 640);
    let i = 0;
    for (let y = 0; y < 640; y++) {
      for (let x = 0; x < 640; x++) {
        const { r, g, b } = Jimp.intToRGBA(image.getPixelColor(x, y));
        input[i++] = r / 255.0;
        input[i++] = g / 255.0;
        input[i++] = b / 255.0;
      }
    }

    return new ort.Tensor("float32", input, [1, 3, 640, 640]);
  }

  async detectObjects(imagePath) {
    await this.loadModel();

    const inputTensor = await this.preprocessImage(imagePath);
    const feeds = { images: inputTensor };

    const results = await this.model.run(feeds);
    const output = results[Object.keys(results)[0]];
    const detections = this.postprocess(output);

    return detections;
  }

  // ✅ Multi-object detection with NMS
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
        detections.push({
          x,
          y,
          width: w,
          height: h,
          confidence: conf,
        });
      }
    }

    // Apply Non-Maximum Suppression to avoid overlapping boxes
    return this.nonMaxSuppression(detections, NMS_THRESHOLD);
  }

  // ✅ Basic NMS implementation
  nonMaxSuppression(boxes, threshold) {
    if (boxes.length === 0) return [];

    boxes.sort((a, b) => b.confidence - a.confidence);

    const selected = [];
    const iou = (boxA, boxB) => {
      const xA = Math.max(boxA.x, boxB.x);
      const yA = Math.max(boxA.y, boxB.y);
      const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
      const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

      const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
      const boxAArea = boxA.width * boxA.height;
      const boxBArea = boxB.width * boxB.height;

      return interArea / (boxAArea + boxBArea - interArea);
    };

    while (boxes.length > 0) {
      const current = boxes.shift();
      selected.push(current);
      boxes = boxes.filter((box) => iou(current, box) < threshold);
    }

    return selected;
  }
}

const imageProcessor = new ImageProcessor();

async function processImage(filePath) {
  try {
    const results = await imageProcessor.detectObjects(filePath);
    console.log(`✅ Detection complete: ${results.length} objects found`);
    return results;
  } catch (error) {
    console.error("❌ Error during YOLO detection:", error);
    throw error;
  } finally {
    try {
      fs.unlinkSync(filePath); // Clean up upload
    } catch {}
  }
}

module.exports = { processImage };
