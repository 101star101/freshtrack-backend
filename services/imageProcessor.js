const ort = require("onnxruntime-node");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

// Load YOLO model once
const modelPath = path.join(__dirname, "../models/best.onnx");
let session = null;

async function loadModel() {
  if (!session) {
    console.log(`📦 Loading YOLO model from: ${modelPath}`);
    session = await ort.InferenceSession.create(modelPath);
    console.log("✅ YOLO model loaded successfully!");
  }
  return session;
}

// Map YOLO class indices → human-readable names
const CLASS_NAMES = [
  "Fresh_Apple",
  "Fresh_Banana",
  "Fresh_Beef",
  "Fresh_Carrot",
  "Fresh_Chicken",
  "Fresh_Cucumber",
  "Fresh_Manggo",
  "Fresh_Okra",
  "Fresh_Orange",
  "Fresh_Pepper",
  "Fresh_Pork",
  "Fresh_Potato",
  "Fresh_Strawberry",
  "Rotten_Apple",
  "Rotten_Banana",
  "Rotten_Beef",
  "Rotten_Carrot",
  "Rotten_Chicken",
  "Rotten_Cucumber",
  "Rotten_Manggo",
  "Rotten_Okra",
  "Rotten_Orange",
  "Rotten_Pepper",
  "Rotten_Pork",
  "Rotten_Potato",
  "Rotten_Strawberry"
];

async function processImage(imagePath) {
  console.log(`🖼️ Preprocessing image: ${imagePath}`);

  const session = await loadModel();
  const imageBuffer = await sharp(imagePath).resize(640, 640).toBuffer();
  const float32Data = Float32Array.from(imageBuffer, v => v / 255.0);

  const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 640, 640]);
  const results = await session.run({ images: inputTensor });

  const output = results.output0 || Object.values(results)[0];
  const detections = parseDetections(output);

  // Clean up uploaded image after processing
  try {
    fs.unlinkSync(imagePath);
  } catch {
    console.log(`ℹ️ Uploaded file already deleted or missing: ${imagePath}`);
  }

  return detections;
}

function parseDetections(output) {
  const detections = [];
  const threshold = 0.25; // confidence threshold

  for (let i = 0; i < output.dims[1]; i++) {
    const conf = output.data[i * output.dims[2] + 4];
    if (conf < threshold) continue;

    // Get class with highest score
    const classScores = output.data.slice(i * output.dims[2] + 5, (i + 1) * output.dims[2]);
    const classId = classScores.indexOf(Math.max(...classScores));
    const className = CLASS_NAMES[classId] || `Unknown_${classId}`;

    detections.push({
      classId,
      className,
      confidence: conf,
    });
  }

  console.log(`✅ Detection complete: ${detections.length} objects found`);
  return detections;
}

module.exports = { processImage };
