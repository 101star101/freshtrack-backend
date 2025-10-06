const path = require('path');
const fs = require('fs');

// Mock mode only - no external ML dependencies
console.log('Running in mock detection mode - no ML dependencies required');

// Food class names (matching your YOLO model)
const CLASS_NAMES = [
  // Fresh items
  'Fresh_Apple', 'Fresh_Banana', 'Fresh_Potato', 'Fresh_Carrot', 'Fresh_Orange',
  'Fresh_Beef', 'Fresh_Chicken', 'Fresh_Pork', 'Fresh_Manggo', 'Fresh_Pepper',
  'Fresh_Cucumber', 'Fresh_Strawberry', 'Fresh_Okra',
  // Rotten items
  'Rotten_Apple', 'Rotten_Banana', 'Rotten_Potato', 'Rotten_Carrot', 'Rotten_Orange',
  'Rotten_Beef', 'Rotten_Chicken', 'Rotten_Pork', 'Rotten_Manggo', 'Rotten_Pepper',
  'Rotten_Cucumber', 'Rotten_Strawberry', 'Rotten_Okra'
];

class ImageProcessor {
  constructor() {
    this.confidenceThreshold = 0.5;
    this.nmsThreshold = 0.4;
    
    // Always use mock mode for Railway deployment
    this.forceMockMode = true;
    console.log('ImageProcessor initialized in mock mode');
  }

  async initialize() {
    // Always return false to use mock mode
    console.log('Initializing in mock mode - no ML model required');
    return false;
  }

  async preprocessImage(imagePath) {
    // Mock preprocessing - just check if file exists
    try {
      if (!fs.existsSync(imagePath)) {
        throw new Error('Image file not found');
      }
      
      const stats = fs.statSync(imagePath);
      console.log('Image file size:', stats.size, 'bytes');
      
      // Return mock tensor data
      return new Float32Array(640 * 640 * 3).fill(0.5);
    } catch (error) {
      console.error('Error in mock preprocessing:', error);
      throw error;
    }
  }

  async detectObjects(imagePath) {
    try {
      console.log('Starting mock object detection for:', imagePath);
      
      // Always use mock detections for Railway
      console.log('Using mock detections (Railway deployment mode)');
      return this.getMockDetections();
      
    } catch (error) {
      console.error('Error in mock object detection:', error);
      console.error('Error stack:', error.stack);
      console.log('Falling back to mock detections due to error');
      return this.getMockDetections();
    }
  }

  // Mock YOLO output processing - not used in mock mode
  processYOLOOutput(outputTensor) {
    console.log('Mock YOLO output processing called');
    return this.getMockDetections();
  }

  // Mock NMS and IoU methods - not used in mock mode
  applyNMS(detections) {
    console.log('Mock NMS called');
    return detections;
  }

  calculateIoU(box1, box2) {
    console.log('Mock IoU calculation called');
    return 0.1; // Return low IoU to avoid filtering
  }

  getMockDetections() {
    // Mock detections for development/testing
    const mockItems = [
      'Fresh_Apple', 'Fresh_Banana', 'Fresh_Carrot', 'Fresh_Orange',
      'Rotten_Apple', 'Rotten_Banana', 'Rotten_Potato', 'Rotten_Chicken'
    ];
    const detections = [];
    
    for (let i = 0; i < Math.floor(Math.random() * 3) + 1; i++) {
      const randomItem = mockItems[Math.floor(Math.random() * mockItems.length)];
      detections.push({
        label: randomItem,
        confidence: 0.8 + Math.random() * 0.2,
        bbox: {
          x: Math.random() * 0.6,
          y: Math.random() * 0.6,
          width: 0.1 + Math.random() * 0.2,
          height: 0.1 + Math.random() * 0.2
        }
      });
    }
    
    return detections;
  }
}

const imageProcessor = new ImageProcessor();

async function processImage(imagePath) {
  try {
    console.log('Starting image processing for:', imagePath);
    
    // Check if file exists
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Image file not found: ${imagePath}`);
    }
    
    // Check file size
    const stats = fs.statSync(imagePath);
    console.log('Image file size:', stats.size, 'bytes');
    
    if (stats.size === 0) {
      throw new Error('Image file is empty');
    }
    
    const detections = await imageProcessor.detectObjects(imagePath);
    console.log('Detections completed:', detections.length, 'objects found');
    
    // Clean up uploaded file
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
      console.log('Cleaned up uploaded file');
    }
    
    return detections;
  } catch (error) {
    console.error('Error processing image:', error);
    console.error('Error stack:', error.stack);
    
    // Return mock detections as fallback
    console.log('Falling back to mock detections due to error');
    return imageProcessor.getMockDetections();
  }
}

module.exports = {
  processImage,
  ImageProcessor
};
