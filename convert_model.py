#!/usr/bin/env python3
"""
Script to convert YOLO .pt model to ONNX format for FreshTrack backend
"""

import os
import sys
from pathlib import Path

def convert_yolo_to_onnx():
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not found. Please install it first:")
        print("   pip install ultralytics")
        return False
    
    # Check if best.pt exists
    pt_model_path = Path("best.pt")
    if not pt_model_path.exists():
        print("❌ best.pt not found in current directory")
        print("   Please place your best.pt file in the backend directory")
        return False
    
    try:
        print("🔄 Loading YOLO model...")
        model = YOLO("best.pt")
        
        print("🔄 Converting to ONNX format...")
        # Export to ONNX
        success = model.export(format='onnx', opset=11, simplify=True)
        
        if success:
            print("✅ Model converted successfully!")
            print("📁 ONNX model saved as: best.onnx")
            
            # Move to models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            onnx_path = Path("best.onnx")
            if onnx_path.exists():
                import shutil
                shutil.move("best.onnx", "models/best.onnx")
                print("📁 Model moved to: models/best.onnx")
            
            return True
        else:
            print("❌ Failed to convert model")
            return False
            
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 YOLO to ONNX Converter for FreshTrack")
    print("=" * 50)
    
    success = convert_yolo_to_onnx()
    
    if success:
        print("\n🎉 Conversion complete!")
        print("📋 Next steps:")
        print("   1. Start your backend: npm run dev")
        print("   2. The system will now use your YOLO model for detection")
    else:
        print("\n💡 Troubleshooting:")
        print("   1. Make sure best.pt is in the backend directory")
        print("   2. Install ultralytics: pip install ultralytics")
        print("   3. Check that your model is compatible with YOLO v8")
