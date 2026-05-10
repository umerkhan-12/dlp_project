"""
FastAPI Backend for Retinal Disease Classification
"""

import os
import io
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_model

MODEL_DIR = Path(__file__).parent.parent / "experiments"
CLASS_NAMES = ["No_DR", "Mild_DR", "Moderate_DR", "Severe_DR", "Proliferative_DR"]
CLASS_DESCRIPTIONS = {
    "No_DR": "No Diabetic Retinopathy - Healthy retina",
    "Mild_DR": "Mild DR - Early stage",
    "Moderate_DR": "Moderate DR - Multiple hemorrhages",
    "Severe_DR": "Severe DR - Extensive hemorrhages",
    "Proliferative_DR": "Proliferative DR - Advanced stage"
}

app = FastAPI(title="Retinal Disease Classification API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = None
device = None
transform = None

def get_val_transforms(image_size: int = 224):
    return A.Compose([A.Resize(width=image_size, height=image_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

def load_model():
    global model, device, transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = list(MODEL_DIR.glob("*/checkpoints/best_model.pth"))
    if not model_paths:
        print(f"Warning: No model found in {MODEL_DIR}")
        return False
    BEST_MODEL = model_paths[0]
    try:
        checkpoint = torch.load(BEST_MODEL, map_location=device, weights_only=False)
        model = create_model(model_name="efficientnet_b4", num_classes=5, pretrained=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        transform = get_val_transforms()
        print(f"✓ Model loaded from {BEST_MODEL}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
async def root():
    return {"message": "Retinal Disease Classification API", "model_loaded": model is not None, "endpoints": {"/predict": "Single image", "/predict/batch": "Multiple images", "/health": "Health check", "/classes": "Class info"}}

@app.get("/health")
async def health():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}

@app.get("/classes")
async def get_classes():
    return {"classes": CLASS_NAMES, "descriptions": CLASS_DESCRIPTIONS}

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    transformed = transform(image=image_array)
    return transformed["image"].unsqueeze(0)

def predict(image_tensor: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        return {"predicted_class": predicted_class, "class_name": CLASS_NAMES[predicted_class], "confidence": float(probabilities[predicted_class]), "probabilities": {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities.cpu().numpy())}}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        result = predict(image_tensor)
        result["description"] = CLASS_DESCRIPTIONS[result["class_name"]]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
