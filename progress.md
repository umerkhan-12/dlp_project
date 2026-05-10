# Retinal Disease Classification - Progress Report

## Training Results (RTX 3060, 64GB RAM)

| Model | Accuracy | Kappa | ROC-AUC | Parameters |
|-------|----------|-------|---------|------------|
| EfficientNet-B4 | 74.00% | 0.8142 | 88.69% | 19.3M |
| ResNet-50 | 63.09% | 0.7743 | 85.81% | 25.6M |
| ViT-B/16 | 59.45% | 0.5337 | 80.82% | 85.8M |

## Web Application

- **Backend**: FastAPI with `/predict` endpoint
- **Frontend**: React with drag-drop image upload
- **Model**: EfficientNet-B4 (best performer)

## Usage

```bash
# Backend
pip install -r api/requirements.txt
uvicorn api.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```
