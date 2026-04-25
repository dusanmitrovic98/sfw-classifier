# 🛠️ MainframeForge SFW Classifier

A lightweight, high-performance image safety classification service. 

## 🚀 Overview
Optimized for **Render Free Tier (512MB RAM)**. This service uses an **Int8 Quantized ONNX** version of the Falconsai ViT model to ensure fast inference on CPU-only hardware with a minimal memory footprint.

### Features
- **UI:** Custom dashboard for manual testing.
- **API:** REST endpoint at `/api/classify` for system integrations.
- **Format:** ONNX Runtime (Quantized).

## 🛠️ Installation
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

