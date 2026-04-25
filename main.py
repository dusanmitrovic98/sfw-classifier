from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

app = FastAPI(title="MainframeForge SFW Classifier API")

# Load the lightweight ONNX model into memory (Runs on CPU, optimized for Render Free Tier)
# Make sure "onnx_model/model.onnx" exists in your directory!
# Update this line to match the new file name created by the script
session = ort.InferenceSession("onnx_model_quant/model_quantized.onnx", providers=['CPUExecutionProvider'])

def process_and_predict(image_bytes):
    """Preprocesses the image and runs ONNX inference"""
    # 1. Load and resize
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    
    # 2. Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # 3. Transpose to (1, Channels, Height, Width)
    input_data = np.expand_dims(img_array.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    # 4. Run Inference
    outputs = session.run(None, {"pixel_values": input_data})[0]
    
    # 5. Softmax to get probabilities
    exp_out = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_out / exp_out.sum()
    
    return {
        "normal": float(probs[0]), 
        "nsfw": float(probs[1])
    }

# ==========================================
# API ENDPOINT (For your local systems/integrations)
# ==========================================
@app.post("/api/classify")
async def classify_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        predictions = process_and_predict(image_bytes)
        
        # Determine safety status based on threshold
        is_safe = predictions["normal"] > predictions["nsfw"]
        
        return {
            "status": "success",
            "filename": file.filename,
            "is_safe": is_safe,
            "confidence": {
                "safe": round(predictions["normal"] * 100, 2),
                "nsfw": round(predictions["nsfw"] * 100, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# WEB UI ENDPOINT (For manual testing)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MainframeForge SFW Classifier</title>
        <style>
            :root {
                --bg-color: #0f1115;
                --card-bg: #1a1d24;
                --primary: #f97316; /* Forge Orange */
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --border: #374151;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                background-color: var(--card-bg);
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                width: 100%;
                max-width: 450px;
                border: 1px solid var(--border);
                text-align: center;
            }
            .brand {
                color: var(--primary);
                font-size: 1.2rem;
                font-weight: bold;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 5px;
            }
            h1 {
                margin-top: 0;
                font-size: 1.8rem;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed var(--border);
                border-radius: 8px;
                padding: 40px 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }
            .upload-area:hover {
                border-color: var(--primary);
                background-color: rgba(249, 115, 22, 0.05);
            }
            input[type="file"] {
                display: none;
            }
            .btn {
                background-color: var(--primary);
                color: #fff;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: background 0.3s ease;
            }
            .btn:hover {
                background-color: #ea580c;
            }
            .btn:disabled {
                background-color: var(--border);
                cursor: not-allowed;
            }
            #preview {
                max-width: 100%;
                max-height: 250px;
                margin-top: 15px;
                border-radius: 6px;
                display: none;
            }
            #result {
                margin-top: 25px;
                padding: 15px;
                border-radius: 6px;
                display: none;
                font-weight: bold;
                font-size: 1.1rem;
            }
            .safe { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e;}
            .nsfw { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444;}
            .loader { display: none; margin-top: 15px; color: var(--text-muted); }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="brand">MainframeForge</div>
        <h1>SFW Classifier</h1>
        
        <form id="uploadForm">
            <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <div id="uploadText">Click to select an image or drag & drop</div>
                <img id="preview" alt="Image Preview">
            </div>
            <input type="file" id="fileInput" accept="image/png, image/jpeg, image/jpg, image/webp">
            <button type="submit" class="btn" id="submitBtn" disabled>Analyze Image</button>
        </form>

        <div class="loader" id="loader">Processing image on the forge...</div>
        <div id="result"></div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const uploadText = document.getElementById('uploadText');
        const submitBtn = document.getElementById('submitBtn');
        const resultDiv = document.getElementById('result');
        const loader = document.getElementById('loader');
        const form = document.getElementById('uploadForm');

        // Handle file selection
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'inline-block';
                    uploadText.style.display = 'none';
                    submitBtn.disabled = false;
                    resultDiv.style.display = 'none'; // hide old results
                }
                reader.readAsDataURL(file);
            }
        });

        // Handle API submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            const file = fileInput.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            submitBtn.disabled = true;
            loader.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                loader.style.display = 'none';
                submitBtn.disabled = false;
                resultDiv.style.display = 'block';

                if (data.status === 'success') {
                    if (data.is_safe) {
                        resultDiv.className = 'safe';
                        resultDiv.innerHTML = `✅ SAFE <br><span style="font-size:0.9rem; font-weight:normal; color:#9ca3af;">Confidence: ${data.confidence.safe}%</span>`;
                    } else {
                        resultDiv.className = 'nsfw';
                        resultDiv.innerHTML = `⚠️ NSFW <br><span style="font-size:0.9rem; font-weight:normal; color:#9ca3af;">Confidence: ${data.confidence.nsfw}%</span>`;
                    }
                } else {
                    resultDiv.className = 'nsfw';
                    resultDiv.textContent = 'Error: ' + data.message;
                }
            } catch (err) {
                loader.style.display = 'none';
                submitBtn.disabled = false;
                resultDiv.style.display = 'block';
                resultDiv.className = 'nsfw';
                resultDiv.textContent = 'Failed to connect to the server.';
            }
        });
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
