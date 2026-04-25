from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
import time

app = FastAPI(title="MainframeForge SFW Classifier")

# ==========================================
# TURBO ONNX ENGINE CONFIGURATION
# ==========================================
sess_options = ort.SessionOptions()
# Maximize graph optimizations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Optimized for Render's low-core count CPU
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

# Load the quantized model
session = ort.InferenceSession(
    "onnx_model_quant/model_quantized.onnx", 
    sess_options=sess_options,
    providers=['CPUExecutionProvider']
)

def process_and_predict(image_bytes):
    start_time = time.perf_counter()
    
    # 1. Faster Resize (Bilinear is quicker than Lanczos)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    
    # 2. Optimized Normalization
    img_array = np.array(img).astype(np.float32)
    # Combined normalization step: (x - (mean*255)) / (std*255)
    img_array -= [123.675, 116.28, 103.53] 
    img_array /= [58.395, 57.12, 57.375]    
    
    input_data = np.expand_dims(img_array.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    # 3. Run Inference
    outputs = session.run(None, {"pixel_values": input_data})[0]
    
    # 4. Probabilities
    exp_out = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_out / exp_out.sum()
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return {
        "normal": float(probs[0]),
        "nsfw": float(probs[1]),
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/api/classify")
async def classify_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        results = process_and_predict(image_bytes)
        is_safe = results["normal"] > results["nsfw"]
        
        return {
            "status": "success",
            "is_safe": is_safe,
            "processing_time": f"{results['latency_ms']}ms",
            "confidence": {
                "safe": round(results["normal"] * 100, 1),
                "nsfw": round(results["nsfw"] * 100, 1)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MainframeForge SFW Classifier</title>
        <style>
            :root { --bg: #0f1115; --card: #1a1d24; --primary: #f97316; --text: #f3f4f6; --muted: #9ca3af; }
            body { font-family: sans-serif; background: var(--bg); color: var(--text); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
            .container { background: var(--card); padding: 40px; border-radius: 12px; border: 1px solid #374151; width: 400px; text-align: center; }
            .brand { color: var(--primary); font-weight: bold; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; font-size: 0.9rem; }
            .upload-area { border: 2px dashed #374151; border-radius: 8px; padding: 40px 20px; cursor: pointer; margin: 20px 0; transition: 0.3s; }
            .upload-area:hover { border-color: var(--primary); background: rgba(249, 115, 22, 0.05); }
            .btn { background: var(--primary); color: white; border: none; padding: 12px; border-radius: 6px; width: 100%; font-weight: bold; cursor: pointer; }
            #preview { max-width: 100%; max-height: 200px; display: none; margin-top: 15px; border-radius: 4px; }
            #result { margin-top: 20px; padding: 15px; border-radius: 6px; display: none; font-weight: bold; }
            .safe { border: 1px solid #22c55e; color: #4ade80; background: rgba(34, 197, 94, 0.1); }
            .nsfw { border: 1px solid #ef4444; color: #f87171; background: rgba(239, 68, 68, 0.1); }
            .stats { font-size: 0.8rem; color: var(--muted); margin-top: 8px; font-weight: normal; }
        </style>
    </head>
    <body>
    <div class="container">
        <div class="brand">MainframeForge</div>
        <h2 style="margin:0 0 20px 0;">SFW Classifier</h2>
        <form id="f" onclick="document.getElementById('i').click()" class="upload-area">
            <div id="t">Select Image to Forge</div>
            <img id="preview">
            <input type="file" id="i" hidden accept="image/*">
        </form>
        <button onclick="upload()" class="btn" id="b" disabled>Analyze Engine</button>
        <div id="result"></div>
    </div>
    <script>
        const i = document.getElementById('i'), b = document.getElementById('b'), r = document.getElementById('result'), p = document.getElementById('preview'), t = document.getElementById('t');
        i.onchange = e => { 
            const f = e.target.files[0]; 
            if(f){ p.src = URL.createObjectURL(f); p.style.display='inline-block'; t.style.display='none'; b.disabled=false; r.style.display='none'; }
        };
        async function upload(){
            b.disabled = true; b.innerText = 'Forging...';
            const fd = new FormData(); fd.append('file', i.files[0]);
            try {
                const res = await fetch('/api/classify', { method: 'POST', body: fd });
                const d = await res.json();
                r.style.display = 'block';
                r.className = d.is_safe ? 'safe' : 'nsfw';
                r.innerHTML = `<div>${d.is_safe ? '✅ SAFE' : '⚠️ NSFW'}</div>
                               <div class="stats">Time: ${d.processing_time} | Conf: ${d.is_safe ? d.confidence.safe : d.confidence.nsfw}%</div>`;
            } catch(e) { alert('Error contacting forge.'); }
            b.disabled = false; b.innerText = 'Analyze Engine';
        }
    </script>
    </body>
    </html>
    """
