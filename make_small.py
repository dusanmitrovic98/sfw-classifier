from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 1. Load your existing 343MB model
quantizer = ORTQuantizer.from_pretrained("onnx_model")

# 2. Define dynamic quantization (best for Render's CPU)
q_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# 3. Export to a new folder
quantizer.quantize(save_dir="onnx_model_quant", quantization_config=q_config)
print("Success! Small model created in 'onnx_model_quant'.")
