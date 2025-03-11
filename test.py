import onnx

model_path = "onnx_models/alphabet.onnx"  # Update with the correct path

try:
    # Load the ONNX model
    model = onnx.load(model_path)

    # Validate the model structure
    onnx.checker.check_model(model)

    print("✅ The ONNX model is valid and correctly formatted!")
except Exception as e:
    print(f"❌ Error in ONNX model: {e}")
