import os
from flask import send_from_directory

def get_sign_recognizer_model(model_name: str):
    # model_name = model_name.replace("/", "")
    if not model_name.endswith(".zip"):
        model_name = f"{model_name}.zip"
    return send_from_directory(directory="./models/onnx/", path=model_name, as_attachment=True)
