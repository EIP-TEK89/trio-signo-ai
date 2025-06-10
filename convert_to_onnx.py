import sys
import torch

from src.model_class.transformer_sign_recognizer import *
from src.datasample import DataSample

# Charger un modèle pré-entraîné (exemple: ResNet18)
model = SignRecognizerTransformer.loadModelFromDir(sys.argv[1])
model.eval()

datasample: DataSample = DataSample("", [], model.info.memory_frame)
valid_fields: list[str] = model.info.active_gestures.getActiveFields()
dummy_input: torch.Tensor = datasample.toTensor(
    model.info.memory_frame, valid_fields)

# Exporter vers ONNX
torch.onnx.export(model, dummy_input, f"{model.info.name}.onnx",
                  input_names=["input"], output_names=["output"])
