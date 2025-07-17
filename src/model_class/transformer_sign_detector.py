import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import onnxruntime as ort
import numpy as np

from src.datasamples import *
from src.model_class.transformer_sign_recognizer import ModelInfo, SignRecognizerTransformer

class SignDetectorTransformer(SignRecognizerTransformer):
    def __init__(self, model_info: ModelInfo, device: torch.device = torch.device("cpu")):
        """
        Args:
            "hidden" seq_len (int): Number of frame in the past the model will remember.

            "hidden" feature_dim (int): number of value we will give the model to recognize the sign (e.g: 3 for one hand point, 73 for a full hand and 146 for a two hand)

            d_model (int): How the many dimension will the model transform the input of size feature_dim

            num_heads (_type_): Number of of attention head in the model (to determine how many we need we must different quantity until finding a sweetspot, start with 8)

            num_layers (_type_): The longer the signs to recognize the more layer we need (start with 3)

            ff_dim (_type_): Usually d_model x 4, not sure what it does but apparently it help the model makes link between frame. (automatically set to d_model x 4 by default)

            "hidden" num_classes (_type_): Number of sign the model will recognize
        """
        super().__init__(model_info, device)

        self.fc: nn.Linear = nn.Linear(self.info.d_model, 1)  # Couche finale de classification

        self.to(self.device)

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self(x)
            return torch.sigmoid(out, dim=1)

class SignDetectorTransformerDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Paramètres :
        - tensor_list : 3 Dimensional tensor [num_samples, seq_len, num_features]
        - labels : 1 Dimensional tensor with the corresponding label [label_id]
        """
        self.data = data
        self.num_samples = data.shape[0]

        self.labels = labels

        assert len(self.labels) == self.num_samples, "Data and labels must have the same length"

    def __len__(self):
        return self.num_samples  # Nombre total de samples

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne un tuple (X, Y)
        - X : [seq_len, num_features] -> Les features du sample
        - Y : Label associé
        """
        return self.data[idx], self.labels[idx]

class SignDetectorTransformerONNX(nn.Module):
    def __init__(self, model_dir: str):

        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        self.info: ModelInfo = ModelInfo.from_json_file(json_files[0])

        onnx_files = glob.glob(f"{model_dir}/*.onnx")
        if len(onnx_files) == 0:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")


        self.session: ort.InferenceSession = ort.InferenceSession(onnx_files[0])
        self.input_name: str = self.session.get_inputs()[0].name  # First input layer name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_dtype = self.session.get_inputs()[0].type

    def predict(self, data: np.ndarray) -> int:
        out: any = self.session.run(None, {self.input_name: data})
        flat_out = np.nditer(out)

        # best_idx: int = 0
        # best_score: float = flat_out[0]

        # for idx, score in enumerate(flat_out):
        #     if score > best_score:
        #         best_score = score
        #         best_idx = idx

        return flat_out[0]
