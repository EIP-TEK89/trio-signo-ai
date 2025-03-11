import time
import os

import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

from src.datasamples import *

@dataclass
class ModelInfo(DataSamplesInfo):
    name: str
    d_model: int = None
    num_heads: int = None
    num_layers: int = None
    ff_dim: int = None

    @classmethod
    def build(cls, info: DataSamplesInfo, active_gestures: ActiveGestures = None, name: str = None) -> 'ModelInfo':
        if name is None:
            name = f"model_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        name = name.replace("/", "").replace("\\", "")

        if active_gestures is None:
            active_gestures = info.active_gestures
        else:
            available_fields = info.active_gestures.getActiveFields()
            for field in active_gestures.getActiveFields():
                if field not in available_fields:
                    raise ValueError(f"Field {field} not found in available fields {available_fields}")

        return cls(
            labels=info.labels,
            label_map=info.label_map,
            label_explicit=info.label_explicit,
            memory_frame=info.memory_frame,
            active_gestures=active_gestures,
            name=name,
            one_side=info.one_side
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInfo':
        data["active_gestures"] = ActiveGestures(**data["active_gestures"])
        return cls(**data)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'ModelInfo':
        with open(file_path, 'r', encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def set_intermediate_layers(self, layers: list[int]):
        self.layers = [self.layers[0]] + layers + [self.layers[-1]]

    def to_dict(self):
        _dict: dict = super(ModelInfo, self).toDict()
        _dict["name"] = self.name
        _dict["d_model"] = self.d_model
        _dict["num_heads"] = self.num_heads
        _dict["num_layers"] = self.num_layers
        _dict["ff_dim"] = self.ff_dim
        return _dict

    def to_json_file(self, file_path: str, indent: int = 4):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)

class SignRecognizerTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: int = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)  # Ajout & Normalisation

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Ajout & Normalisation
        return x

class SignRecognizerTransformer(nn.Module):
    def __init__(self, model_info: ModelInfo, d_model: int, num_heads: int, num_layers: int, ff_dim: int = None, device: torch.device = torch.device("cpu")):
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
        super().__init__()

        self.device = device
        self.info = model_info
        feature_dim = len(model_info.active_gestures.getActiveFields()) * FIELD_DIMENSION

        if ff_dim is None:
            ff_dim = d_model * 4


        self.embedding: nn.Linear = nn.Linear(feature_dim, d_model)  # On projette feature_dim → d_model
        self.pos_encoding: nn.Parameter = nn.Parameter(torch.randn(1, model_info.memory_frame, d_model))  # Encodage positionnel

        # Empilement des encodeurs
        self.encoder_layers: nn.ModuleList[SignRecognizerTransformerLayer] = nn.ModuleList([
            SignRecognizerTransformerLayer(d_model, num_heads, ff_dim) for _ in range(num_layers)
        ])

        self.fc: nn.Linear = nn.Linear(d_model, len(self.info.labels))  # Couche finale de classification

        self.info.d_model = d_model
        self.info.num_heads = num_heads
        self.info.num_layers = num_layers
        self.info.ff_dim = ff_dim

        self.to(self.device)

    @classmethod
    def loadModelFromDir(cls, model_dir: str, device: torch.device = torch.device("cpu")):
        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        info: ModelInfo = ModelInfo.from_json_file(json_files[0])
        print(info)
        cls = cls(info, info.d_model, info.num_heads, info.num_layers, info.ff_dim, device=device)

        pth_files = glob.glob(f"{model_dir}/*.pth")
        if len(pth_files) == 0:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
        cls.loadPthFile(pth_files[0])

        return cls

    def loadPthFile(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def saveModel(self, model_path: str = None):
        if model_path is None:
            model_path = self.info.name
        print(f"Saving model to {model_path}...", end="", flush=True)

        os.makedirs(model_path, exist_ok=True)
        full_name = f"{model_path}/{self.info.name}"
        torch.save(self.state_dict(), full_name + ".pth")
        self.info.to_json_file(full_name + ".json")
        print("[DONE]")
        return model_path

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        x = self.embedding(x) + self.pos_encoding  # Embedding + Positional Encoding
        if return_embeddings:
            return x
        x = x.permute(1, 0, 2)  # Transformer attend [seq_len, batch_size, d_model]

        for encoder in self.encoder_layers:
            x = encoder(x)

        x = x.mean(dim=0)  # Moyenne sur la séquence
        x = self.fc(x)  # Classification finale
        return x

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self(x)
            return torch.argmax(out, dim=1)

class SignRecognizerTransformerDataset(Dataset):
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

class SignRecognizerTransformerONNX(nn.Module):
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

        best_idx: int = 0
        best_score: float = flat_out[0]

        for idx, score in enumerate(flat_out):
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx
