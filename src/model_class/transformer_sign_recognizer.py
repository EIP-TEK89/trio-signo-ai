import time
import os
import json
from dataclasses import dataclass, field
from typing import Self, cast, override
from numbers import Number

import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import onnxruntime as ort
import numpy as np
from numpy.typing import NDArray

from src.gesture import FIELD_DIMENSION, default_device
from src.datasamples import DataSamplesInfo, TensorPair

@dataclass
class TrackingModel:
    hand: int = 2
    body: int = 1
    face: int = 1

@dataclass
class ModelInfo(DataSamplesInfo):
    name: str
    d_model: int
    num_heads: int
    num_layers: int
    ff_dim: int
    tracking_model: TrackingModel = field(default_factory=TrackingModel)

    @classmethod
    def build(cls,
              info: DataSamplesInfo,
              name: str | None = None,
              d_model: int = 32,
              num_heads: int = 8,
              num_layers: int = 3,
              ff_dim: int | None = None
              ) -> Self:
        if name is None:
            name = f"model_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        name = name.replace("/", "").replace("\\", "")

        return cls(
            labels=info.labels,
            label_map=info.label_map,
            memory_frame=info.memory_frame,
            active_gestures=info.active_gestures,
            one_side=info.one_side,
            null_sample_id=info.null_sample_id,
            name=name,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim if ff_dim is not None else d_model * 4
        )

    @override
    @classmethod
    def fromDict(cls,
                 data: dict[str, object]
                 ) -> Self:
        info: DataSamplesInfo = DataSamplesInfo.fromDict(data)
        assert "name" in data, "name must be in the dict"
        assert isinstance(data["name"], str), "name must be a str or None"

        assert "d_model" in data, "d_model must be in the dict"
        assert isinstance(data["d_model"], int), "d_model must be an int"

        assert "num_heads" in data, "num_heads must be in the dict"
        assert isinstance(data["num_heads"], int), "num_heads must be an int"

        assert "num_layers" in data, "num_layers must be in the dict"
        assert isinstance(data["num_layers"], int), "num_layers must be an int"

        assert "ff_dim" in data, "ff_dim must be in the dict"
        assert isinstance(data["ff_dim"], int), "ff_dim must be an int"

        tracking_model: TrackingModel = TrackingModel()
        if "tracking_model" in data:
            assert isinstance(data["tracking_model"], dict), "tracking_model must be a dict"
            assert "hand" in data["tracking_model"], "hand must be in the tracking_model dict"
            assert isinstance(data["tracking_model"]["hand"], Number), "hand must be a Number"
            assert "body" in data["tracking_model"], "body must be in the tracking_model dict"
            assert isinstance(data["tracking_model"]["body"], Number), "body must be a Number"
            assert "face" in data["tracking_model"], "face must be in the tracking_model dict"
            assert isinstance(data["tracking_model"]["face"], Number), "face must be a Number"
            tracking_model: TrackingModel = TrackingModel(
                hand=int(data["tracking_model"]["hand"]),
                body=int(data["tracking_model"]["body"]),
                face=int(data["tracking_model"]["face"])
            )

        cls = cls.build(info,
                         data["name"],
                         data["d_model"],
                         data["num_heads"],
                         data["num_layers"],
                         data["ff_dim"])
        cls.tracking_model = tracking_model
        return cls

    @classmethod
    def fromJsonFile(cls,
                     file_path: str
                     ) -> Self:
        with open(file_path, 'r', encoding="utf-8") as f:
            return cls.fromDict(cast(dict[str, object], json.load(f)))

    @override
    def toDict(self
               ) -> dict[str, object]:
        _dict: dict[str, object] = super(ModelInfo, self).toDict()
        _dict["name"] = self.name
        _dict["d_model"] = self.d_model
        _dict["num_heads"] = self.num_heads
        _dict["num_layers"] = self.num_layers
        _dict["ff_dim"] = self.ff_dim
        _dict["tracking_model"] = {
            "hand": self.tracking_model.hand,
            "body": self.tracking_model.body,
            "face": self.tracking_model.face
        }
        return _dict

    def toJsonFile(self,
                   file_path: str,
                   indent: int = 4
                   ) -> None:
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(self.toDict(), f, ensure_ascii=False, indent=indent)


class SignRecognizerTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention
        attn_output: torch.Tensor
        _: torch.Tensor
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)  # Ajout & Normalisation

        # Feed-Forward Network
        ffn_output: torch.Tensor = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Ajout & Normalisation
        return x


class SignRecognizerTransformer(nn.Module):
    def __init__(self, info: ModelInfo, device: torch.device | None = None):
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

        self.device: torch.device = default_device(device)
        self.info: ModelInfo = info
        feature_dim = len(
            self.info.active_gestures.getActiveFields()) * FIELD_DIMENSION

        # On projette feature_dim → d_model
        self.embedding: nn.Linear = nn.Linear(feature_dim, self.info.d_model)
        self.pos_encoding: torch.Tensor = nn.Parameter(torch.randn(
            1, self.info.memory_frame, self.info.d_model))  # Encodage positionnel

        # Empilement des encodeurs
        self.encoder_layers: nn.ModuleList = nn.ModuleList([
            SignRecognizerTransformerLayer(self.info.d_model, self.info.num_heads, self.info.ff_dim) for _ in range(self.info.num_layers)
        ])

        # Couche finale de classification
        self.fc: nn.Linear = nn.Linear(
            self.info.d_model, len(self.info.labels))

        self.to(self.device)

    @classmethod
    def loadModelFromDir(cls, model_dir: str, device: torch.device | None = None) -> Self:
        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        info: ModelInfo = ModelInfo.fromJsonFile(json_files[0])
        print(info)
        cls = cls(info, device=device)

        pth_files = glob.glob(f"{model_dir}/*.pth")
        if len(pth_files) == 0:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
        cls.loadPthFile(pth_files[0])

        return cls

    def loadPthFile(self, model_path: str) -> Self:
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        return self

    def saveModel(self, model_path: str | None = None):
        if model_path is None:
            model_path = self.info.name
        print(f"Saving model to {model_path}...", end="", flush=True)

        os.makedirs(model_path, exist_ok=True)
        full_name: str = f"{model_path}/{self.info.name}"
        torch.save(self.state_dict(), full_name + ".pth")
        self.info.toJsonFile(full_name + ".json")
        print("[DONE]")
        return model_path

    def getEmbeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the embeddings of the input tensor

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            torch.Tensor: Embeddings of shape [batch_size, d_model]
        """
        # Embedding + Positional Encoding
        x = self.embedding(x) + self.pos_encoding
        # Transformer attend [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)

        for encoder in self.encoder_layers:
            x = encoder(x)

        return x.mean(dim=0)

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify the input embeddings using the final linear layer.
        Args:
            embeddings (torch.Tensor): Input embeddings of shape [batch_size, d_model]

        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        return cast(torch.Tensor, self.fc(embeddings))

    def getLabelID(self, logits: torch.Tensor) -> list[int]:
        """
        Get the label id from the logits.
        Args:
            logits (torch.Tensor): Input logits of shape [batch_size, num_classes]

        Returns:
            list[int]: List of predicted label ids
        """
        return cast(list[int], torch.argmax(logits, dim=1).tolist())

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.getEmbeddings(x))

    def predict(self, x: torch.Tensor) -> int:
        """
        Predict the label id for the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]
        Returns:
            int: Predicted label id
        """
        with torch.no_grad():
            print(self.getLabelID(self.forward(x)))
            return self.getLabelID(self.forward(x))[0]


class SignRecognizerTransformerDataset(Dataset[TensorPair]):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Paramètres :
        - tensor_list : 3 Dimensional tensor [num_samples, seq_len, num_features]
        - labels : 1 Dimensional tensor with the corresponding label [label_id]
        """
        self.data: torch.Tensor = data
        self.num_samples: int = data.shape[0]

        self.labels: torch.Tensor = labels

        assert len(
            self.labels) == self.num_samples, "Data and labels must have the same length"

    def __len__(self):
        return self.num_samples  # Nombre total de samples

    @override
    def __getitem__(self, idx: int) -> TensorPair:
        """
        Retourne un tuple (X, Y)
        - X : [seq_len, num_features] -> Les features du sample
        - Y : Label associé
        """
        return self.data[idx], self.labels[idx]


class SignRecognizerTransformerONNX(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()

        json_files = glob.glob(f"{model_dir}/*.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"No .json file found in {model_dir}")
        self.info: ModelInfo = ModelInfo.fromJsonFile(json_files[0])

        onnx_files = glob.glob(f"{model_dir}/*.onnx")
        if len(onnx_files) == 0:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")

        self.session: ort.InferenceSession = ort.InferenceSession(
            onnx_files[0])
        self.input_name: str = cast(str, self.session.get_inputs()[0].name)

    def predict(self, data: NDArray[np.float32]) -> int:
        session_data: dict[str, NDArray[np.float32]] = {
            self.input_name: data}
        out: list[NDArray[np.float32]] = cast(
            list[NDArray[np.float32]], self.session.run(None, session_data))
        flat_out: np.nditer = np.nditer(out)

        best_idx: int = 0
        best_score: NDArray[np.float32] = flat_out[0]

        for i in range(len(flat_out)):
            if flat_out[0] > best_score:
                best_score = flat_out[0]
                best_idx = i
            i += 1

        return best_idx
