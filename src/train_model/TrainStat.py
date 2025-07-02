import time
from dataclasses import dataclass
import copy
import json

@dataclass
class TrainStatEpochResult:
    loss: float
    accuracy: list[list[int, int]]
    duration: float

@dataclass
class TrainStatEpoch:
    learning_rate: float
    train: TrainStatEpochResult
    validation: TrainStatEpochResult | None
    confusing_pairs: dict[int, int]
    batch_size: int
    weights_balance: list[float] | None

    def toJson(self) -> dict:
        cpy = copy.copy(self)
        cpy.train = self.train.__dict__
        if self.validation is not None:
            cpy.validation = self.validation.__dict__
        return cpy.__dict__

    @classmethod
    def fromJson(cls, data: dict) -> 'TrainStatEpoch':
        data["train"] = TrainStatEpochResult(**data["train"])
        if data["validation"] is not None:
            data["validation"] = TrainStatEpochResult(**data["validation"])
        return cls(**data)

@dataclass
class TrainStat:
    name: str
    trainset_name: str
    labels: list[str]
    label_map: dict[str, int]
    sample_quantity: list[int]
    validation_ratio: float
    epochs: list[TrainStatEpoch]
    final_accuracy: TrainStatEpochResult | None = None
    total_duration: float | None = None

    def __init__(self, name: str,
                 trainset_name: str,
                 labels: list[str],
                 label_map: dict[str, int],
                 sample_quantity: list[int],
                 validation_ratio: float,
                 file_suffix: str = time.strftime('%d-%m-%Y_%H-%M-%S')):
        self.name = f"{name}_{file_suffix}"
        self.trainset_name = trainset_name
        self.labels = labels
        self.label_map = label_map
        self.sample_quantity = sample_quantity
        self.validation_ratio = validation_ratio
        self.epochs = []

    @classmethod
    def fromJson(cls, data: dict) -> 'TrainStat':
        data["epochs"] = [TrainStatEpoch.fromJson(epoch) for epoch in data["epochs"]]
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> 'TrainStat':
        with open(path, 'r', encoding="utf-8") as f:
            return cls.fromJson(json.load(f))

    def rename(self, name: str, file_suffix: str | int = time.strftime('%d-%m-%Y_%H-%M-%S')):
        self.name = f"{name}_{file_suffix}"

    def toJson(self) -> dict:
        cpy = copy.copy(self)
        cpy.epochs = [epoch.toJson() for epoch in self.epochs]
        cpy.final_accuracy = self.final_accuracy.__dict__ if self.final_accuracy is not None else None
        return cpy.__dict__

    def save(self, path: str = None):
        if path is None:
            path = f"{self.name}.json"
        if not path.endswith(".json"):
            path += ".json"
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self.toJson(), f, ensure_ascii=False, indent=4)

    def addEpoch(self, epoch: TrainStatEpoch):
        self.epochs.append(epoch)
