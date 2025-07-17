from dataclasses import dataclass
from torch.utils.data import DataLoader
from src.datasamples import TensorPair

@dataclass
class TrainDataLoader:
    train: DataLoader[TensorPair]
    validation: DataLoader[TensorPair] | None = None
    confusion: DataLoader[TensorPair] | None = None
    counter_example: DataLoader[TensorPair] | None = None
