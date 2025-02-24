import torch
import time
from torch.utils.data import DataLoader
from src.train_model.train import CustomDataset
from src.model_class.transformer_sign_recognizer import ModelInfo, SignRecognizerTransformerDataset

from src.datasample import *
from src.datasamples import *
from src.train_model.ConfusedSets import ConfusedSets
from src.train_model.parse_args import Args
from src.train_model.TrainStat import TrainStat

def init_train_set(args: Args, current_time: str = time.strftime('%d-%m-%Y_%H-%M-%S')) -> tuple[DataLoader, DataLoader | None, DataLoader | None, ModelInfo, torch.Tensor | None, ConfusedSets, TrainStat]:
    """_summary_

    Args:
        trainset_path (str): _description_
        validation_ratio (float, optional): _description_. Defaults to 0.2.
        balance_weights (bool, optional): _description_. Defaults to True.
        model_name (str, optional): _description_. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader | None, ModelInfo, torch.Tensor | None]: TrainData, ValidationData, ModelInfo, WeightsBalance
    """

    print("Loading trainset...", end="", flush=True)
    train_data: DataSamplesTensors = DataSamplesTensors.fromCborFile(args.trainset_path)
    print("[DONE]")
    print("Labels:", train_data.info.label_explicit)
    time.sleep(5)

    sample_quantity: list[int] = []
    for samples in train_data.samples:
        sample_quantity.append(len(samples))
    train_stats: TrainStat = TrainStat(
        name=args.name,
        trainset_name=args.trainset_path,
        labels=train_data.info.labels,
        label_map=train_data.info.label_map,
        sample_quantity=sample_quantity,
        validation_ratio=args.validation_set_ratio
    )


    print("Preparing confused labels...", end="", flush=True)
    confused_sets: ConfusedSets = ConfusedSets(confusing_pair=args.confusing_label, data_samples=train_data)
    print("[DONE]")

    print("Converting trainset to tensor...", end="", flush=True)
    tensors: TrainTensors = train_data.toTensors(args.validation_set_ratio, confused_label=list(confused_sets.confusing_pair.keys()))
    # print(tensors)
    train_dataloader: DataLoader = DataLoader(SignRecognizerTransformerDataset(tensors.train[0], tensors.train[1]), batch_size=args.batch_size, shuffle=True)
    validation_dataloader: DataLoader = None
    if tensors.validation[0] is not None:
        validation_dataloader: DataLoader = DataLoader(SignRecognizerTransformerDataset(tensors.validation[0], tensors.validation[1]), batch_size=args.batch_size, shuffle=True)
    confuse_dataloader: DataLoader = None
    if tensors.confusion[0] is not None:
        confuse_dataloader: DataLoader = DataLoader(SignRecognizerTransformerDataset(tensors.confusion[0], tensors.confusion[1]), batch_size=args.batch_size, shuffle=True)
    print("[DONE]")
    # print(train_dataloader, validation_dataloader, confuse_dataloader)

    model_info: ModelInfo = ModelInfo.build(
            info=train_data.info,
            name=args.name)

    weigths_balance: torch.Tensor = None
    if args.balance_weights:
        weigths_balance = train_data.getClassWeights(class_weights=args.class_weights)

    return (train_dataloader, validation_dataloader, confuse_dataloader, model_info, weigths_balance, confused_sets, train_stats)
