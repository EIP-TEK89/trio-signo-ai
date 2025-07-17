from dataclasses import dataclass
import torch
import time
from torch.utils.data import DataLoader
from src.model_class.transformer_sign_recognizer import ModelInfo, SignRecognizerTransformerDataset

from src.datasamples import DataSamplesTensors, TensorPair
from src.train_model.ConfusedSets import ConfusedSets
from src.train_model.parse_args import Args
from src.train_model.TrainStat import TrainStat

from src.train_model.recognition.init_recognition_train_data import init_recognition_train_data
from src.train_model.detection.init_detection_train_data import init_detection_train_data

from src.train_model.train_dataloader import TrainDataLoader


def init_train_set(args: Args,
                   ) -> tuple[TrainDataLoader, ConfusedSets, ModelInfo, TrainStat, torch.Tensor]:
    """Load the training data and format it to make it easy to use for training

    Args:
        args (Args): _description_

    Returns:
        tuple[TrainDataLoader, ConfusedSets, ModelInfo, TrainStat, torch.Tensor | None]:
            TrainDataLoader: The dataloader for the training data
            ConfusedSets: The confused sets
            ModelInfo: The model info
            TrainStat: The train stat
            torch.Tensor: The weights balance
    """

    print("Loading trainset...", end="", flush=True)
    train_data: DataSamplesTensors = DataSamplesTensors.fromCborFile(
        args.trainset_path)
    print("[DONE]")
    print("Labels:", train_data.info.labels)

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
    confused_sets: ConfusedSets = ConfusedSets(
        train_data, args.confusing_label, args.device)
    print("[DONE]")

    print("Balancing class weight...", end="", flush=True)
    weigths_balance: torch.Tensor
    if args.balance_weights:
        weigths_balance = train_data.getClassWeights(
            class_weights=args.class_weights)
        print("[DONE]")
    else:
        weigths_balance = torch.ones(len(train_data.info.labels))
        print("[SKIPPED]")

    dataloaders: TrainDataLoader

    if not args.sign_detector:
        dataloaders = init_recognition_train_data(
            args, train_data, confused_sets)
    else:
        dataloaders = init_detection_train_data(args, train_data)


    model_info: ModelInfo = ModelInfo.build(
        info=train_data.info,
        name=args.name,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim
    )

    return (dataloaders, confused_sets, model_info, train_stats, weigths_balance)
