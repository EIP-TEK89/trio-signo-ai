from dataclasses import dataclass
import torch
import time
from torch.utils.data import DataLoader
from src.model_class.transformer_sign_recognizer import ModelInfo, SignRecognizerTransformerDataset

from src.datasamples import DataSamplesTensors, TensorPair
from src.train_model.ConfusedSets import ConfusedSets
from src.train_model.parse_args import Args
from src.train_model.TrainStat import TrainStat


@dataclass
class TrainDataLoader:
    train: DataLoader[TensorPair]
    validation: DataLoader[TensorPair] | None = None
    confusion: DataLoader[TensorPair] | None = None
    counter_example: DataLoader[TensorPair] | None = None


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
        train_data, args.confusing_label)
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

    print("Converting trainset to tensor...", end="", flush=True)
    train_tensor: TensorPair
    validation_tensor: TensorPair | None
    train_tensor, validation_tensor = train_data.toTensors(
        args.validation_set_ratio)
    # print(tensors)
    dataloaders: TrainDataLoader = TrainDataLoader(
        train=DataLoader(SignRecognizerTransformerDataset(
            train_tensor[0], train_tensor[1]), batch_size=args.batch_size, shuffle=True)
    )
    if validation_tensor is not None:
        dataloaders.validation = DataLoader(SignRecognizerTransformerDataset(
            validation_tensor[0], validation_tensor[1]), batch_size=args.batch_size, shuffle=True)
    print("[DONE]")

    print("Converting confused labels to tensor...", end="", flush=True)
    if args.embedding_optimization_thresold >= 0:
        confuse_tensor: TensorPair | None = confused_sets.getConfusedSamplesTensor()
        if confuse_tensor is not None:
            dataloaders.confusion = DataLoader(SignRecognizerTransformerDataset(
                confuse_tensor[0], confuse_tensor[1]), batch_size=args.batch_size, shuffle=True)
            print("[DONE]")
        else:
            print("[NO CONFUSED LABELS]")
    else:
        print("[SKIPPED]")

    if args.embedding_optimization_thresold >= 0:
        print("Converting counter examples to tensor...", end="", flush=True)
        counter_tensor: TensorPair | None = confused_sets.getCounterExamplesTensor()
        if counter_tensor is not None:
            dataloaders.counter_example = DataLoader(SignRecognizerTransformerDataset(
                counter_tensor[0], counter_tensor[1]), batch_size=args.batch_size, shuffle=True)
            print("[DONE]")
        else:
            print("[NO COUNTER EXAMPLES]")
    else:
        print("[SKIPPED]")

    model_info: ModelInfo = ModelInfo.build(
        info=train_data.info,
        name=args.name,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim
    )

    return (dataloaders, confused_sets, model_info, train_stats, weigths_balance)
