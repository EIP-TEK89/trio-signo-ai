from torch.utils.data import DataLoader
from src.model_class.transformer_sign_recognizer import  SignRecognizerTransformerDataset

from src.datasample import *
from src.datasamples import *
from src.train_model.ConfusedSets import ConfusedSets
from src.train_model.parse_args import Args
from src.train_model.train_dataloader import TrainDataLoader

def init_recognition_train_data(args: Args, train_data: DataSamplesTensors, confused_sets: ConfusedSets) -> TrainDataLoader:
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
    if args.embedding_optimization_threshold >= 0:
        confuse_tensor: TensorPair | None = confused_sets.getConfusedSamplesTensor()
        if confuse_tensor is not None:
            dataloaders.confusion = DataLoader(SignRecognizerTransformerDataset(
                confuse_tensor[0], confuse_tensor[1]), batch_size=args.batch_size, shuffle=True)
            print("[DONE]")
        else:
            print("[NO CONFUSED LABELS]")
    else:
        print("[SKIPPED]")

    print("Converting counter examples to tensor...", end="", flush=True)
    if args.embedding_optimization_threshold >= 0:
        counter_tensor: TensorPair | None = confused_sets.getCounterExamplesTensor()
        if counter_tensor is not None:
            dataloaders.counter_example = DataLoader(SignRecognizerTransformerDataset(
                counter_tensor[0], counter_tensor[1]), batch_size=args.batch_size, shuffle=True)
            print("[DONE]")
        else:
            print("[NO COUNTER EXAMPLES]")
    else:
        print("[SKIPPED]")
    return dataloaders
