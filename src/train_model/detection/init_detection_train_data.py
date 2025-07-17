from torch.utils.data import DataLoader
from src.model_class.transformer_sign_detector import SignDetectorTransformerDataset

from src.datasample import *
from src.datasamples import DataSamplesTensors, TensorPair
from src.train_model.parse_args import Args

from src.train_model.train_dataloader import TrainDataLoader

def init_detection_train_data(args: Args, train_data: DataSamplesTensors) -> TrainDataLoader:
    print("Converting trainset to tensor...", end="", flush=True)

    train_tensor: TensorPair
    validation_tensor: TensorPair | None
    train_tensor, validation_tensor = train_data.toTensorsForDetection(
        args.validation_set_ratio)
    # print(tensors)
    dataloaders: TrainDataLoader = TrainDataLoader(
        train=DataLoader(SignDetectorTransformerDataset(
            train_tensor[0], train_tensor[1]), batch_size=args.batch_size, shuffle=True)
    )
    if validation_tensor is not None:
        dataloaders.validation = DataLoader(SignDetectorTransformerDataset(
            validation_tensor[0], validation_tensor[1]), batch_size=args.batch_size, shuffle=True)
    print("[DONE]")

    print("Converting confused labels to tensor... [NOT SUPPORTED FOR DETECTION]")
    print("Converting counter examples to tensor... [NOT SUPPORTED FOR DETECTION]")

    return dataloaders
