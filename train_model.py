import os
import shutil
import time

from src.model_class.transformer_sign_recognizer import SignRecognizerTransformer
from src.train_model.train import train_model
from src.train_model.parse_args import parse_args, Args
from src.train_model.init_train_data import init_train_set
from src.train_model.TrainStat import TrainStat


args: Args = parse_args()

model: SignRecognizerTransformer | None = None

copy_previous_model: bool = False

if args.model_path:
    print("Loading model...", end="", flush=True)
    copy_previous_model = True
    model = SignRecognizerTransformer.loadModelFromDir(
        args.model_path, args.device)
    print("[DONE]")


current_time: str = time.strftime('%d-%m-%Y_%H-%M-%S')
# train_stats: TrainStat
dataloaders, confused_sets, model_info, train_stats, weights = init_train_set(
    args)

if model is None:
    model = SignRecognizerTransformer(model_info, device=args.device)

assert model is not None, "Model is None"

print("Starting training...")
train_stats = train_model(model, dataloaders, confused_sets, train_stats,
                          weights, args.embedding_optimization_thresold,
                          num_epochs=args.epoch,
                          device=args.device)

nb_prev_model: int = 0
if copy_previous_model:
    path: str = args.model_path + "/previous_models/"
    os.makedirs(path, exist_ok=True)

    nb_prev_model = len(os.listdir(path))

    pth_file = model.info.name + ".pth"
    shutil.copy(args.model_path + "/" + pth_file, path)
    os.rename(path + pth_file, f"{path}/{model.info.name}_{nb_prev_model}.pth")

    model.saveModel(args.model_path)
    nb_prev_model += 1
else:
    args.model_path = model.saveModel()
    try:
        shutil.rmtree(args.model_path + "/train_stats")
    except Exception as e:
        print(e)
    try:
        shutil.rmtree(args.model_path + "/previous_models")
    except Exception as e:
        print(e)

train_stats.rename(model.info.name, nb_prev_model)
os.makedirs(args.model_path + "/train_stats/", exist_ok=True)
train_stats.save(f"{args.model_path}/train_stats/{train_stats.name}.json")
