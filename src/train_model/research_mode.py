import optuna

from src.train_model.parse_args import Args
from src.model_class.sign_recognizer_v1 import *
from src.train_model.train import train_model

def research_mode(args: Args, model_info: ModelInfo, train_dataloader: DataLoader, validation_dataloader: DataLoader, weigths_balance: torch.Tensor, ntrial):
    def objective(trial: optuna.trial.Trial) -> float:
        num_layers = trial.suggest_int("num_layers", args.min_layer, args.max_layer)
        layers: list[int] = []
        for i in range(num_layers):
            layers.append(trial.suggest_int(f"hidden_size_layer_{i}", args.min_neuron, args.max_neuron))
        dropout = trial.suggest_float("dropout", args.min_dropout, args.max_dropout)

        model_info.set_intermediate_layers(layers)

        validation_loss = 0
        model = SignRecognizerV1(model_info, device=args.device, dropout=dropout)

        validation_loss = train_model(model, args.device, train_dataloader, validation_dataloader, num_epochs=5, weights_balance=weigths_balance, validation_interval=-1)

        return validation_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=ntrial)

    sorted_trials = sorted(study.trials, key=lambda t: t.value)

    # Display the ranking
    print("Ranked Trials:")
    for rank, trial in enumerate(sorted_trials, 1):
        print(f"Rank {rank}: Value={trial.value:.4f}, Params={trial.params}")
