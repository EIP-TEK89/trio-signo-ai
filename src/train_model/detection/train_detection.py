import time
from typing import Callable

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from src.train_model.detection.AccuracyCalculator import AccuracyCalculator
from src.model_class.transformer_sign_detector import SignDetectorTransformer
from src.train_model.TrainStat import TrainStat, TrainStatEpoch, TrainStatEpochResult
from src.datasamples import TensorPair
from src.train_model.init_train_data import TrainDataLoader


def train_epoch_optimize(optimizer: optim.Optimizer, loss: torch.Tensor):
    """_summary_

    Args:
        optimizer (optim.Optimizer): _description_
        loss (torch.Tensor): _description_
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def binary_train_epoch(model: SignDetectorTransformer,
                       dataloader: DataLoader[TensorPair],
                       criterion: nn.Module,
                       optimizer: optim.Optimizer
                       ) -> tuple[float, AccuracyCalculator]:
    """Will run the model and then optimize it for binary detection.

    Args:
        model (SignDetectorTransformer): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function (should be BCEWithLogitsLoss)
        optimizer (optim.Optimizer): Optimizer

    Returns:
        tuple[float, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.train()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator()

    total_loss: float = 0
    num_batches: int = 0
    inputs: torch.Tensor
    labels: torch.Tensor
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs: torch.Tensor = model(inputs)

        # Convert labels to float for binary classification
        labels = labels.float()

        # Reshape outputs to match labels if needed
        if outputs.dim() > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)  # Remove dimension of size 1

        loss: torch.Tensor = criterion(outputs, labels)
        train_epoch_optimize(optimizer, loss)

        total_loss += loss.item()
        num_batches += 1

        # Convert outputs to probabilities for accuracy calculation
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        accuracy_calculator.calculate_binary_accuracy(predictions, labels)

        print("\rBinary BCE loss batch:", num_batches, "/", len(dataloader), end="")

    print()
    return total_loss / num_batches, accuracy_calculator



def validation_epoch(model: SignDetectorTransformer, dataloader: DataLoader[TensorPair], criterion: nn.Module) -> tuple[float, AccuracyCalculator]:
    """Will run the model without doing the optimization part for binary detection.

    Args:
        model (SignDetectorTransformer): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function (should be BCEWithLogitsLoss)

    Returns:
        tuple[float, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.eval()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator()

    with torch.no_grad():
        total_loss: float = 0
        num_batches: int = 0
        inputs: torch.Tensor
        labels: torch.Tensor
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs: torch.Tensor = model(inputs)

            # Convert labels to float for binary classification
            labels = labels.float()

            # Reshape outputs to match labels if needed
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)

            loss: torch.Tensor = criterion(outputs, labels)

            # Convert outputs to probabilities for accuracy calculation
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            accuracy_calculator.calculate_binary_accuracy(predictions, labels)

            total_loss += loss.item()
            num_batches += 1

    return (total_loss / num_batches, accuracy_calculator)


def log_validation_info(val_acc: AccuracyCalculator, loss: float, val_loss: float):
    loss_diff: float = abs(loss - val_loss)
    mean_loss: float = (loss + val_loss) / 2

    print(f"\tValidation Loss: {val_loss:.4f}, " +
          f"Validation accuracy: {(val_acc.get_accuracy()[0] * 100):.2f}%")
    val_acc.print_accuracy_table()
    print(f"\tLoss Diff: {loss_diff:.4f}, Mean Loss: {mean_loss:.4f}")


def log_train_info(train_acc: AccuracyCalculator,
                   loss: float,
                   learning_rate: float,
                   epoch: int,
                   num_epochs: int,
                   remain_time: str,
                   ) -> None:
    print(f"--- " +
          f"Epoch [{epoch+1}/{num_epochs}], " +
          f"Remaining time: {remain_time}, " +
          f"Learning Rate: {learning_rate}" +
          f" ---")
    print(f"\tTrain Loss: {loss:.4f}, " +
          f"Train Accuracy: {(train_acc.get_accuracy()[0] * 100):.2f}%")
    train_acc.print_accuracy_table()


def get_remain_time(epoch: int,
                    num_epochs: int,
                    train_epoch_durations: list[float],
                    validation_epoch_durations: list[float],
                    validation_interval: int
                    ) -> int:
    remain_epoch: int = num_epochs - (epoch + 1)
    estimated_train_epoch_total_duration: int = 0
    if len(train_epoch_durations) > 0:
        estimated_train_epoch_total_duration = int(
            sum(train_epoch_durations) / len(train_epoch_durations)) * remain_epoch
    estimated_validation_epoch_total_duration: int = 0
    if len(validation_epoch_durations) > 0:
        estimated_validation_epoch_total_duration = int((sum(validation_epoch_durations) / len(
            validation_epoch_durations)) * (remain_epoch / validation_interval))
    return estimated_train_epoch_total_duration + estimated_validation_epoch_total_duration


def run_validation(model: SignDetectorTransformer,
                   validation_data: DataLoader[TensorPair],
                   binary_criterion: nn.Module,
                   total_loss: float,
                   silent: bool = False
                   ) -> tuple[TrainStatEpochResult, float]:
    duration: float = time.time()
    val_loss, val_acc = validation_epoch(
        model, validation_data, binary_criterion)
    duration = time.time() - duration

    validation_epoch_stats = TrainStatEpochResult(
        loss=val_loss,
        accuracy=0,
        duration=duration
    )

    if not silent:
        log_validation_info(val_acc, total_loss, val_loss)

    return validation_epoch_stats, duration


def train_detection_model(model: SignDetectorTransformer,
                dataloaders: TrainDataLoader,
                train_stats: TrainStat,
                weights_balance: torch.Tensor,
                num_epochs: int = 20,
                learning_rate: float = 0.001,
                device: torch.device | None = None,
                validation_interval: int = 2,
                silent: bool = False
                ) -> TrainStat:

    model.to(device)

    # Use BCEWithLogitsLoss for binary classification
    binary_criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    binary_criterion.to(device)
    total_loss: float = 0

    optimizer: optim.Optimizer = optim.Adam(
        model.parameters(), lr=learning_rate)
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5)

    total_time: float = time.time()
    train_epoch_durations: list[float] = []
    validation_epoch_durations: list[float] = []
    validation_epoch_stats: TrainStatEpochResult | None = None

    remain_time: str = "Estimating..."

    for epoch in range(num_epochs):
        cumulated_loss: int = 1
        start_time: float = time.time()
        total_loss = 0

        # Binary training with BCE loss
        bce_loss, train_acc = binary_train_epoch(
            model, dataloaders.train, binary_criterion, optimizer)
        total_loss += bce_loss
        cumulated_loss += 1

        # Average the loss
        total_loss /= cumulated_loss
        scheduler.step(total_loss)

        # Getting some values for stats
        train_epoch_durations.append(time.time() - start_time)
        lr: float = optimizer.param_groups[0]['lr']

        # Print the training information
        if not silent:
            log_train_info(
                train_acc, total_loss, lr, epoch, num_epochs, remain_time)

        # Run model on a validation set if it exists
        validation_epoch_stats = None
        if dataloaders.validation is not None \
                and validation_interval > 1 and \
                epoch % validation_interval == validation_interval - 1:
            validation_epoch_stats, duration = run_validation(
                model, dataloaders.validation, binary_criterion, total_loss, silent)
            validation_epoch_durations.append(duration)

        # Getting some training data for potential future analysis
        train_stats.addEpoch(TrainStatEpoch(
            learning_rate=lr,
            train=TrainStatEpochResult(
                loss=total_loss,
                accuracy=[],
                duration=train_epoch_durations[-1]
            ),
            validation=validation_epoch_stats,
            confusing_pairs={},
            batch_size=0,
            weights_balance=weights_balance.tolist(),
        ))

        # Estimating remaining time
        remain_time = time.strftime(
            '%H:%M:%S', time.gmtime(get_remain_time(
                epoch, num_epochs, train_epoch_durations, validation_epoch_durations, validation_interval)))

    if dataloaders.validation is not None and \
            validation_epoch_stats is None:
        validation_epoch_stats, duration = run_validation(
            model, dataloaders.validation, binary_criterion, total_loss, silent)
        validation_epoch_durations.append(duration)

    train_stats.final_accuracy = validation_epoch_stats

    total_time = time.time() - total_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    train_stats.total_duration = total_time

    return train_stats
