import time
from typing import Callable

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from src.train_model.AccuracyCalculator import AccuracyCalculator
from src.model_class.transformer_sign_recognizer import SignRecognizerTransformer
from src.train_model.ConfusedSets import ConfusedSets
from src.train_model.TrainStat import TrainStat, TrainStatEpoch, TrainStatEpochResult
from src.datasamples import TensorPair
from src.train_model.init_train_data import TrainDataLoader
from src.tools import from_1d_tensor_to_list_int


def train_epoch_optimize(optimizer: optim.Optimizer, loss: torch.Tensor):
    """_summary_

    Args:
        optimizer (optim.Optimizer): _description_
        loss (torch.Tensor): _description_
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def cross_entropy_train_epoch(model: SignRecognizerTransformer,
                              dataloader: DataLoader[TensorPair],
                              criterion: nn.Module,
                              optimizer: optim.Optimizer
                              ) -> tuple[float, AccuracyCalculator]:
    """Will run the model and then optimize it.

    Args:
        model (SignRecognizerTransformer): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer

    Returns:
        tuple[torch.Tensor, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.train()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator(
        model.info.labels)

    total_loss: float = 0
    num_batches: int = 0
    inputs: torch.Tensor
    labels: torch.Tensor
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        # print(loss.shape)
        train_epoch_optimize(optimizer, loss)

        total_loss += loss.item()
        num_batches += 1
        accuracy_calculator.calculate_accuracy(outputs, labels)

    return total_loss / num_batches, accuracy_calculator


def triplet_margin_train_epoch(model: SignRecognizerTransformer,
                               dataloader: DataLoader[TensorPair],
                               criterion: nn.TripletMarginLoss,
                               optimizer: optim.Optimizer,
                               pos_neg_func: Callable[[
                                   SignRecognizerTransformer,
                                   list[int],
                                   torch.Tensor,
                                   list[int]],
                                   tuple[TensorPair, list[bool]] | None]
                               ) -> float:
    """Will run the model and then optimize it.

    Args:
        model (SignRecognizerTransformer): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        confused_sets (ConfusedSets): Confused sets

    Returns:
        tuple[torch.Tensor, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.train()

    total_loss: float = 0
    num_batches: int = 0
    inputs: torch.Tensor
    labels: torch.Tensor
    # print(len(dataloader))
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        anchor_emb: torch.Tensor = model.getEmbeddings(inputs)
        anchor_out: list[int] = model.getLabelID(model.classify(anchor_emb))
        correction: tuple[TensorPair, list[bool]] | None = pos_neg_func(
            model,
            from_1d_tensor_to_list_int(labels),
            anchor_emb,
            anchor_out)
        if correction is None:
            continue
        mask: list[bool] = correction[1]
        positive = correction[0][0].to(model.device)
        negative = correction[0][1].to(model.device)
        # print(len(mask), positive.shape, negative.shape)
        loss: torch.Tensor = criterion(
            anchor_emb[mask], positive[mask], negative[mask])
        total_loss += loss.item()
        num_batches += 1
        train_epoch_optimize(optimizer, loss)
        print("\r", num_batches, len(dataloader), end="")
    print()

    return total_loss / num_batches


def validation_epoch(model: SignRecognizerTransformer, dataloader: DataLoader[TensorPair], criterion: nn.Module) -> tuple[float, AccuracyCalculator]:
    """Will run the model without doing the optimization part (Use <strong>train_epoch</strong> function for that).

    Args:
        model (SignRecognizerV1): Model to run
        dataloader (DataLoader): Data to run the model on
        criterion (nn.Module): Loss function

    Returns:
        tuple[torch.Tensor, AccuracyCalculator]: tuple(loss, accuracy_calculator)
    """
    model.eval()
    accuracy_calculator: AccuracyCalculator = AccuracyCalculator(
        model.info.labels)

    with torch.no_grad():
        total_loss: float = 0
        num_batches: int = 0
        inputs: torch.Tensor
        labels: torch.Tensor
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            accuracy_calculator.calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

    return (total_loss / num_batches, accuracy_calculator)


def log_validation_info(val_acc: AccuracyCalculator, loss: float, val_loss: float):
    loss_diff: float = abs(loss - val_loss)
    mean_loss: float = (loss + val_loss) / 2

    validation_avg_acc, _ = val_acc.get_accuracy()
    print(f"\tValidation Loss: {val_loss:.4f}, " +
          f"Validation accuracy: {(validation_avg_acc * 100):.2f}%")
    val_acc.print_accuracy_table()
    print(f"\tLoss Diff: {loss_diff:.4f}, Mean Loss: {mean_loss:.4f}")


def log_train_info(train_acc: AccuracyCalculator,
                   loss: float,
                   learning_rate: float,
                   epoch: int,
                   num_epochs: int,
                   remain_time: str,
                   confused_run: bool = False,
                   counter_example_run: bool = False,
                   ) -> None:
    train_avg_acc, _ = train_acc.get_accuracy()
    print(f"--- " +
          f"Epoch [{epoch+1}/{num_epochs}], " +
          f"Remaining time: {remain_time}, " +
          f"Learning Rate: {learning_rate:.6f}" +
          f" ---")
    print(f"\tTrain Loss: {loss:.4f}, " +
          f"Train Accuracy: {(train_avg_acc * 100):.2f}%")
    print(f"\tConfusion loss run: {confused_run}, " +
          f"Counter example loss run: {counter_example_run}")
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


def run_validation(model: SignRecognizerTransformer,
                   validation_data: DataLoader[TensorPair],
                   cross_entropy_criterion: nn.Module,
                   total_loss: float,
                   silent: bool = False
                   ) -> tuple[TrainStatEpochResult, float]:
    duration: float = time.time()
    val_loss, val_acc = validation_epoch(
        model, validation_data, cross_entropy_criterion)
    duration = time.time() - duration

    validation_epoch_stats = TrainStatEpochResult(
        loss=val_loss,
        accuracy=val_acc.get_correct_over_total(),
        duration=duration
    )

    if not silent:
        log_validation_info(val_acc, total_loss, val_loss)

    return validation_epoch_stats, duration


def train_model(model: SignRecognizerTransformer,
                dataloaders: TrainDataLoader,
                confused_sets: ConfusedSets,
                train_stats: TrainStat,
                weights_balance: torch.Tensor,
                embedding_optimization_threshold: float,
                num_epochs: int = 20,
                learning_rate: float = 0.001,
                device: torch.device | None = None,
                validation_interval: int = 2,
                silent: bool = False
                ) -> TrainStat:

    model.to(device)

    cross_entropy_criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    triplet_margin_criterion: nn.TripletMarginLoss = nn.TripletMarginLoss(
        margin=1.0)
    cross_entropy_criterion.to(device)
    triplet_margin_criterion.to(device)
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
    train_avg_acc: float = 0.0

    confused_run: bool = False
    counter_example_run: bool = False
    #
    # def confused_pos_neg_pair(
    #         model: SignRecognizerTransformer,
    #         anchor_label: list[int],
    #         anchor_embeddings: torch.Tensor,
    #         anchor_outputs: list[int]) -> TensorPair | None:
    #     return confused_sets.getConfusedSamplePosNegPair(model, anchor_label,
    #                                                      anchor_embeddings, anchor_outputs)
    #
    # def counter_example_pair(
    #         model: SignRecognizerTransformer,
    #         non_counter_label: list[int],
    #         anchor_embeddings: torch.Tensor,
    #         anchor_outputs: list[int]) -> tuple[TensorPair, list[bool]] | None:
    #     return confused_sets.getCounterExamplePosNegPair(model, non_counter_label,
                                                         # anchor_embeddings, anchor_outputs)

    for epoch in range(num_epochs):
        cumulated_loss: int = 1
        start_time: float = time.time()
        total_loss = 0

        # Basic training with cross entropy loss
        ce_loss, train_acc = cross_entropy_train_epoch(
            model, dataloaders.train, cross_entropy_criterion, optimizer)
        total_loss += ce_loss
        cumulated_loss += 1
        train_avg_acc = train_acc.get_accuracy()[0]

        # Triplet margin loss to refine the embedding between label that are similar
        # e.g french v sign and u sign
        # confused_run = False
        # if dataloaders.confusion is not None and \
        #         train_avg_acc >= embedding_optimization_threshold:
        #     confused_run = True
        #     tm_loss = triplet_margin_train_epoch(
        #         model, dataloaders.confusion, triplet_margin_criterion, optimizer, confused_pos_neg_pair)
        #     total_loss += tm_loss
        #     cumulated_loss += 1

        # Triplet margin loss to refine the embedding between null label that looks like the label and label
        # e.g a sign and a slightly wrong a sign
        counter_example_run = False
        if dataloaders.counter_example is not None \
                and train_avg_acc >= embedding_optimization_threshold:
            counter_example_run = True
            tm_loss = triplet_margin_train_epoch(
                model, dataloaders.counter_example, triplet_margin_criterion, optimizer, confused_sets.getCounterExamplePosNegPair)
            total_loss += tm_loss
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
                train_acc, total_loss, lr, epoch, num_epochs, remain_time, confused_run, counter_example_run)

        # Run model on a validation set if it exists
        validation_epoch_stats = None
        if dataloaders.validation is not None \
                and validation_interval > 1 and \
                epoch % validation_interval == validation_interval - 1:
            validation_epoch_stats, duration = run_validation(
                model, dataloaders.validation, cross_entropy_criterion, total_loss, silent)
            validation_epoch_durations.append(duration)

        # Getting some training data for potential future analysis
        train_stats.addEpoch(TrainStatEpoch(
            learning_rate=lr,
            train=TrainStatEpochResult(
                loss=total_loss,
                accuracy=train_acc.get_correct_over_total(),
                duration=train_epoch_durations[-1]
            ),
            validation=validation_epoch_stats,
            confusing_pairs=confused_sets.confusing_pair,
            batch_size=0,
            weights_balance=weights_balance.tolist()
        ))

        # Estimating remaining time
        remain_time = time.strftime(
            '%H:%M:%S', time.gmtime(get_remain_time(
                epoch, num_epochs, train_epoch_durations, validation_epoch_durations, validation_interval)))

    if dataloaders.validation is not None and \
            validation_epoch_stats is None:
        validation_epoch_stats, duration = run_validation(
            model, dataloaders.validation, cross_entropy_criterion, total_loss, silent)
        validation_epoch_durations.append(duration)

    train_stats.final_accuracy = validation_epoch_stats

    total_time = time.time() - total_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    train_stats.total_duration = total_time

    return train_stats
