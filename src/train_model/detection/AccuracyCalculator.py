import torch
from typing import Self
from src.misc.color_terminal import print_color, print_color_reset
from src.train_model.pick_color import pick_color

class AccuracyCalculator:
    correct_results: list[int]
    results: list[int]

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_results = []
        self.results = []

    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor):
        # Get predictions
        # print(outputs)
        predictions = torch.sigmoid(outputs)

        self.results.extend(predictions.tolist())
        self.correct_results.extend(labels.tolist())

    def calculate_binary_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Calculate accuracy for binary classification with discrete predictions (0 or 1).

        Args:
            predictions (torch.Tensor): Binary predictions (0 or 1)
            labels (torch.Tensor): True binary labels (0 or 1)
        """
        # Convert predictions to probabilities for consistency with existing methods
        # Since predictions are already binary (0 or 1), we use them directly as probabilities
        self.results.extend(predictions.tolist())
        self.correct_results.extend(labels.tolist())

    def get_accuracy(self, separation: float = 0.5) -> tuple[float, float, tuple[tuple[int, int, float], tuple[int, int, float]]]:
        """_summary_

        Args:
            separation (float, optional): Where to distinguish between valid and invalid where 0 is invalid and 1 valid. Defaults to 0.5.

        Returns:
            tuple[float, tuple[tuple[int, int, float], tuple[int, int, float]]]:

            *All value are based on the separation value*
            float: Average accuracy [0, 1]
            float: Average accuracy normalized [0, 1]
            tuple[int, int, float]:
                - (correct_correct, correct_total, correct_accuracy [0, 1])
            tuple[int, int, float]:
                - (correct_incorect, incorrect_total, incorrect_accuracy [0, 1])
        """

        correct_correct: int = 0
        correct_incorect: int = 0
        correct_total: int = 0
        incorrect_total: int = 0

        for i in range(len(self.correct_results)):
            correct_total += int(self.correct_results[i])
            incorrect_total += int(not self.correct_results[i])

            if self.correct_results[i] and self.results[i] >= separation:
                correct_correct += 1
            elif not self.correct_results[i] and self.results[i] < separation:
                correct_incorect += 1

        correct_acc: float = correct_correct / correct_total if correct_total > 0 else 0
        incorrect_acc: float = correct_incorect / incorrect_total if incorrect_total > 0 else 0
        total_values: int = correct_total + incorrect_total
        return (
            (correct_correct + correct_incorect) / total_values if total_values > 0 else 0,
            (correct_acc + incorrect_acc) / 2,
            (correct_correct, correct_total, correct_acc),
            (correct_incorect, incorrect_total, incorrect_acc)
        )

    def print_accuracy_table(self, pre_str: str = "\t",
                             color_chart: list[tuple[float, tuple[int, int, int]]] = [
                              (0, (192, 0, 0)),
                              (0.5, (255, 0, 0)),
                              (0.7, (255, 50, 0)),
                              (0.8, (255, 150, 0)),
                              (0.9, (255, 255, 0)),
                              (0.95, (32, 255, 0)),
                              (0.99, (0, 200, 128)),
                              (1, (0, 0, 255))]):


        for separation in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            accuracies: tuple[float, tuple[int, int, float], tuple[int, int, float]] = self.get_accuracy(separation)

            # print(accuracies)
            print_color(pick_color(accuracies[0], color_chart))
            print(f"{pre_str}{separation}: {(accuracies[0] * 100):.2f}%", end="")
            print_color(pick_color(accuracies[1], color_chart))
            print(f" ({(accuracies[1] * 100):.2f}%)", end="")
            print_color(pick_color(accuracies[2][2], color_chart))
            print(f" V: {(accuracies[2][2] * 100):.2f}% {accuracies[2][0]}/{accuracies[2][1]}", end="")
            print_color(pick_color(accuracies[3][2], color_chart))
            print(f" X: {(accuracies[3][2] * 100):.2f}% {accuracies[3][0]}/{accuracies[3][1]}", end="")
            print_color_reset()

    def add(self, other: Self):
        """Add another AccuracyCalculator to this one."""
        self.correct_results.extend(other.correct_results)
        self.results.extend(other.results)
