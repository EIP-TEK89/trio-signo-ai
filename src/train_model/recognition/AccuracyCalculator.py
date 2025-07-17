import torch
from typing import Self
from src.misc.color_terminal import print_color, print_color_reset
from src.train_model.pick_color import pick_color

class AccuracyCalculator:
    def __init__(self, labels: list[str]):
        self.num_classes: int = len(labels)
        self.correct_per_class: list[int] = None
        self.total_per_class: list[int] = None
        self.labels: list[str] = labels
        self.reset()

    def reset(self):
        self.correct_per_class = [0] * self.num_classes # To track correct predictions per class
        self.total_per_class = [0] * self.num_classes # To track total samples per class

    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor):
        # Get predictions
        # print(outputs)
        _, predictions = torch.max(outputs, 1)  # Predicted class indices
        # print(predictions)

        # Update correct and total counts for each class
        for label in range(self.num_classes):
            self.correct_per_class[label] += ((predictions == label) & (labels == label)).sum().item()
            self.total_per_class[label] += (labels == label).sum().item()

    def get_accuracy(self) -> tuple[float, list[float]]:
        avg_accuracy = sum(self.correct_per_class) / sum(self.total_per_class) if sum(self.total_per_class) > 0 else 0
        return (avg_accuracy, [correct / total if total > 0 else 0 for correct, total in zip(self.correct_per_class, self.total_per_class)])

    def get_correct_over_total(self) -> list[list[int, int]]:
        return [[correct, total] for correct, total in zip(self.correct_per_class, self.total_per_class)]

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

        _, accuracies = self.get_accuracy()

        for i, label in enumerate(self.labels):
            print_color(pick_color(accuracies[i], color_chart))
            print(f"{pre_str}{label}: {(accuracies[i] * 100):.2f}% {self.correct_per_class[i]}/{self.total_per_class[i]}")
            print_color_reset()

    def add(self, other: Self):
        for i in range(self.num_classes):
            self.correct_per_class[i] += other.correct_per_class[i]
            self.total_per_class[i] += other.total_per_class[i]
