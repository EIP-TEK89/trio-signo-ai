import random
from dataclasses import dataclass
from typing import cast

import torch

from src.datasamples import DataSamplesTensors, DataSamplesInfo, label_int_size, TensorPair


def getConfusingPairIds(info: DataSamplesInfo, confusing_pair: dict[str, str]) -> dict[int, int]:
    confusing_pair_ids: dict[int, int] = dict()

    for key, value in confusing_pair.items():
        key = info.label_map[key]
        value = info.label_map[value]
        confusing_pair_ids[key] = value

    return confusing_pair_ids


@dataclass
class ConfusedSets:
    """
    This class stores the tensors for the triplet margin loss.
    """
    confusing_pair: dict[int, int]
    tensors: dict[int, torch.Tensor]
    counter_examples: dict[int, torch.Tensor]

    def __init__(self, dst: DataSamplesTensors, confusing_pair: dict[int, int] | None = None):
        self.confusing_pair = confusing_pair if confusing_pair is not None else dict()
        self.tensors = dict()
        self.counter_examples = dict()

        def add_tensor_to_dict(tensor: torch.Tensor, label: int):
            if label not in self.tensors:
                self.tensors[label] = tensor

        # Set up confusing pair
        for key, val in self.confusing_pair.items():
            if key not in self.tensors:
                self.tensors[key] = dst.getTensorsOfLabel(key)
            if val not in self.tensors:
                self.tensors[val] = dst.getTensorsOfLabel(val)

        # Set up counter examples
        for i in range(len(dst.info.labels)):
            tensor: torch.Tensor | None = dst.getCounterExampleTensorOfLabel(i)
            if tensor is not None:
                self.counter_examples[i] = tensor
                if i not in self.tensors:
                    self.tensors[i] = dst.getTensorsOfLabel(i)

    def getConfusedSamplesTensor(self) -> TensorPair | None:
        if len(self.tensors) == 0:
            return None
        inputs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        dtype: torch.dtype = label_int_size(len(self.tensors))
        for confused in self.confusing_pair.keys():
            tmp_tensor: torch.Tensor = self.tensors[confused]
            inputs.append(tmp_tensor)
            labels.append(torch.full(
                (tmp_tensor.shape[0],), confused))
        if len(labels) == 0:
            return None
        return (torch.cat(inputs), torch.cat(labels))

    def getCounterExamplesTensor(self) -> TensorPair | None:
        labels: list[torch.Tensor] = []
        dtype: torch.dtype = label_int_size(len(self.counter_examples))
        for key, val in self.counter_examples.items():
            labels.append(torch.full((val.shape[0],), key))
        if len(labels) == 0:
            return None
        return (torch.cat(list(self.counter_examples.values())), torch.cat(labels))

    def getConfusedSamplePosNegPair(self, anchor_labels: torch.Tensor) -> TensorPair:
        """
        Get the positive and negative samples for a given anchor label.

        Returns:
            A tuple of two tensors: the positive samples and the negative samples.
        """
        positive_samples: list[torch.Tensor] = []
        negative_samples: list[torch.Tensor] = []
        for i in range(anchor_labels.shape[0]):
            anchor_label: int = cast(int, anchor_labels[i].item())
            positive_tensors: torch.Tensor = self.tensors[anchor_label]
            negative_tensors: torch.Tensor = self.tensors[self.confusing_pair[anchor_label]]
            positive_samples.append(
                positive_tensors[random.randint(0, positive_tensors.shape[0] - 1)])
            negative_samples.append(
                negative_tensors[random.randint(0, negative_tensors.shape[0] - 1)])

        return (torch.stack(positive_samples), torch.stack(negative_samples))

    def getCounterExamplePosNegPair(self, anchor_labels: torch.Tensor) -> TensorPair:
        """
        Get the positive and negative samples for a given anchor label.

        Returns:
            A tuple of two tensors: the positive samples and the negative samples.
        """
        positive_samples: list[torch.Tensor] = []
        negative_samples: list[torch.Tensor] = []
        for i in range(anchor_labels.shape[0]):
            anchor_label: int = cast(int, anchor_labels[i].item())
            positive_tensors: torch.Tensor = self.tensors[anchor_label]
            negative_tensors: torch.Tensor = self.counter_examples[anchor_label]
            positive_samples.append(
                positive_tensors[random.randint(0, positive_tensors.shape[0] - 1)])
            negative_samples.append(
                negative_tensors[random.randint(0, negative_tensors.shape[0] - 1)])

        return (torch.stack(positive_samples), torch.stack(negative_samples))
