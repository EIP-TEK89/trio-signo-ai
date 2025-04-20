import random
from dataclasses import dataclass
from typing import cast, Any

import torch

from src.model_class.transformer_sign_recognizer import SignRecognizerTransformer
from src.datasamples import DataSamplesTensors, DataSamplesInfo, TensorPair


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
    tensors_embeddings: dict[int, torch.Tensor]
    null_embeddings: torch.Tensor | None = None
    null_label_id: int | None = None

    def __init__(self, dst: DataSamplesTensors, confusing_pair: dict[int, int] | None = None):
        self.confusing_pair = confusing_pair if confusing_pair is not None else dict()
        self.tensors = dict()
        self.counter_examples = dict()
        self.null_label_id = dst.info.null_sample_id

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
                # Adding the non counter example to the tensors if not already present
                if i not in self.tensors:
                    self.tensors[i] = dst.getTensorsOfLabel(i)

    def getConfusedSamplesTensor(self) -> TensorPair | None:
        if len(self.tensors) == 0:
            return None
        inputs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for confused in self.confusing_pair.keys():
            tmp_tensor: torch.Tensor = self.tensors[confused]
            inputs.append(tmp_tensor)
            labels.append(torch.full(
                (tmp_tensor.shape[0],), confused))
        if len(labels) == 0:
            return None
        return (torch.cat(inputs), torch.cat(labels))

    def getCounterExamplesTensor(self) -> TensorPair | None:
        """
        Get the counter examples tensor.

        Returns:
            A tuple of two tensors: the counter examples and the label_id they are related to. Or None if there are no counter examples.

            Technically all counter examples are null_label,
            but in order to make the triplet margin loss work,
            we need to know the label_id of the actual valid label related.
            (e.g: for a sample of A sign, the counter examples are null but we will still return the label_id of A)

            tuple[torch.Tensor(counter_example_data), torch.Tensor(non_counter_example_label_id)]
        """
        labels: list[torch.Tensor] = []
        for key, val in self.counter_examples.items():
            labels.append(torch.full((val.shape[0],), key))
        if len(labels) == 0:
            return None
        return (torch.cat(list(self.counter_examples.values())), torch.cat(labels))

    def get_sample(self,
                   embeddings: torch.Tensor,
                   cmp_embeddings: torch.Tensor,
                   closest_embedding: bool = True
                   ) -> torch.Tensor:
        factor: int = 1 if closest_embedding else -1

        distance: float = float("inf") * factor
        best_distance: float = float("inf") * factor
        best_embeddings: torch.Tensor = cmp_embeddings[0]

        for emb in cmp_embeddings:
            # print(emb.shape, embeddings.shape)
            if torch.equal(emb, embeddings):
                continue
            tmp = torch.norm(embeddings - emb)
            if not isinstance(tmp, torch.Tensor):
                print("tmp is not a tensor")
                continue
            distance = float(tmp.item())
            if factor * distance < factor * best_distance:
                best_embeddings = emb
                best_distance = distance

        return best_embeddings

    def getConfusedSamplePosNegPair(self,
                                    model: SignRecognizerTransformer,
                                    anchor_labels: list[int],
                                    anchor_embeddings: torch.Tensor,
                                    anchor_outputs: list[int]
                                    ) -> TensorPair | None:
        """
        Get the positive and negative samples for a given anchor label.

        Returns:
            A tuple of two tensors: the positive embeddings and the negative embeddings.
        """

        positive_samples: list[torch.Tensor] = []
        negative_samples: list[torch.Tensor] = []

        confused_embeddings: dict[int, torch.Tensor] = {}
        for i, anchor_label_id in enumerate(anchor_labels):
            if anchor_outputs[i] != anchor_label_id:
                continue

            # Getting the positive sample
            if anchor_label_id not in confused_embeddings:
                confused_embeddings[anchor_label_id] = model.getEmbeddings(
                    self.tensors[anchor_label_id])

            positive_samples.append(self.get_sample(
                anchor_embeddings, confused_embeddings[anchor_label_id], True))

            # Getting the negative sample
            confused_label_id: int = self.confusing_pair[anchor_label_id]
            if confused_label_id not in confused_embeddings:
                confused_embeddings[confused_label_id] = model.getEmbeddings(
                    self.tensors[confused_label_id])

            negative_samples.append(self.get_sample(
                anchor_embeddings, confused_embeddings[confused_label_id], False))

        if len(positive_samples) == 0 or len(negative_samples) == 0:
            return None
        return (torch.stack(positive_samples), torch.stack(negative_samples))

    def getCounterExamplePosNegPair(self,
                                    model: SignRecognizerTransformer,
                                    non_counter_example_labels: list[int],
                                    anchor_embeddings: torch.Tensor,
                                    anchor_outputs: list[int]
                                    ) -> tuple[TensorPair, list[bool]] | None:
        """
        Get the positive and negative samples for a given anchor label.

        Returns:
            A tuple of two tensors: the positive embeddings and the negative embeddings.
        """
        positive_samples: list[torch.Tensor] = []
        negative_samples: list[torch.Tensor] = []
        mask: list[bool] = []

        confused_embeddings: dict[int, torch.Tensor] = {}
        tensor_embeddings: dict[int, torch.Tensor] = {}

        for i, nce_label_id in enumerate(non_counter_example_labels):
            # Did the model properly classified the sample as null_label
            # Is the label_id available in the counter examples
            if anchor_outputs[i] == self.null_label_id or \
                    not anchor_outputs[i] in self.counter_examples.keys():
                mask.append(False)
                positive_samples.append(anchor_embeddings[0])
                negative_samples.append(anchor_embeddings[0])
                continue

            mask.append(True)
            # Getting the positive sample
            if nce_label_id not in confused_embeddings:
                confused_embeddings[nce_label_id] = model.getEmbeddings(
                    self.counter_examples[nce_label_id])

            positive_samples.append(self.get_sample(
                anchor_embeddings[i], confused_embeddings[nce_label_id], True))

            # Getting the negative sample by providing a real sample of what the model thinks
            negative_label_id: int = anchor_outputs[i]
            if negative_label_id not in tensor_embeddings:
                tensor_embeddings[negative_label_id] = model.getEmbeddings(
                    self.tensors[negative_label_id])

            negative_samples.append(self.get_sample(
                anchor_embeddings[i], tensor_embeddings[negative_label_id], False))

        # print(positive_samples, negative_samples)
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            return None
        return ((torch.stack(positive_samples), torch.stack(negative_samples)), mask)
