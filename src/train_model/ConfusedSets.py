import random
from dataclasses import dataclass
from typing import cast, Any

import torch

from src.model_class.transformer_sign_recognizer import SignRecognizerTransformer
from src.datasamples import DataSamplesTensors, DataSamplesInfo, TensorPair
from src.gesture import default_device


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
    null_embeddings: torch.Tensor | None = None
    null_label_id: int | None = None

    def __init__(self,
                 dst: DataSamplesTensors,
                 confusing_pair: dict[int, int] | None = None,
                 device: torch.device | None = None):
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
                self.tensors[-i] = tensor
                # Adding the non counter example to the tensors if not already present
                if i not in self.tensors:
                    self.tensors[i] = dst.getTensorsOfLabel(i)

        for key, val in self.tensors.items():
            self.tensors[key] = val.to(default_device(device))

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
            A tuple of two tensors: the valid and their counter examples and
            the label_id they are related to. Or None if there are no counter examples.

            Technically all counter examples are null_label,
            but in order to use the triplet margin loss,
            we instead set them negative valid_label_id
            (e.g: for a sample of A sign, the counter examples are null but we still return -A label_id)

            Return data shape:\n
            Samples shape: [samples, frames, data_point]
            Label_ids shape: [samples, corresponding label_id]\n
            Samples          Label_ids
            [                [
              // Counter examples and then valid samples alternating
              [sample, ...]    [-1, ...]
              [sample, ...]    [1, ...]
              [sample, ...]    [-2, ...]
              [sample, ...]    [2, ...]
            ]                ]

            Returns None if there are no counter examples.
        """
        labels: list[torch.Tensor] = []
        samples: list[torch.Tensor] = []
        for key, val in self.tensors.items():
            if key < 0:
                labels.append(torch.full((val.shape[0],), key))
                samples.append(val)
                labels.append(torch.full((self.tensors[abs(key)].shape[0],), abs(key)))
                samples.append(self.tensors[abs(key)])

        if len(labels) == 0:
            return None
        # Shuffle samples and labels together
        # combined = list(zip(samples, labels))
        # random.shuffle(combined)
        # samples, labels = zip(*combined)
        #second shuffle
        # indices = torch.randperm(samples.size(0))
        # samples = samples[indices]
        # labels = labels[indices]
        return (torch.cat(samples), torch.cat(labels))

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

    def getRandomSample(self, label_id: int, counter_example: bool = False) -> torch.Tensor | None:
        """
        Get a random sample from the tensors.

        Args:
            label_id: The label id of the sample to get.
            counter_example: If True, get a counter example.

        Returns:
            A random sample from the tensors.
        """
        tensor: torch.Tensor

        if counter_example:
            if label_id not in self.counter_examples:
                return None
            tensor = self.counter_examples[label_id]
        else:
            if label_id not in self.tensors:
                return None
            tensor = self.tensors[label_id]
        return random.choice(tensor)

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
                                    anchor_labels: list[int],
                                    anchor_embeddings: torch.Tensor,
                                    anchor_outputs: list[int]
                                    ) -> tuple[TensorPair, list[bool]] | None:
        """
        Get the positive and negative samples for a given anchor label.

        Returns:
            A tuple of two tensors: the positive embeddings and the negative embeddings.
        """
        assert self.null_label_id is not None, "Null label id is not set"

        positive_samples: list[torch.Tensor] = []
        negative_samples: list[torch.Tensor] = []
        mask: list[bool] = []

        # print(anchor_embeddings.shape)
        sample_embeddings: dict[int, torch.Tensor] = {}
        for i, anchor_label_id in enumerate(anchor_labels):
            label_id_to_compare: int = anchor_label_id
            # If the label_id is negative (which happens if the sample is a counter example)
            # We update the label_id to null_sample_id as counter examples are null samples.
            if label_id_to_compare < 0:
                label_id_to_compare = self.null_label_id
            # If the model properly classified the sample we skip it
            # if anchor_outputs[i] == label_id_to_compare:
            #     mask.append(False)
            #     positive_samples.append(anchor_embeddings[0])
            #     negative_samples.append(anchor_embeddings[0])
            #     continue
            mask.append(True)

            # if abs(anchor_label_id) not in sample_embeddings:
            #     sample_embeddings[anchor_label_id] = model.getEmbeddings(
            #         self.tensors[anchor_label_id])
            # if -abs(anchor_label_id) not in sample_embeddings:
            #     sample_embeddings[-abs(anchor_label_id)] = model.getEmbeddings(
            #         self.tensors[-abs(anchor_label_id)])

            # valid_sample: torch.Tensor | None = random.choice(
            #     sample_embeddings[anchor_label_id])
            # counter_sample: torch.Tensor | None = random.choice(
            #     sample_embeddings[-abs(anchor_label_id)])

            print(self.tensors[anchor_label_id].shape, self.tensors[-abs(anchor_label_id)].shape)
            valid_samples: torch.Tensor = model.getEmbeddings(self.tensors[anchor_label_id])
            invalid_samples: torch.Tensor = model.getEmbeddings(self.tensors[-abs(anchor_label_id)])


            valid_sample: torch.Tensor = random.choice(valid_samples)
            counter_sample: torch.Tensor = random.choice(invalid_samples)
            del valid_samples
            del invalid_samples

            assert valid_sample is not None, "Valid sample is None"
            assert counter_sample is not None, "Counter sample is None"
            valid_sample = valid_sample.to(model.device)
            counter_sample = counter_sample.to(model.device)
            print(valid_sample.shape, counter_sample.shape)

            if anchor_label_id < 0:
                positive_samples.append(counter_sample)
                negative_samples.append(valid_sample)
            else:
                positive_samples.append(valid_sample)
                negative_samples.append(counter_sample)

        if len(positive_samples) == 0 or len(negative_samples) == 0:
            return None
        # print(positive_samples, len(positive_samples))
        # for i in range(len(positive_samples)):
        #     print(positive_samples[i].shape, mask[i])
        return ((torch.stack(positive_samples), torch.stack(negative_samples)), mask)
        # positive_samples: list[torch.Tensor] = []
        # negative_samples: list[torch.Tensor] = []
        # mask: list[bool] = []
        #
        # confused_embeddings: dict[int, torch.Tensor] = {}
        # tensor_embeddings: dict[int, torch.Tensor] = {}
        #
        # for i, nce_label_id in enumerate(non_counter_example_labels):
        #     # Did the model properly classified the sample as null_label
        #     # Is the label_id available in the counter examples
        #     if anchor_outputs[i] == self.null_label_id or \
        #             not anchor_outputs[i] in self.counter_examples.keys():
        #         mask.append(False)
        #         positive_samples.append(anchor_embeddings[0])
        #         negative_samples.append(anchor_embeddings[0])
        #         continue
        #
        #     mask.append(True)
        #     # Getting the positive sample
        #     if nce_label_id not in confused_embeddings:
        #         confused_embeddings[nce_label_id] = model.getEmbeddings(
        #             self.counter_examples[nce_label_id])
        #
        #     positive_samples.append(self.get_sample(
        #         anchor_embeddings[i], confused_embeddings[nce_label_id], True))
        #
        #     # Getting the negative sample by providing a real sample of what the model thinks
        #     negative_label_id: int = anchor_outputs[i]
        #     if negative_label_id not in tensor_embeddings:
        #         tensor_embeddings[negative_label_id] = model.getEmbeddings(
        #             self.tensors[negative_label_id])
        #
        #     negative_samples.append(self.get_sample(
        #         anchor_embeddings[i], tensor_embeddings[negative_label_id], False))
        #
        # # print(positive_samples, negative_samples)
        # if len(positive_samples) == 0 or len(negative_samples) == 0:
        #     return None
        # return ((torch.stack(positive_samples), torch.stack(negative_samples)), mask)
