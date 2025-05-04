import cbor2
import io
import torch
import json
from collections import deque
from dataclasses import dataclass
from typing import Self, cast, TypeAlias

from src.gesture import ActiveGestures, ALL_GESTURES
from src.datasample import DataSample2

TensorPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def label_int_size(nb_label: int) -> torch.dtype:
    """
    Get the size of the label int.

    Args:
        nb_label (int): Number of labels.

    Returns:
        torch.dtype: The size of the label int.
    """
    if nb_label < 256:
        return torch.int8
    elif nb_label < 65536:
        return torch.int16
    else:
        return torch.int32


@dataclass
class DataSamplesInfo:
    labels: list[str]
    label_map: dict[str, int]
    memory_frame: int
    active_gestures: ActiveGestures
    one_side: bool
    null_sample_id: int | None

    def __init__(
        self,
        labels: list[str],
        memory_frame: int,
        active_gestures: ActiveGestures = ALL_GESTURES,
        one_side: bool = False,
        null_sample_id: int | None = None,
    ):
        self.labels = labels
        self.memory_frame = memory_frame
        self.active_gestures = active_gestures
        self.label_map = {label: i for i,
                          label in enumerate(self.labels)}
        self.one_side = one_side
        self.null_sample_id = null_sample_id

    @classmethod
    def fromDict(cls, data: dict[str, object]) -> Self:
        assert "labels" in data, "Missing 'labels' field in DataSamplesInfo"
        assert "memory_frame" in data, "Missing 'memory_frame' field in DataSamplesInfo"
        assert isinstance(
            data["labels"], list), "Invalid 'labels' field in DataSamplesInfo"
        assert isinstance(
            data["memory_frame"], int), "Invalid 'memory_frame' field in DataSamplesInfo"
        labels: list[str] = data["labels"]
        tmp: Self = cls(
            labels=labels,
            memory_frame=data["memory_frame"],
        )
        if "active_gestures" in data and isinstance(data["active_gestures"], dict):
            active_gest_dict: dict[str, bool | None] = data["active_gestures"]
            tmp.active_gestures = ActiveGestures(**active_gest_dict)
        if "one_side" in data and isinstance(data["one_side"], bool):
            tmp.one_side = data["one_side"]
        if "null_sample_id" in data and isinstance(data["null_sample_id"], int):
            tmp.null_sample_id = data["null_sample_id"]
        return tmp

    def toDict(self) -> dict[str, object]:
        return {
            "labels": self.labels,
            "memory_frame": self.memory_frame,
            "active_gestures": self.active_gestures.toDict(),
            "label_map": self.label_map,
            "one_side": self.one_side,
            "null_sample_id": self.null_sample_id,
        }


IDX_VALID_SAMPLE: int = 0
IDX_INVALID_SAMPLE: int = 1


class DataSamples:
    info: DataSamplesInfo
    samples: list[tuple[  # each tuple is a label
                  # valid sample, each dict entry is a sample
                  dict[int, list[float]],
                  # invalid sample, each dict entry is a sample
                  dict[int, list[float]]
                  ]]
    sample_count: int

    def __init__(self, info: DataSamplesInfo):
        self.info = info
        self.sample_count = 0

        self.valid_fields: list[str] = info.active_gestures.getActiveFields()
        self.samples = []
        while len(self.samples) < len(info.labels):
            self.samples.append(({}, {}))

    @classmethod
    def fromDict(cls, json_data: dict[str, object]):
        assert "info" in json_data, "Missing 'info' field in DataSamples"
        assert "samples" in json_data, "Missing 'samples' field in DataSamples"
        assert isinstance(
            json_data["info"], dict), "Invalid 'info' field in DataSamples"
        assert isinstance(
            json_data["samples"], list), "Invalid 'samples' field in DataSamples"
        info_dict: dict[str, object] = json_data["info"]
        cls = cls(info=DataSamplesInfo.fromDict(info_dict))
        # (list)labels/(list)un|valid samples/(list)samples/(list[float])data
        dict_sample: list[list[list[list[float]]]] = json_data["samples"]
        # print(len(cls.samples))

        sample_label_id: int = 0
        for labeled_samples in dict_sample:
            for i, sample_kind in enumerate(labeled_samples):
                for sample in sample_kind:
                    # new_datasample: DataSample2 = DataSample2.unflat(cls.info.label_explicit[sample_label_id], sample, cls.valid_fields)
                    cls.samples[sample_label_id][i][id(sample)] = sample
            sample_label_id += 1
        cls.getNumberOfSamples()
        return cls

    @classmethod
    def fromJsonFile(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data: dict[str, object] = json.load(f)
        return cls.fromDict(data)

    @classmethod
    def fromCbor(cls, cbor_data: bytes):
        return cls.fromDict(cast(dict[str, object], cbor2.loads(cbor_data)))

    @classmethod
    def fromCborFile(cls, file_path: str):
        with open(file_path, "rb") as f:
            data: bytes = f.read()
        return cls.fromCbor(data)

    def getNumberOfSamples(self):
        self.sample_count = sum([len(label_samples)
                                for label_samples in self.samples])
        return self.sample_count

    def toDict(self) -> dict[str, object]:
        # self.sample_count = self.getNumberOfSamples()

        # (list)labels/(list)un|valid samples/(list)samples/(list[float])data
        samples: list[list[list[list[float]]]] = []
        for labeled_samples in self.samples:
            tmp: list[list[list[float]]] = [[], []]
            for i, sample_kind in enumerate(labeled_samples):
                for sample in sample_kind.values():
                    # samples[-1].append(sample.flat(self.valid_fields))
                    tmp[i].append(sample)
            samples.append(tmp)

        return {"info": self.info.toDict(), "samples": samples}

    def toJsonFile(self, file_path: str, indent: int | str | None = 0):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.toDict(), f, indent=indent)

    def toCbor(self) -> bytes:
        return cbor2.dumps(self.toDict())

    def toCborFile(self, file_path: str):
        with open(file_path, "wb") as f:
            f.write(self.toCbor())

    def addDataSample(self, data_sample: DataSample2, valid_example: bool = True):
        # Get or cache label_id
        label_id = self.info.label_map.get(data_sample.label)
        if label_id is None:
            raise ValueError(
                f"Label {
                    data_sample.label} is not registered in label_map of this DataSamples class."
            )

        if self.info.one_side:
            data_sample.move_to_one_side()

        idx: int = IDX_VALID_SAMPLE if valid_example else IDX_INVALID_SAMPLE
        flat_list: list[float] = data_sample.flat(self.valid_fields)
        # Use self.sample_count as a unique identifier instead of len(self.samples[label_id])
        self.samples[label_id][idx][id(flat_list)] = flat_list
        # self.samples[label_id].add(sample_data)

        # Increment the overall sample count
        self.sample_count += 1

    def addDataSamples(self, data_samples: list[DataSample2], valid: bool = True):
        for data_sample in data_samples:
            # print(type(data_sample))
            self.addDataSample(data_sample, valid)

    def getNumberOfSamplesOfLabel(self, label_id: int) -> int:
        total_length: int = len(self.samples[label_id][IDX_VALID_SAMPLE])
        if self.info.null_sample_id == label_id:
            for labels in self.samples:
                total_length += len(labels[IDX_INVALID_SAMPLE])
        return total_length


class DataSamplesTensors:
    info: DataSamplesInfo
    samples: list[tuple[torch.Tensor, torch.Tensor | None]]
    sample_count: int
    valid_fields: list[str] = []

    def __init__(self, info: DataSamplesInfo):
        self.info = info
        self.sample_count = 0

        self.valid_fields = info.active_gestures.getActiveFields()
        self.samples = []
        while len(self.samples) < len(info.labels):
            self.samples.append((torch.tensor([0]), torch.tensor([0])))

    @classmethod
    def fromDict(cls, json_data: dict[str, object]):
        assert "info" in json_data, "Missing 'info' field in DataSamples"
        assert "samples" in json_data, "Missing 'samples' field in DataSamples"
        assert isinstance(
            json_data["info"], dict), "Invalid 'info' field in DataSamples"
        assert isinstance(
            json_data["samples"], list), "Invalid 'samples' field in DataSamples"
        cls = cls(info=DataSamplesInfo.fromDict(json_data["info"]))
        # (list)labels/(list)un|valid samples/(list)samples/(list[float])data
        dict_sample: list[list[list[list[float]]]] = json_data["samples"]
        # print(len(cls.samples))

        for label_id, label in enumerate(dict_sample):
            tensor_samples: list[list[torch.Tensor]] = []
            for sample_kind in label:
                tensor_sample_kind: list[torch.Tensor] = []
                for sample in sample_kind:
                    tensor_sample_kind.append(
                        DataSample2.unflat(
                            "", sample, cls.valid_fields
                        ).toTensor(cls.info.memory_frame, cls.valid_fields)
                    )
            del dict_sample[label_id]
            dict_sample.insert(0, [])
            cls.samples[label_id] = (
                torch.stack(tensor_samples[IDX_VALID_SAMPLE]),
                torch.stack(tensor_samples[IDX_INVALID_SAMPLE])
            )
            del tensor_samples
        cls.getNumberOfSamples()
        return cls

    @classmethod
    def fromJsonFile(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.fromDict(data)

    @classmethod
    def fromCbor(cls, cbor_data):
        return cls.fromDict(cbor2.loads(cbor_data))

    @staticmethod
    def readFileHeader(decoder: cbor2.CBORDecoder) -> int:
        # Decode the CBOR map header
        initial_byte: int = decoder.read(1)[0]  # Read the first byte
        # Extract the major type (should be 5 for maps)
        major_type: int = initial_byte >> 5
        assert major_type == 5, "The first item is not a CBOR map!"

        # Extract the size of the map
        map_size: int = initial_byte & 0x1F
        if map_size == 31:  # Indefinite-length maps not supported in this example
            raise ValueError(
                "Indefinite-length maps are not supported in this example!"
            )
        return map_size

    @staticmethod
    def readSampleHeader(decoder: cbor2.CBORDecoder) -> int:
        header: int = decoder.read(1)[0]  # Read the array header
        major_type: int = header >> 5
        additional_info: int = header & 0x1F

        if major_type != 4:
            raise ValueError(
                f"Expected array for 'samples', got type {major_type}."
            )

        # Decode the size of the array based on the additional information
        if additional_info < 24:
            return additional_info  # Directly encoded size
        elif additional_info == 24:
            # Next byte contains the size
            return decoder.read(1)[0]
        elif additional_info == 25:
            return int.from_bytes(decoder.read(2), "big")  # 16-bit size
        elif additional_info == 26:
            return int.from_bytes(decoder.read(4), "big")  # 32-bit size
        elif additional_info == 27:
            return int.from_bytes(decoder.read(8), "big")  # 64-bit size
        raise ValueError("Indefinite-length arrays are not supported.")

    @classmethod
    def fromCborDecoder(cls, fd: io.BufferedReader, decoder: cbor2.CBORDecoder):
        samples: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        info: DataSamplesInfo | None = None
        valid_fields: list[str] | None = None

        map_size: int = cls.readFileHeader(decoder)
        for _ in range(map_size):
            key = decoder.decode()  # Decode the key
            # print(f"Processing field: {key}")

            if key == "info":
                data = decoder.decode()
                assert isinstance(
                    data, dict), "Invalid 'info' field in DataSamples"
                info = DataSamplesInfo.fromDict(data)
                valid_fields = info.active_gestures.getActiveFields()
            elif key == "samples":
                assert info is not None, "Missing 'info' field in DataSamples"
                assert valid_fields is not None, "Missing 'info' field in DataSamples"

                list_size: int = cls.readSampleHeader(decoder)

                print(
                    f"\r\033[KLoading trainset samples: 0/{list_size}",
                    end="",
                    flush=True,
                )

                for i in range(list_size):
                    data = decoder.decode()
                    assert isinstance(
                        data, list), "Invalid 'samples' field in DataSamples"
                    sample_from_label: list[list[list[float]]] = data
                    tensor_samples: list[deque[torch.Tensor]] = []
                    for sample_kind in sample_from_label:
                        tensor_sample_kind: deque[torch.Tensor] = deque()
                        for sample in sample_kind:
                            tensor_sample_kind.append(
                                DataSample2.unflat("", sample, valid_fields).toTensor(
                                    info.memory_frame, valid_fields
                                )
                            )
                        tensor_samples.append(tensor_sample_kind)
                    del sample_from_label
                    counter_example: torch.Tensor | None = None
                    if len(tensor_samples[IDX_INVALID_SAMPLE]) > 0:
                        counter_example = torch.stack(
                            list(tensor_samples[IDX_INVALID_SAMPLE]))
                    samples.append((
                        torch.stack(list(tensor_samples[IDX_VALID_SAMPLE])),
                        counter_example))

                    del tensor_samples
                    print(
                        f"\r\033[KLoading trainset samples: {
                            i + 1}/{list_size} [{info.labels[i]}]",
                        end="",
                        flush=True,
                    )

        assert info is not None, "Missing 'info' field in DataSamples"
        cls = cls(info=info)
        cls.samples = samples
        cls.getNumberOfSamples()
        return cls

    @classmethod
    def dumpInfo(cls, file_path: str) -> tuple[DataSamplesInfo, list[tuple[int, int]]]:
        with open(file_path, "rb") as f:
            decoder: cbor2.CBORDecoder = cbor2.CBORDecoder(f)

            info: DataSamplesInfo | None = None
            sublist_lengths: list[tuple[int, int]] = []

            map_size: int = cls.readFileHeader(decoder)

            for _ in range(map_size):
                key = decoder.decode()  # Decode the key

                if key == "info":
                    data = decoder.decode()
                    assert isinstance(
                        data, dict), "Invalid 'info' field in DataSamples"
                    info = DataSamplesInfo.fromDict(data)
                elif key == "samples":
                    assert info is not None, "Missing 'info' field in DataSamples"

                    list_size: int = cls.readSampleHeader(decoder)

                    for _ in range(list_size):
                        data = decoder.decode()
                        assert isinstance(
                            data, list), "Invalid 'samples' field in DataSamples"
                        sample_from_label: list[list[list[float]]] = data
                        sublist_lengths.append((
                            len(sample_from_label[IDX_VALID_SAMPLE]),
                            len(sample_from_label[IDX_INVALID_SAMPLE])
                        ))
                        del sample_from_label

            assert info is not None, "Missing 'info' field in DataSamples"
            return info, sublist_lengths

    @classmethod
    def fromCborFile(cls, file_path: str):
        with open(file_path, "rb") as f:
            decoder: cbor2.CBORDecoder = cbor2.CBORDecoder(f)
            return cls.fromCborDecoder(f, decoder)

    def getNumberOfSamples(self) -> int:
        self.sample_count = sum([len(label_samples)
                                for label_samples in self.samples])
        return self.sample_count

    def getTensorsOfLabel(self, label_id: int, counter_example: bool = True) -> torch.Tensor:
        """
        Get the tensors of a label, with the option to include counter example.
        Counter example only change the return for the null label if defined.

        Args:
            label_id (int): The label id to get the tensors from.
            counter_example (bool, optional): Include counter example. Defaults to True.

        Returns:
            torch.Tensor: The tensors of the label.
        """
        tensors: list[torch.Tensor] = [
            cast(torch.Tensor, self.samples[label_id][IDX_VALID_SAMPLE])]
        if self.info.null_sample_id == label_id and counter_example:
            for labels in self.samples:
                counter_examples: torch.Tensor | None = labels[IDX_INVALID_SAMPLE]
                if counter_examples is not None:
                    tensors.append(counter_examples)
        return torch.cat(tensors, dim=0)

    def getLabelIdsWithCounterExample(self) -> list[int]:
        """
        Get the label ids with counter example.

        Returns:
            list[int]: The label ids with counter example.
        """
        label_ids: list[int] = []
        for label_id in range(len(self.samples)):
            if self.samples[label_id][IDX_INVALID_SAMPLE] is not None:
                label_ids.append(label_id)
        return label_ids

    def getCounterExampleTensorOfLabel(self, label_id: int) -> torch.Tensor | None:
        return self.samples[label_id][IDX_INVALID_SAMPLE]

    def getCounterExampleTensors(self) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for label_id in range(len(self.samples)):
            tensor: torch.Tensor | None = self.getCounterExampleTensorOfLabel(
                label_id)
            if tensor is not None:
                tensors.append(tensor)
        return tensors

    def getNumberOfSamplesOfLabel(self, label_id: int) -> int:
        total_length: int = cast(
            torch.Tensor, self.samples[label_id][IDX_VALID_SAMPLE]).shape[0]
        if self.info.null_sample_id == label_id:
            for labels in self.samples:
                counter_examples: torch.Tensor | None = labels[IDX_INVALID_SAMPLE]
                if counter_examples is not None:
                    total_length += counter_examples.shape[0]
        return len(self.samples[label_id])

    def toTensors(
        self,
        split_ratio: float = 0,
    ) -> tuple[TensorPair, TensorPair | None]:
        train_in: list[torch.Tensor] = []
        train_out: list[torch.Tensor] = []
        validation_in: list[torch.Tensor] = []
        validation_out: list[torch.Tensor] = []

        dtype: torch.dtype = label_int_size(len(self.samples))
        torch.manual_seed(0)
        for i in range(len(self.samples)):
            sample: torch.Tensor = self.getTensorsOfLabel(i)
            # Random permutation of row indices so sample are evenly reparted between train and validation
            indices = torch.randperm(sample.size(0))
            shuffled_tensor: torch.Tensor = sample[indices]

            sample_count = shuffled_tensor.shape[0]
            split_index = int(sample_count * (1 - split_ratio))

            train_in.append(shuffled_tensor[:split_index])
            train_out.append(torch.full((split_index,), i))

            validation_in.append(shuffled_tensor[split_index:])
            validation_out.append(torch.full(
                (sample_count - split_index,), i))

            del shuffled_tensor

        validation: TensorPair | None = None
        if split_ratio > 0:
            validation = (torch.cat(validation_in, dim=0),
                          torch.cat(validation_out, dim=0))
        return ((torch.cat(train_in, dim=0), torch.cat(train_out, dim=0)), validation)

    def getClassWeights(
        self,
        balance_weight: bool = True,
        class_weights: dict[str, float] = {},
        device: torch.device | None = None,
    ) -> torch.Tensor:
        weigths: list[float] = []
        if balance_weight:
            smallest_class: int = self.getNumberOfSamplesOfLabel(0)
            sample_length: int = len(self.samples)
            for i in range(sample_length):
                sample_size = self.getNumberOfSamplesOfLabel(i)
                if sample_size < smallest_class:
                    smallest_class = sample_size
            for i in range(sample_length):
                weigths.append(smallest_class / self.getNumberOfSamplesOfLabel(i)
                               * class_weights.get(self.info.labels[i], 1))
        else:
            for sample in self.samples:
                weigths.append(1)
        total = sum(weigths)
        weigths = [weight / total for weight in weigths]
        # print(weigths)
        return torch.tensor(weigths, dtype=torch.float32, device=device)
