import copy
import math
import json
from dataclasses import dataclass, fields
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Self, cast
from src.gesture import DataGestures, FIELD_DIMENSION, FIELDS, default_device, FaceLandmarkResult


def clamp(value: float, min_value: float, max_value: float):
    return max(min_value, min(value, max_value))


@dataclass
class DataSample2:
    label: str
    gestures: list[DataGestures]
    framerate: int = 30
    # This attribute tells the trainset generator if the sample can be mirrored, put it to false if the gesture is not symmetrical (e.g z french sign)
    mirrorable: bool = True
    invalid: bool = False

    @classmethod
    def fromDict(cls, json_data: dict[str, object]) -> Self:
        assert "gestures" in json_data, "Invalid JSON data, missing 'gestures' key"
        assert isinstance(
            json_data["gestures"], list), "Invalid JSON data, 'gestures' must be a list"
        dict_gestures: list[dict[str, list[float] | None]
                            ] = json_data["gestures"]
        gestures: list[DataGestures] = []
        for gesture in dict_gestures:
            assert isinstance(
                gesture, dict), "Invalid JSON data, 'gestures' must be a list of dict"
            gestures.append(DataGestures.fromDict(gesture))

        tmp: DataSample2 = cls(
            label=str(json_data["label"]),
            gestures=gestures,
        )

        if "framerate" in json_data:
            assert isinstance(
                json_data["framerate"], int), "Invalid JSON data, 'framerate' must be an integer"
            tmp.framerate = int(json_data["framerate"])

        if "mirrorable" in json_data:
            assert isinstance(
                json_data["mirrorable"], bool), "Invalid JSON data, 'mirrorable' must be a boolean"
            tmp.mirrorable = bool(json_data["mirrorable"])
        tmp.mirrorable = bool(json_data.get("mirrorable", tmp.mirrorable))

        tmp.computeHandVelocity()
        return tmp

    @classmethod
    def fromJsonFile(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data: dict[str, object] = json.load(f)
        return cls.fromDict(data)

    @classmethod
    def unflat(
        cls, label: str, raw_data: list[float], valid_fields: list[str] = FIELDS
    ):
        tmp = cls(label=label, gestures=[])
        len_valid_fields: int = (len(valid_fields)) * FIELD_DIMENSION
        for i in range(0, len(raw_data), len_valid_fields):
            tmp.gestures.append(
                DataGestures.from1DArray(
                    raw_data[i: i + len_valid_fields], valid_fields
                )
            )
        tmp.computeHandVelocity()
        return tmp

    def toDict(self) -> dict[str, object]:
        tmp: dict[str, object] = copy.deepcopy(self).__dict__
        tmp["gestures"] = [gesture.__dict__ for gesture in self.gestures]
        # print(tmp["gestures"])
        return tmp

    def toJsonFile(self, file_path: str, indent: bool = False):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.toDict(), f, indent=indent)

    def toTensor(
        self,
        sequence_length: int,
        valid_fields: list[str] = FIELDS,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        data: torch.Tensor = torch.zeros(
            sequence_length, len(valid_fields) * FIELD_DIMENSION)
        for i in range(min(sequence_length, len(self.gestures))):
            data[i] = self.gestures[i].toTensor(valid_fields)
        return data.to(default_device(device))

    def to_onnx_tensor(
        self, sequence_length: int, valid_fields: list[str] = FIELDS
    ) -> NDArray[np.float32]:
        """
        Converts the DataSample2 object into an ONNX-compatible NumPy tensor.

        Args:
            sequence_length (int): The number of frames (sequence length) expected by the model.
            valid_fields (list[str], optional): The selected fields to include in the tensor.

        Returns:
            np.ndarray: ONNX-compatible tensor with shape (sequence_length, num_features)
        """
        # Determine the number of valid features
        num_features = len(valid_fields) * FIELD_DIMENSION

        # Initialize a zero tensor with the expected shape
        data = np.zeros((sequence_length, num_features), dtype=np.float32)

        # Fill the tensor with gesture data
        for i in range(min(sequence_length, len(self.gestures))):
            data[i, :] = self.gestures[i].toNumpy(valid_fields)

        return data

    def flat(self, valid_fields: list[str] = FIELDS) -> list[float]:
        """Transform the gesture data into a 1D array
        WARNING: This function will not store the framerate information nor the label
        """
        raw_data: list[float] = []
        for gesture in self.gestures:
            raw_data.extend(gesture.get1DArray(valid_fields))
        return raw_data

    def insertGestureFromLandmarks(
        self, position: int,
        handmark: HandLandmarkerResult = None,
        facemark: FaceLandmarkResult = None,
        bodymark: object = None
    ) -> Self:
        self.gestures.insert(
            position, DataGestures.buildFromLandmarkerResult(
                handmark, facemark, bodymark)
        )
        self.computeHandVelocity()
        return self

    def samples_to_1d_array(self, valid_fields: list[str] = FIELDS) -> list[float]:
        raw_data = []
        for gesture in self.gestures:
            raw_data.extend(gesture.get1DArray(valid_fields))
        return raw_data

    def setNonePointsRandomlyToRandomOrZero(self, proba: float = 0.1) -> Self:
        for gest in self.gestures:
            gest.setNonePointsRandomlyToRandomOrZero(proba)
        return self

    def computeHandVelocity(self) -> Self:
        for i in range(len(self.gestures) - 1):
            if (
                self.gestures[i].r_hand_position is not None
                and self.gestures[i + 1].r_hand_position is not None
            ):
                r1 = cast(list[float], self.gestures[i].r_hand_position)
                r2 = cast(list[float], self.gestures[i + 1].r_hand_position)

                self.gestures[i].r_hand_velocity = [r1[k] - r2[k]
                                                    for k in range(3)]
            if (
                self.gestures[i].l_hand_position is not None
                and self.gestures[i + 1].l_hand_position is not None
            ):
                r1 = cast(list[float], self.gestures[i].l_hand_position)
                r2 = cast(list[float], self.gestures[i + 1].l_hand_position)

                self.gestures[i].l_hand_velocity = [r1[k] - r2[k]
                                                    for k in range(3)]
        return self

    def noise_sample(
        self, range: float = 0.004, valid_fields: list[str] = FIELDS
    ) -> Self:
        """Will randomize the sample gesture points by doing `new_val = old_val + rand_val(-range, range)` to each selected point.

        Args:
            range (float, optional): Random value will be between -range and range. Defaults to 0.005.
            valid_fields (list[str], optional): Let you pick which fields should be randomized. Defaults to None (All point affected).

        Returns:
            DataSample2: Return this class instance for chaining
        """
        for gesture in self.gestures:
            gesture.noise(range, valid_fields)
        return self

    def mirror_sample(
        self, x: bool = True, y: bool = False, z: bool = False
    ) -> Self:
        for gesture in self.gestures:
            gesture.mirror(x, y, z)
        return self

    def rotate_sample(
        self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] = FIELDS
    ) -> Self:
        for gesture in self.gestures:
            gesture.rotate(x, y, z, valid_fields)
        return self

    def scale_sample(
        self, x: float = 1, y: float = 1, z: float = 1, valid_fields: list[str] = FIELDS
    ) -> Self:
        for gesture in self.gestures:
            gesture.scale(x, y, z, valid_fields)
        return self

    def translate_sample(
        self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] = FIELDS
    ) -> Self:
        for gesture in self.gestures:
            gesture.translate(x, y, z, valid_fields)
        return self

    def reframe(self, target_frame: int) -> Self:
        """Change the number of frame to execute the full gesture sequence
        Be careful, if frame are reducedn reincresing the frame will not restore the original gesture

        Args:
            frame (int): Target frame number
        """
        if target_frame <= 1:
            raise ValueError("Target frame must be greater than 1")

        def list_lerp(a: list[float] | None, b: list[float] | None, t: float):
            if a is None or b is None:
                return None
            return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]

        new_gestures: list[DataGestures] = []

        for i in range(target_frame):
            progression = i / (target_frame - 1)
            frame_scaled_value = min(
                progression * (len(self.gestures) - 1), len(self.gestures) - 1
            )
            # print(i, frame_scaled_value, len(self.gestures))
            start_frame = math.floor(frame_scaled_value)
            end_frame = math.ceil(frame_scaled_value)
            interpolation_coef = frame_scaled_value - start_frame

            new_gesture: DataGestures = DataGestures()

            for field in fields(new_gesture):
                field_start: list[float] | None = getattr(
                    self.gestures[start_frame], field.name)
                field_end: list[float] | None = getattr(
                    self.gestures[end_frame], field.name)
                setattr(
                    new_gesture,
                    field.name,
                    list_lerp(
                        field_start,
                        field_end,
                        interpolation_coef,
                    ),
                )

            new_gestures.append(new_gesture)

        self.gestures = new_gestures
        # print(self.gestures)
        self.computeHandVelocity()
        return self

    def set_sample_gestures_point_to(
        self, point_field_name: str, value: list[float]
    ) -> Self:
        for gesture in self.gestures:
            setattr(gesture, point_field_name, value)
        return self

    def swap_hands(self) -> Self:
        """Should not be used.<br>
        This function is used to swap the left hand and right hand data,
        in case the hands are mirrored or the data is not in the right order.

        Returns:
            DataSample2: _description_
        """
        for gesture in self.gestures:
            gesture.swapHands()
        return self

    def move_to_one_side(self, right_side: bool = True) -> Self:
        for gesture in self.gestures:
            gesture.moveToOneSide(right_side)
        return self
