import math
import random
import torch
from dataclasses import dataclass, fields
from typing import Generic, TypeVar

import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *

from src.rot_3d import *
from src.tools import *


def is_valid_field(field_name: str, valid_fields: list[str] | None) -> bool:
    return valid_fields is None or field_name in valid_fields

T = TypeVar('T')
@dataclass
class Gestures(Generic[T]):
    # NEVER CHANGE THE POINTS ORDER OR IT WILL BREAK BACKWARD COMPATIBILITY

    # Always start your variable name with the hand side (l_ or r_)
    # Method move_one_side() use this prefix to work

    # Left hand data
    l_hand_position: T = None
    l_wrist: T = None
    l_thumb_cmc: T = None
    l_thumb_mcp: T = None
    l_thumb_ip: T = None
    l_thumb_tip: T = None
    l_index_mcp: T = None
    l_index_pip: T = None
    l_index_dip: T = None
    l_index_tip: T = None
    l_middle_mcp: T = None
    l_middle_pip: T = None
    l_middle_dip: T = None
    l_middle_tip: T = None
    l_ring_mcp: T = None
    l_ring_pip: T = None
    l_ring_dip: T = None
    l_ring_tip: T = None
    l_pinky_mcp: T = None
    l_pinky_pip: T = None
    l_pinky_dip: T = None
    l_pinky_tip: T = None

    # Right hand data
    r_hand_position: T = None
    r_wrist: T = None
    r_thumb_cmc: T = None
    r_thumb_mcp: T = None
    r_thumb_ip: T = None
    r_thumb_tip: T = None
    r_index_mcp: T = None
    r_index_pip: T = None
    r_index_dip: T = None
    r_index_tip: T = None
    r_middle_mcp: T = None
    r_middle_pip: T = None
    r_middle_dip: T = None
    r_middle_tip: T = None
    r_ring_mcp: T = None
    r_ring_pip: T = None
    r_ring_dip: T = None
    r_ring_tip: T = None
    r_pinky_mcp: T = None
    r_pinky_pip: T = None
    r_pinky_dip: T = None
    r_pinky_tip: T = None

    l_hand_velocity: T = None
    r_hand_velocity: T = None

    def __new__(cls, *args, **kwargs):
        print(f"Creating instance of {cls.__name__}")
        return super().__new__(cls)  # Ensures correct instance type

    @classmethod
    def from1DArray(cls, array: list[T], valid_fields: list[str] = None) -> "Gestures":
        tmp = cls()
        valid_fields = get_fields(valid_fields)
        for i, field_name in enumerate(valid_fields):
            setattr(tmp, field_name, array[i * FIELD_DIMENSION: (i + 1) * FIELD_DIMENSION])
        return tmp

    @classmethod
    def fromDict(cls, data: dict[str, T], valid_fields: list[str] = None) -> "Gestures":
        tmp = cls()
        valid_fields = get_fields(valid_fields)
        for field_name in valid_fields:
            setattr(tmp, field_name, data.get(field_name))
        return tmp

    def setFieldsTo(self, value: T, valid_fields: list[str] = None) -> "Gestures":
        if valid_fields is None:
            valid_fields = FIELDS
        for field_name in valid_fields:
            setattr(self, field_name, value)
        return self

    def to_dict(self) -> dict[str, bool]:
        return self.__dict__

FIELDS: list[str] = [field.name for field in fields(Gestures())]
FIELD_DIMENSION: int = 3

@dataclass
class ActiveGestures(Gestures[bool | None]):
    """ActiveGestures class defines what the model will take into account when predicting gestures.
    For example, if the model is only interested in the position of the hands, then only the hand_position fields will be set to True,
    the rest such as the fingers positions will be set to False and will be ignored by the model.

    Args:
        Gestures (_type_): _description_
    """
    @classmethod
    def buildWithPreset(self, gestures_to_set: Gestures[bool | None] | list[Gestures[bool | None]]) -> "ActiveGestures":
        tmp = ActiveGestures()
        return tmp.setActiveGestures(gestures_to_set)

    def setActiveGestures(self, gestures_to_set: Gestures[bool | None] | list[Gestures[bool | None]]) -> "ActiveGestures":
        if not isinstance(gestures_to_set, list):
            gestures_to_set = [gestures_to_set]

        # print("===", gestures_to_set)

        self.resetActiveGestures()

        for gesture in gestures_to_set:
            # print("---")
            for field_name in FIELDS:
                field_data = getattr(gesture, field_name)
                # print("x>", field_name, getattr(self, field_name))
                if field_data is not None:
                    setattr(self, field_name, field_data)
                # print("=>", field_name, getattr(self, field_name))
        return self

    def resetActiveGestures(self, valid_fields: list[str] = None) -> "ActiveGestures":
        return self.setFieldsTo(None, valid_fields)

    def activateAllGesture(self, valid_fields: list[str] = None) -> "ActiveGestures":
        return self.setFieldsTo(True, valid_fields)

    def deactivateAllGesture(self, valid_fields: list[str] = None) -> "ActiveGestures":
        return self.setFieldsTo(False, valid_fields)

    def getActiveFields(self) -> list[str]:
        active_fields = []
        for field_name in FIELDS:
            if getattr(self, field_name):
                active_fields.append(field_name)
        return active_fields


LEFT_HAND_POINTS: ActiveGestures = ActiveGestures(
    l_wrist=True,
    l_thumb_cmc=True,
    l_thumb_mcp=True,
    l_thumb_ip=True,
    l_thumb_tip=True,
    l_index_mcp=True,
    l_index_pip=True,
    l_index_dip=True,
    l_index_tip=True,
    l_middle_mcp=True,
    l_middle_pip=True,
    l_middle_dip=True,
    l_middle_tip=True,
    l_ring_mcp=True,
    l_ring_pip=True,
    l_ring_dip=True,
    l_ring_tip=True,
    l_pinky_mcp=True,
    l_pinky_pip=True,
    l_pinky_dip=True,
    l_pinky_tip=True,
)
LEFT_HAND_POSITION: ActiveGestures = ActiveGestures(
    l_hand_position=True
)
LEFT_HAND_VELOCITY: ActiveGestures = ActiveGestures(
    l_hand_velocity=True
)
LEFT_HAND_FULL: ActiveGestures = ActiveGestures.buildWithPreset([LEFT_HAND_POINTS, LEFT_HAND_POSITION, LEFT_HAND_VELOCITY])
RIGHT_HAND_POINTS: ActiveGestures = ActiveGestures(
    r_wrist=True,
    r_thumb_cmc=True,
    r_thumb_mcp=True,
    r_thumb_ip=True,
    r_thumb_tip=True,
    r_index_mcp=True,
    r_index_pip=True,
    r_index_dip=True,
    r_index_tip=True,
    r_middle_mcp=True,
    r_middle_pip=True,
    r_middle_dip=True,
    r_middle_tip=True,
    r_ring_mcp=True,
    r_ring_pip=True,
    r_ring_dip=True,
    r_ring_tip=True,
    r_pinky_mcp=True,
    r_pinky_pip=True,
    r_pinky_dip=True,
    r_pinky_tip=True,
)
RIGHT_HAND_POSITION: ActiveGestures = ActiveGestures(
    r_hand_position=True
)
RIGHT_HAND_VELOCITY: ActiveGestures = ActiveGestures(
    r_hand_velocity=True
)
RIGHT_HAND_FULL: ActiveGestures = ActiveGestures.buildWithPreset([RIGHT_HAND_POINTS, RIGHT_HAND_POSITION, RIGHT_HAND_VELOCITY])

HANDS_POINTS: ActiveGestures = ActiveGestures.buildWithPreset([LEFT_HAND_POINTS, RIGHT_HAND_POINTS])
HANDS_POSITION: ActiveGestures = ActiveGestures.buildWithPreset([LEFT_HAND_POSITION, RIGHT_HAND_POSITION])
HANDS_VELOCITY: ActiveGestures = ActiveGestures.buildWithPreset([LEFT_HAND_VELOCITY, RIGHT_HAND_VELOCITY])

HANDS_FULL: ActiveGestures = ActiveGestures.buildWithPreset([LEFT_HAND_FULL, RIGHT_HAND_FULL])
ALL_GESTURES: ActiveGestures = ActiveGestures()
ALL_GESTURES.activateAllGesture()

ACTIVATED_GESTURES_PRESETS: dict[str, tuple[ActiveGestures, str]] = {
    'all': (
        ALL_GESTURES,
        "Will include every available point."
    ),
    'left_hand_points': (
        LEFT_HAND_POINTS,
        "Will only provide information about left hand finger position or hand rotation."
    ),
    'left_hand_position': (
        LEFT_HAND_POSITION,
        "Will only provide information about left hand position."
    ),
    'left_hand_velocity': (
        LEFT_HAND_VELOCITY,
        "Will only provide information about left hand velocity."
    ),
    'left_hand_full': (
        LEFT_HAND_FULL,
        "Will provide information about left hand finger position, hand rotation and position."
    ),
    'right_hand_points': (
        RIGHT_HAND_POINTS,
        "Will only provide information about right hand finger position or hand rotation."
    ),
    'right_hand_position': (
        RIGHT_HAND_POSITION,
        "Will only provide information about right hand position."
    ),
    'right_hand_velocity': (
        RIGHT_HAND_VELOCITY,
        "Will only provide information about right hand velocity."
    ),
    'right_hand_full': (
        RIGHT_HAND_FULL,
        "Will provide information about right hand finger position, hand rotation and position."
    ),
    'hands_points': (
        HANDS_POINTS,
        "Will only provide information about both hands finger position and hands rotation."
    ),
    'hands_position': (
        HANDS_POSITION,
        "Will only provide information about both hands position."
    ),
    'hands_velocity': (
        HANDS_VELOCITY,
        "Will only provide information about both hands velocity."
    ),
    'hands_full': (
        HANDS_FULL,
        "Will provide information about both hands finger position, hands rotation and position."
    )
}

CACHE_HANDS_POINTS: list[str] = HANDS_POINTS.getActiveFields()
CACHE_HANDS_POSITION: list[str] = HANDS_POSITION.getActiveFields()

def get_fields(valid_fields: list[str] | None = None) -> list[str]:
    if valid_fields is None:
        return FIELDS
    return valid_fields

@dataclass
class DataGestures(Gestures[list[float, float, float] | None]):

    @classmethod
    def buildFromHandLandmarkerResult(self, landmark_result: HandLandmarkerResult, valid_fields: list[str] = None) -> "DataGestures":
        tmp = DataGestures()
        tmp.setHandsFromHandLandmarkerResult(landmark_result, valid_fields)
        return tmp

    @classmethod
    def from1DArray(self, array: list[float], valid_fields: list[str] = None) -> "DataGestures":
        tmp = DataGestures()
        valid_fields = get_fields(valid_fields)
        for i, field_name in enumerate(valid_fields):
            setattr(tmp, field_name, array[i * FIELD_DIMENSION: (i + 1) * FIELD_DIMENSION])
        return tmp

    def setHandsFromHandLandmarkerResult(self, landmark_result: HandLandmarkerResult, valid_fields: list[str] = None) -> "DataGestures":
        """Convert the HandLandmarkerResult object into a DataGestures object.

        HandLandmarkerResult.hand_landmark represent the position of the hand in the image.
        HandLandmarkerResult.hand_world_landmarks represent a normalized hand that is not altered by the position or distance of the camera.

        Args:
            landmark_result (HandLandmarkerResult): _description_
        """

        hand_fields: list[str] = [
            "wrist",
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",
            "index_mcp",
            "index_pip",
            "index_dip",
            "index_tip",
            "middle_mcp",
            "middle_pip",
            "middle_dip",
            "middle_tip",
            "ring_mcp",
            "ring_pip",
            "ring_dip",
            "ring_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip"
        ]

        for i in range(len(landmark_result.hand_world_landmarks)):
            handlandmark: list[NormalizedLandmark] = landmark_result.hand_landmarks[i]
            handworldlandmark: list[NormalizedLandmark] = landmark_result.hand_world_landmarks[i]

            if landmark_result.handedness[i][0].category_name == "Right":
                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                if valid_fields is None or "r_hand_position" in valid_fields:
                    self.r_hand_position = [handlandmark[0].x - 0.5, handlandmark[0].y - 0.5, handlandmark[0].z - 0.5]

                # Adding position of each finger articulation
                for j, field_name in enumerate(hand_fields):
                    if valid_fields is None or f"r_{field_name}" in valid_fields:
                        setattr(self, f"r_{field_name}", [handworldlandmark[j].x, handworldlandmark[j].y, handworldlandmark[j].z])

            else:
                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                if valid_fields is None or "l_hand_position" in valid_fields:
                    self.l_hand_position = [handlandmark[0].x, handlandmark[0].y, handlandmark[0].z]

                # Adding position of each finger articulation
                for j, field_name in enumerate(hand_fields):
                    if valid_fields is None or f"l_{field_name}" in valid_fields:
                        setattr(self, f"l_{field_name}", [handworldlandmark[j].x, handworldlandmark[j].y, handworldlandmark[j].z])

        return self

    def setPointTo(self, point_field_name, x: float, y: float, z: float) -> "DataGestures":
        setattr(self, point_field_name, [x, y, z])
        return self

    def setPointToZero(self, point_field_name: str) -> "DataGestures":
        self.setPointTo(point_field_name, 0, 0, 0)
        return self

    def setPointToRandom(self, point: str) -> "DataGestures":
        if point in CACHE_HANDS_POSITION:
            self.setPointTo(point, rand_fix_interval(1), rand_fix_interval(1), rand_fix_interval(1))
        else: # elif point in CACHE_HANDS_POINTS:
            # 0.15 is the max value I can find on hand landmark
            self.setPointTo(point, rand_fix_interval(0.15), rand_fix_interval(0.15), rand_fix_interval(0.15))
        return self

    def setAllPointsToZero(self) -> "DataGestures":
        for field_name in FIELDS:
            self.setPointToZero(field_name)
        return self

    def setAllPointsToRandom(self) -> "DataGestures":
        for field_name in FIELDS:
            self.setPointToRandom(field_name)
        return self

    def setNonePointsToZero(self) -> "DataGestures":
        for field_name in FIELDS:
            if getattr(self, field_name) is None:
                self.setPointToZero(field_name)
        return self

    def setNonePointsToRandom(self) -> "DataGestures":
        for field_name in FIELDS:
            if getattr(self, field_name) is None:
                self.setPointToRandom(field_name)
        return self

    def setNonePointsRandomlyToRandomOrZero(self, proba: float = 0.1) -> "DataGestures":
        # Filter fields where the attribute is None
        none_fields = [field_name for field_name in FIELDS if getattr(self, field_name) is None]

        for field_name in none_fields:
            if random.random() < proba:
                setattr(self, field_name, [0, 0, 0])  # Replace setPointToZero with direct set to 0
            else:
                self.setPointToRandom(field_name)

        return self

    def get1DArray(self, valid_fields: list[str] = None) -> list[float]:
        valid_fields = get_fields(valid_fields)  # Récupérer les bons champs
        tmp = [coord for field_name in valid_fields for coord in (getattr(self, field_name, [0, 0, 0]) or [0, 0, 0])]
        # print(self, "\n")
        # print(tmp, "\n\n")
        return tmp

    def toNumpy(self, valid_fields: list[str] = FIELDS) -> np.ndarray:
        return np.array(self.get1DArray(valid_fields), dtype=np.float32)

    def toTensor(self, valid_fields: list[str] = FIELDS, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.as_tensor(self.get1DArray(valid_fields), dtype=torch.float32).to(device)

    def noise(self, range: float = 0.005, valid_fields: list[str] | None = None) -> "DataGestures":
        """Will randomize the gesture points by doing `new_val = old_val + rand_val(-range, range)` to each selected point.

        Args:
            range (float, optional): Random value will be between -range and range. Defaults to 0.005.
            valid_fields (list[str], optional): Let you pick which fields should be randomized. Defaults to None (All point affected).

        Returns:
            DataSample2: Return this class instance for chaining
        """
        valid_fields = get_fields(valid_fields)
        for field_name in valid_fields:
            field_value: list[float] = getattr(self, field_name)
            if field_value is not None:
                field_value[0] += rand_fix_interval(range)
                field_value[1] += rand_fix_interval(range)
                field_value[2] += rand_fix_interval(range)
                setattr(self, field_name, field_value)
        return self

    def mirror(self, x: bool = True, y: bool = False, z: bool = False) -> "DataGestures":
        for field_name in FIELDS:
            field_value: list[float] = getattr(self, field_name)
            if field_value is None:
                continue
            if x:
                field_value[0] *= -1
            if y:
                field_value[1] *= -1
            if z:
                field_value[2] *= -1
            setattr(self, field_name, field_value)

        # Mirroring the hand make the hand become the opposite hand
        # This if statement will swap the left hand and right hand data
        if (x + y + z) % 2 == 1:
            self.swapHands()
        return self

    def rotate(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] | None = None) -> "DataGestures":
        for field_name in get_fields(valid_fields):
            field_value: list[float] = getattr(self, field_name)
            if field_value is None:
                continue
            field_value = rot_3d_x(field_value, x)
            field_value = rot_3d_y(field_value, y)
            field_value = rot_3d_z(field_value, z)
            setattr(self, field_name, field_value)
        return self

    def scale(self, x: float = 1, y: float = 1, z: float = 1, valid_fields: list[str] | None = None) -> "DataGestures":
        valid_fields = get_fields(valid_fields)
        for field_name in valid_fields:
            field_value: list[float] = getattr(self, field_name)
            if field_value is None:
                continue
            field_value[0] *= x
            field_value[1] *= y
            field_value[2] *= z
            setattr(self, field_name, field_value)
        return self

    def translate(self, x: float = 0, y: float = 0, z: float = 0, valid_fields: list[str] | None = None) -> "DataGestures":
        valid_fields = get_fields(valid_fields)
        for field_name in valid_fields:
            field_value: list[float] = getattr(self, field_name)
            if field_value is None:
                continue
            field_value[0] += x
            field_value[1] += y
            field_value[2] += z
            setattr(self, field_name, field_value)
        return self

    def swapHands(self) -> "DataGestures":
        """Should not be used.<br>
        This function is used to swap the left hand and right hand data,
        in case the hands are mirrored or the data is not in the right order.

        Returns:
            DataGestures: _description_
        """

        self.r_wrist, self.l_wrist = self.l_wrist, self.r_wrist

        self.r_thumb_cmc, self.l_thumb_cmc = self.l_thumb_cmc, self.r_thumb_cmc
        self.r_thumb_mcp, self.l_thumb_mcp = self.l_thumb_mcp, self.r_thumb_mcp
        self.r_thumb_ip, self.l_thumb_ip = self.l_thumb_ip, self.r_thumb_ip
        self.r_thumb_tip, self.l_thumb_tip = self.l_thumb_tip, self.r_thumb_tip

        self.r_index_mcp, self.l_index_mcp = self.l_index_mcp, self.r_index_mcp
        self.r_index_pip, self.l_index_pip = self.l_index_pip, self.r_index_pip
        self.r_index_dip, self.l_index_dip = self.l_index_dip, self.r_index_dip
        self.r_index_tip, self.l_index_tip = self.l_index_tip, self.r_index_tip

        self.r_middle_mcp, self.l_middle_mcp = self.l_middle_mcp, self.r_middle_mcp
        self.r_middle_pip, self.l_middle_pip = self.l_middle_pip, self.r_middle_pip
        self.r_middle_dip, self.l_middle_dip = self.l_middle_dip, self.r_middle_dip
        self.r_middle_tip, self.l_middle_tip = self.l_middle_tip, self.r_middle_tip

        self.r_ring_mcp, self.l_ring_mcp = self.l_ring_mcp, self.r_ring_mcp
        self.r_ring_pip, self.l_ring_pip = self.l_ring_pip, self.r_ring_pip
        self.r_ring_dip, self.l_ring_dip = self.l_ring_dip, self.r_ring_dip
        self.r_ring_tip, self.l_ring_tip = self.l_ring_tip, self.r_ring_tip

        self.r_pinky_mcp, self.l_pinky_mcp = self.l_pinky_mcp, self.r_pinky_mcp
        self.r_pinky_pip, self.l_pinky_pip = self.l_pinky_pip, self.r_pinky_pip
        self.r_pinky_dip, self.l_pinky_dip = self.l_pinky_dip, self.r_pinky_dip
        self.r_pinky_tip, self.l_pinky_tip = self.l_pinky_tip, self.r_pinky_tip

        self.r_hand_position, self.l_hand_position = self.l_hand_position, self.r_hand_position
        self.r_hand_velocity, self.l_hand_velocity = self.l_hand_velocity, self.r_hand_velocity

        return self

    def moveToOneSide(self, right_side: bool = True) -> "DataGestures":
        dest_side = "r_" if right_side else "l_"
        src_side = "l_" if right_side else "r_"

        for field_name in FIELDS:
            if field_name.startswith(src_side):
                src_side_val: list[float] | None = getattr(self, field_name)
                opposite_field_name = field_name.replace(src_side, dest_side, 1)
                dest_side_value: list[float] | None = getattr(self, field_name.replace(src_side, dest_side))
                if dest_side_value is None:
                    if src_side_val is not None:
                        src_side_val[0] *= -1
                        # src_side_val[1] *= -1
                        src_side_val[2] *= -1
                    setattr(self, opposite_field_name, src_side_val)
                    setattr(self, field_name, None)
