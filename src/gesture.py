import random
import torch
import mediapipe
from dataclasses import dataclass, fields
from typing import Generic, TypeVar, Self, final, cast, Sequence, NamedTuple
import math

import numpy as np
from numpy.typing import NDArray
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
from typings.mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from src.rot_3d import rot_3d_x, rot_3d_y, rot_3d_z
from src.tools import rand_fix_interval


def is_valid_field(field_name: str, valid_fields: list[str] | None) -> bool:
    return valid_fields is None or field_name in valid_fields


def default_device(device: torch.device | None = None) -> torch.device:
    return device if device is not None else torch.device("cpu")


def landmark_to_list(landmark: Landmark | NormalizedLandmark | None) -> tuple[float, float, float] | None:
    if landmark is None:
        return None
    if landmark.x is None and landmark.y is None and landmark.z is None:
        return None
    return (
        landmark.x if landmark.x is not None else 0.0,
        landmark.y if landmark.y is not None else 0.0,
        landmark.z if landmark.z is not None else 0.0
    )


class FaceLandmarkResult(NamedTuple):
    multi_face_landmarks: list[NormalizedLandmarkList]


class BodyLandmarkResult(NamedTuple):
    pose_landmarks: NormalizedLandmarkList


def get_dist_between_points(
    point1: tuple[float | None, float | None, float | None],
    point2: tuple[float | None, float | None, float | None]
) -> float | None:
    point1_full: tuple[float, float, float] = (
        point1[0] if point1[0] is not None else 0.0,
        point1[1] if point1[1] is not None else 0.0,
        point1[2] if point1[2] is not None else 0.0
    )
    point2_full: tuple[float, float, float] = (
        point2[0] if point2[0] is not None else 0.0,
        point2[1] if point2[1] is not None else 0.0,
        point2[2] if point2[2] is not None else 0.0
    )
    return math.dist(point1_full, point2_full)


T = TypeVar("T")
FIELD_DIMENSION: int = 3


@dataclass
class _Gestures(Generic[T]):
    # NEVER CHANGE THE POINTS ORDER OR IT WILL BREAK BACKWARD COMPATIBILITY

    # Always start your variable name with the hand side (l_ or r_)
    # Method move_one_side() use this prefix to work

    # Left hand data
    l_hand_position: T | None = None
    l_wrist: T | None = None
    l_thumb_cmc: T | None = None
    l_thumb_mcp: T | None = None
    l_thumb_ip: T | None = None
    l_thumb_tip: T | None = None
    l_index_mcp: T | None = None
    l_index_pip: T | None = None
    l_index_dip: T | None = None
    l_index_tip: T | None = None
    l_middle_mcp: T | None = None
    l_middle_pip: T | None = None
    l_middle_dip: T | None = None
    l_middle_tip: T | None = None
    l_ring_mcp: T | None = None
    l_ring_pip: T | None = None
    l_ring_dip: T | None = None
    l_ring_tip: T | None = None
    l_pinky_mcp: T | None = None
    l_pinky_pip: T | None = None
    l_pinky_dip: T | None = None
    l_pinky_tip: T | None = None

    # Right hand data
    r_hand_position: T | None = None
    r_wrist: T | None = None
    r_thumb_cmc: T | None = None
    r_thumb_mcp: T | None = None
    r_thumb_ip: T | None = None
    r_thumb_tip: T | None = None
    r_index_mcp: T | None = None
    r_index_pip: T | None = None
    r_index_dip: T | None = None
    r_index_tip: T | None = None
    r_middle_mcp: T | None = None
    r_middle_pip: T | None = None
    r_middle_dip: T | None = None
    r_middle_tip: T | None = None
    r_ring_mcp: T | None = None
    r_ring_pip: T | None = None
    r_ring_dip: T | None = None
    r_ring_tip: T | None = None
    r_pinky_mcp: T | None = None
    r_pinky_pip: T | None = None
    r_pinky_dip: T | None = None
    r_pinky_tip: T | None = None

    l_hand_velocity: T | None = None
    r_hand_velocity: T | None = None

    # MID FACE SET
    m_nose_point: T | None = None  # Middle nose point
    m_top_nose: T | None = None  # Middle Top nose
    m_eyebrows: T | None = None  # Middle of eyebrows
    m_forehead: T | None = None  # Middle forehead
    m_top_chin: T | None = None  # Top chin
    m_bot_up_lip: T | None = None  # Bottom upper lip
    m_top_low_lip: T | None = None  # Top lower lip
    m_bot_nose: T | None = None  # Bottom nose
    m_chin: T | None = None  # Middle chin
    m_nose: T | None = None  # Middle nose

    # LEFT FACE SET
    l_eye_exterior: T | None = None  # Left eye exterior
    l_temple: T | None = None  # Left temple
    l_mid_chin: T | None = None  # Left middle chin
    l_up_lip: T | None = None  # Left upper lip
    l_ext_nostril: T | None = None  # Exterior left nostril
    l_mid_cheek: T | None = None  # Middle left cheek
    l_mid_eyebrow: T | None = None  # Middle left eyebrow
    l_ext_eyebrow: T | None = None  # Left exterior eyebrow
    l_ext_lips: T | None = None  # Exterior left lips
    l_jaw_angle: T | None = None  # Left jaw angle
    l_mid_ext_face: T | None = None  # Left middle exterior face
    l_int_eyebrow: T | None = None  # Interor left eyebrow
    l_mid_jaw: T | None = None  # Middle left jaw
    l_mid_bot_eyelid: T | None = None  # Left eye middle bottom eyelid
    l_ext_mouth: T | None = None  # Left exterior mouth
    l_top_eyelid: T | None = None  # Left eye middle top eyelid
    l_eye_int: T | None = None  # Left eye interior
    l_pupil: T | None = None  # Left pupil

    # RIGHT FACE SET
    r_eye_exterior: T | None = None  # Right eye exterior
    r_temple: T | None = None  # Right temple
    r_mid_chin: T | None = None  # Right middle chin
    r_up_lip: T | None = None  # Right upper lip
    r_ext_nostril: T | None = None  # Exterior right nostril
    r_mid_cheek: T | None = None  # Middle right cheek
    r_mid_eyebrow: T | None = None  # Middle right eyebrow
    r_ext_eyebrow: T | None = None  # Right exterior eyebrow
    r_ext_lips: T | None = None  # Exterior right lips
    r_jaw_angle: T | None = None  # Right jaw angle
    r_mid_ext_face: T | None = None  # Right middle exterior face
    r_int_eyebrow: T | None = None  # Interor right eyebrow
    r_mid_jaw: T | None = None  # Middle right jaw
    r_mid_bot_eyelid: T | None = None  # Right eye middle bottom eyelid
    r_ext_mouth: T | None = None  # Right exterior mouth
    r_top_eyelid: T | None = None  # Right eye middle top eyelid
    r_eye_int: T | None = None  # Right eye interior
    r_pupil: T | None = None  # Right pupil

    l_shoulder: T | None = None  # Left shoulder
    l_elbow: T | None = None  # Left elbow
    l_hip: T | None = None  # Left hip
    l_knee: T | None = None  # Left knee
    l_ankle: T | None = None  # Left ankle
    l_body_wrist: T | None = None  # Left wrist, used for body gestures

    r_shoulder: T | None = None  # Right shoulder
    r_elbow: T | None = None  # Right elbow
    r_hip: T | None = None  # Right hip
    r_knee: T | None = None  # Right knee
    r_ankle: T | None = None  # Right ankle
    r_body_wrist: T | None = None  # Right wrist, used for body gestures

    m_face_position: T | None = None  # Middle face position
    m_body_position: T | None = None  # Middle body position
    m_face_scale: T | None = None  # Middle face scale
    m_body_scale: T | None = None  # Middle body scale
    l_hand_scale: T | None = None  # Left hand scale
    r_hand_scale: T | None = None  # Right hand scale


FIELDS: list[str] = [f.name for f in fields(_Gestures)]


class Gestures(_Gestures[T]):
    @classmethod
    def fromArray(cls, array: list[T], valid_fields: list[str] = FIELDS) -> Self:
        tmp = cls()
        for i, field_name in enumerate(valid_fields):
            setattr(
                tmp, field_name, array[i *
                                       FIELD_DIMENSION: (i + 1) * FIELD_DIMENSION]
            )
        return tmp

    @classmethod
    def fromDict(
        cls, data: dict[str, T], valid_fields: list[str] = FIELDS
    ) -> Self:
        tmp = cls()
        for field_name in valid_fields:
            setattr(tmp, field_name, data.get(field_name))
        return tmp

    def setFieldsTo(
        self, value: T | None, valid_fields: list[str] = FIELDS
    ) -> Self:
        for field_name in valid_fields:
            setattr(self, field_name, value)
        return self

    def getFields(self, valid_fields: list[str] = FIELDS) -> list[T | None]:
        """
        Get the values of the fields in a list.

        Args:
            valid_fields (list[str], optional): The names of the fields to get.
            Defaults to FIELDS.

        Returns:
            list[T | None]: The values of the fields.
        """
        return [getattr(self, field_name) for field_name in valid_fields]

    def getField(self, field_name: str) -> T | None:
        """
        Get the value of a field by its name.

        Args:
            field_name (str): The name of the field to get.

        Returns:
            T | None: The value of the field, or None if it doesn't exist.
        """
        return cast(T | None, getattr(self, field_name))

    def setField(self, field_name: str, value: T | None) -> Self:
        """
        Set the value of a field by its name.
        Args:
            field_name (str): The name of the field to set.
            value (T | None): The value to set.
        Returns:
            Self: The instance itself for method chaining.
        """
        setattr(self, field_name, value)
        return self

    def toDict(self) -> dict[str, bool]:
        return self.__dict__


@dataclass
class ActiveGestures(Gestures[bool | None]):
    """ActiveGestures class defines what the model will take into account when predicting gestures.
    For example, if the model is only interested in the position of the hands, then only the hand_position fields will be set to True,
    the rest such as the fingers positions will be set to False and will be ignored by the model.

    Args:
        Gestures (_type_): _description_
    """

    @classmethod
    def buildWithPreset(cls,
                        gestures_to_set: Self | Sequence[Self]
                        ) -> Self:
        return cls().setActiveGestures(gestures_to_set)

    def setActiveGestures(
        self, gestures_to_set: Self | Sequence[Self]
    ) -> Self:
        if not isinstance(gestures_to_set, Sequence):
            gestures_to_set = [gestures_to_set]

        # print("===", gestures_to_set)

        self.resetActiveGestures()

        for gesture in gestures_to_set:
            # print("---")
            for field_name in FIELDS:
                field_data: bool | None = getattr(gesture, field_name)
                # print("x>", field_name, getattr(self, field_name))
                if field_data is not None:
                    setattr(self, field_name, field_data)
                # print("=>", field_name, getattr(self, field_name))
        return self

    def resetActiveGestures(self, valid_fields: list[str] = FIELDS) -> Self:
        return self.setFieldsTo(None, valid_fields)

    def activateAllGesture(self, valid_fields: list[str] = FIELDS) -> Self:
        return self.setFieldsTo(True, valid_fields)

    def deactivateAllGesture(self, valid_fields: list[str] = FIELDS) -> Self:
        return self.setFieldsTo(False, valid_fields)

    def getActiveFields(self) -> list[str]:
        active_fields: list[str] = []
        for field_name in FIELDS:
            if self.getField(field_name):
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
LEFT_HAND_POSITION: ActiveGestures = ActiveGestures(l_hand_position=True)
LEFT_HAND_VELOCITY: ActiveGestures = ActiveGestures(l_hand_velocity=True)
LEFT_HAND_FULL: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_HAND_POINTS, LEFT_HAND_POSITION, LEFT_HAND_VELOCITY]
)
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
RIGHT_HAND_POSITION: ActiveGestures = ActiveGestures(r_hand_position=True)
RIGHT_HAND_VELOCITY: ActiveGestures = ActiveGestures(r_hand_velocity=True)
RIGHT_HAND_FULL: ActiveGestures = ActiveGestures.buildWithPreset(
    [RIGHT_HAND_POINTS, RIGHT_HAND_POSITION, RIGHT_HAND_VELOCITY]
)

HANDS_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_HAND_POINTS, RIGHT_HAND_POINTS]
)
HANDS_POSITION: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_HAND_POSITION, RIGHT_HAND_POSITION]
)
HANDS_VELOCITY: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_HAND_VELOCITY, RIGHT_HAND_VELOCITY]
)

HANDS_FULL: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_HAND_FULL, RIGHT_HAND_FULL]
)

LEFT_BODY_POINTS: ActiveGestures = ActiveGestures(
    l_shoulder=True,
    l_elbow=True,
    l_hip=True,
    l_knee=True,
    l_ankle=True,
    l_body_wrist=True
)

RIGHT_BODY_POINTS: ActiveGestures = ActiveGestures(
    r_shoulder=True,
    r_elbow=True,
    r_hip=True,
    r_knee=True,
    r_ankle=True,
    r_body_wrist=True
)

BODY_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_BODY_POINTS, RIGHT_BODY_POINTS]
)

LEFT_ARM_POINTS: ActiveGestures = ActiveGestures(
    l_shoulder=True,
    l_elbow=True,
    l_body_wrist=True
)

RIGHT_ARM_POINTS: ActiveGestures = ActiveGestures(
    r_shoulder=True,
    r_elbow=True,
    r_body_wrist=True
)

ARM_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_ARM_POINTS, RIGHT_ARM_POINTS]
)

MID_FACE_POINTS: ActiveGestures = ActiveGestures(
    m_nose_point=True,
    m_top_nose=True,
    m_eyebrows=True,
    m_forehead=True,
    m_top_chin=True,
    m_bot_up_lip=True,
    m_top_low_lip=True,
    m_bot_nose=True,
    m_chin=True,
    m_nose=True
)

LEFT_FACE_POINTS: ActiveGestures = ActiveGestures(
    l_eye_exterior=True,
    l_temple=True,
    l_mid_chin=True,
    l_up_lip=True,
    l_ext_nostril=True,
    l_mid_cheek=True,
    l_mid_eyebrow=True,
    l_ext_eyebrow=True,
    l_ext_lips=True,
    l_jaw_angle=True,
    l_mid_ext_face=True,
    l_int_eyebrow=True,
    l_mid_jaw=True,
    l_mid_bot_eyelid=True,
    l_ext_mouth=True,
    l_top_eyelid=True,
    l_eye_int=True,
    l_pupil=True
)

RIGHT_FACE_POINTS: ActiveGestures = ActiveGestures(
    r_eye_exterior=True,
    r_temple=True,
    r_mid_chin=True,
    r_up_lip=True,
    r_ext_nostril=True,
    r_mid_cheek=True,
    r_mid_eyebrow=True,
    r_ext_eyebrow=True,
    r_ext_lips=True,
    r_jaw_angle=True,
    r_mid_ext_face=True,
    r_int_eyebrow=True,
    r_mid_jaw=True,
    r_mid_bot_eyelid=True,
    r_ext_mouth=True,
    r_top_eyelid=True,
    r_eye_int=True,
    r_pupil=True
)

MIDDLE_FACE_POINTS: ActiveGestures = ActiveGestures(
    m_nose_point=True,
    m_top_nose=True,
    m_eyebrows=True,
    m_forehead=True,
    m_top_chin=True,
    m_bot_up_lip=True,
    m_top_low_lip=True,
    m_bot_nose=True,
    m_chin=True,
    m_nose=True
)

FACE_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_FACE_POINTS, MIDDLE_FACE_POINTS, RIGHT_FACE_POINTS]
)

HANDS_BODY_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [HANDS_POINTS, BODY_POINTS]
)

HANDS_FACE_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [HANDS_POINTS, FACE_POINTS]
)

HANDS_BODY_FACE_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [HANDS_POINTS, BODY_POINTS, FACE_POINTS]
)

HANDS_ARMS_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [HANDS_POINTS, ARM_POINTS]
)

HANDS_FACE_ARMS_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [HANDS_POINTS, FACE_POINTS, ARM_POINTS]
)

ALL_GESTURES: ActiveGestures = ActiveGestures()
ALL_GESTURES.activateAllGesture()

ACTIVATED_GESTURES_PRESETS: dict[str, tuple[ActiveGestures, str]] = {
    "all": (ALL_GESTURES, "Will include every available point."),
    "left_hand_points": (
        LEFT_HAND_POINTS,
        "Will only provide information about left hand finger position or hand rotation.",
    ),
    "left_hand_position": (
        LEFT_HAND_POSITION,
        "Will only provide information about left hand position.",
    ),
    "left_hand_velocity": (
        LEFT_HAND_VELOCITY,
        "Will only provide information about left hand velocity.",
    ),
    "left_hand_full": (
        LEFT_HAND_FULL,
        "Will provide information about left hand finger position, hand rotation and position.",
    ),
    "right_hand_points": (
        RIGHT_HAND_POINTS,
        "Will only provide information about right hand finger position or hand rotation.",
    ),
    "right_hand_position": (
        RIGHT_HAND_POSITION,
        "Will only provide information about right hand position.",
    ),
    "right_hand_velocity": (
        RIGHT_HAND_VELOCITY,
        "Will only provide information about right hand velocity.",
    ),
    "right_hand_full": (
        RIGHT_HAND_FULL,
        "Will provide information about right hand finger position, hand rotation and position.",
    ),
    "hands_points": (
        HANDS_POINTS,
        "Will only provide information about both hands finger position and hands rotation.",
    ),
    "hands_position": (
        HANDS_POSITION,
        "Will only provide information about both hands position.",
    ),
    "hands_velocity": (
        HANDS_VELOCITY,
        "Will only provide information about both hands velocity.",
    ),
    "hands_full": (
        HANDS_FULL,
        "Will provide information about both hands finger position, hands rotation and position.",
    ),
    "left_body_points": (
        LEFT_BODY_POINTS,
        "Will only provide information about left body points such as shoulder, elbow, hip, knee, ankle and wrist.",
    ),
    "right_body_points": (
        RIGHT_BODY_POINTS,
        "Will only provide information about right body points such as shoulder, elbow, hip, knee, ankle and wrist.",
    ),
    "body_points": (
        BODY_POINTS,
        "Will only provide information about both body points such as shoulder, elbow, hip, knee, ankle and wrist.",
    ),
    "left_arm_points": (
        LEFT_ARM_POINTS,
        "Will only provide information about left arm points such as shoulder, elbow and wrist.",
    ),
    "right_arm_points": (
        RIGHT_ARM_POINTS,
        "Will only provide information about right arm points such as shoulder, elbow and wrist.",
    ),
    "arm_points": (
        ARM_POINTS,
        "Will only provide information about both arm points such as shoulder, elbow and wrist.",
    ),
    "mid_face_points": (
        MID_FACE_POINTS,
        "Will only provide information about middle face points such as nose, eyebrows, forehead, chin, lips.",
    ),
    "left_face_points": (
        LEFT_FACE_POINTS,
        "Will only provide information about left face points such as eye, temple, chin, lips, nostril, cheek, eyebrow.",
    ),
    "right_face_points": (
        RIGHT_FACE_POINTS,
        "Will only provide information about right face points such as eye, temple, chin, lips, nostril, cheek, eyebrow.",
    ),
    "face_points": (
        FACE_POINTS,
        "Will only provide information about face points such as eye, temple, chin, lips, nostril, cheek, eyebrow.",
    ),
    "hands_body_points": (
        HANDS_BODY_POINTS,
        "Will provide information about both hands finger position, hands rotation and body points such as shoulder, elbow, hip, knee, ankle and wrist.",
    ),
    "hands_face_points": (
        HANDS_FACE_POINTS,
        "Will provide information about both hands finger position, hands rotation and face points such as eye, temple, chin, lips, nostril, cheek, eyebrow.",
    ),
    "hands_body_face_points": (
        HANDS_BODY_FACE_POINTS,
        "Will provide information about both hands finger position, hands rotation, body points such as shoulder, elbow, hip, knee, ankle and wrist and face points such as eye, temple, chin, lips, nostril, cheek, eyebrow.",
    ),
    "hands_face_arms_points": (
        HANDS_FACE_ARMS_POINTS,
        "Will provide information about both hands finger position, hands rotation, face points such as eye, temple, chin, lips, nostril, cheek, eyebrow and arm points such as shoulder, elbow and wrist.",
    ),
    "hands_arms_points": (
        HANDS_ARMS_POINTS,
        "Will provide information about both hands finger position, hands rotation and arm points such as shoulder, elbow and wrist.",
    ),
}

CACHE_HANDS_POINTS: list[str] = HANDS_POINTS.getActiveFields()
CACHE_HANDS_POSITION: list[str] = HANDS_POSITION.getActiveFields()


@final
@dataclass
class DataGestures(Gestures[tuple[float, float, float] | None]):
    @classmethod
    def buildFromLandmarkerResult(
        cls,
        landmark_result: HandLandmarkerResult | None = None,
        facemark_result: FaceLandmarkResult | None = None,
        bodymark_result: BodyLandmarkResult | None = None,
    ) -> Self:
        tmp = cls()
        return tmp.setHandsFromHandLandmarkerResult(landmark_result, facemark_result, bodymark_result)

    @classmethod
    def from1DArray(
        cls, array: list[float], valid_fields: list[str] = FIELDS
    ) -> Self:
        cls = cls()
        for i, field_name in enumerate(valid_fields):
            cls.setPointTo(field_name,
                           array[i * FIELD_DIMENSION],
                           array[i * FIELD_DIMENSION + 1],
                           array[i * FIELD_DIMENSION + 2])
        return cls

    def setHandsFromHandLandmarkerResult(
        self,
        landmark_result: HandLandmarkerResult | None = None,
        facemark_result: FaceLandmarkResult | None = None,
        bodymark_result: BodyLandmarkResult | None = None
    ) -> Self:
        """Convert the HandLandmarkerResult object into a DataGestures object.

        HandLandmarkerResult.hand_landmark represent the position of the hand in the image.
        HandLandmarkerResult.hand_world_landmarks represent a normalized hand that is not altered by the position or distance of the camera.

        Args:
            landmark_result (HandLandmarkerResult): _description_
        """
        scale: float | None
        tmp: tuple[float, float, float] | None

        if landmark_result is not None:
            hand_fields: list[str] = [
                "wrist", # 0
                "thumb_cmc", # 1
                "thumb_mcp", # 2
                "thumb_ip", # 3
                "thumb_tip", # 4
                "index_mcp", # 5
                "index_pip", # 6
                "index_dip", # 7
                "index_tip", # 8
                "middle_mcp", # 9
                "middle_pip", # 10
                "middle_dip", # 11
                "middle_tip", # 12
                "ring_mcp", # 13
                "ring_pip", # 14
                "ring_dip", # 15
                "ring_tip", # 16
                "pinky_mcp", # 17
                "pinky_pip", # 18
                "pinky_dip", # 19
                "pinky_tip", # 20
            ]
            hand_pos_id: int = 9
            hand_pos_id2: int = 0

            for i in range(len(landmark_result.hand_world_landmarks)):
                # Hand world placed
                handlandmark: list[NormalizedLandmark] = landmark_result.hand_landmarks[i]

                # print(handlandmark, handworldlandmark)
                prefix: str = "l_" if landmark_result.handedness[i][0].category_name == "Left" else "r_"

                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                hand_pos: tuple[float, float, float] = landmark_to_list(handlandmark[hand_pos_id]) or (0.0, 0.0, 0.0)

                scale = get_dist_between_points(
                            (handlandmark[hand_pos_id].x, handlandmark[hand_pos_id].y, handlandmark[hand_pos_id].z),
                            (handlandmark[hand_pos_id2].x, handlandmark[hand_pos_id2].y, handlandmark[hand_pos_id2].z)
                        ) or 1.0
                self.setPointTo(f"{prefix}hand_scale", scale, scale, scale)

                for j, field_name in enumerate(hand_fields):
                    tmp = landmark_to_list(handlandmark[j])
                    if tmp is not None:
                        tmp = (
                            (tmp[0] - hand_pos[0]) / scale,
                            (tmp[1] - hand_pos[1]) / scale,
                            (tmp[2] - hand_pos[2]) / scale
                        )
                    setattr(self, f"{prefix}{field_name}", tmp)

                self.setPointTo(f"{prefix}hand_position",
                                hand_pos[0] - 0.5,
                                hand_pos[1] - 0.5,
                                hand_pos[2] - 0.5)


        if facemark_result is not None and len(facemark_result.multi_face_landmarks) > 0:
            face_fields: dict[str, int] = {
                "m_nose_point": 1,  # Middle nose point
                "m_top_nose": 6,  # Middle Top nose
                "m_eyebrows": 9,  # Middle of eyebrows
                "m_forehead": 10,  # Middle forehead
                "m_top_chin": 18,  # Top chin
                "m_bot_up_lip": 13,  # Bottom upper lip
                "m_top_low_lip": 14,  # Top lower lip
                "m_bot_nose": 141,  # Bottom nose
                "m_chin": 152,  # Middle chin
                "m_nose": 197,  # Middle nose
                "l_eye_exterior": 7,  # Left eye exterior
                "l_temple": 21,  # Left temple
                "l_mid_chin": 32,  # Left middle chin
                "l_up_lip": 39,  # Left upper lip
                "l_ext_nostril": 48,  # Exterior left nostril
                "l_mid_cheek": 50,  # Middle left cheek
                "l_mid_eyebrow": 52,  # Middle left eyebrow
                "l_ext_eyebrow": 53,  # Left exterior eyebrow
                "l_ext_lips": 57,  # Exterior left lips
                "l_jaw_angle": 58,  # Left jaw angle
                "l_mid_ext_face": 93,  # Left middle exterior face
                "l_int_eyebrow": 107,  # Interor left eyebrow
                "l_mid_jaw": 136,  # Middle left jaw
                "l_mid_bot_eyelid": 145,  # Left eye middle bottom eyelid
                "l_ext_mouth": 146,  # Left exterior mouth
                "l_top_eyelid": 159,  # Left eye middle top eyelid
                "l_eye_int": 173,  # Left eye interior
                "l_pupil": 468,  # Left pupil
                "r_eye_exterior": 359,  # Right eye exterior
                "r_temple": 251,  # Right temple
                "r_mid_chin": 262,  # Right middle chin
                "r_up_lip": 269,  # Right upper lip
                "r_ext_nostril": 331,  # Exterior right nostril
                "r_mid_cheek": 280,  # Middle right cheek
                "r_mid_eyebrow": 283,  # Middle right eyebrow
                "r_ext_eyebrow": 282,  # Right exterior eyebrow
                "r_ext_lips": 273,  # Exterior right lips
                "r_jaw_angle": 288,  # Right jaw angle
                "r_mid_ext_face": 323,  # Right middle exterior face
                "r_int_eyebrow": 336,  # Interor right eyebrow
                "r_mid_jaw": 365,  # Middle right jaw
                "r_mid_bot_eyelid": 374,  # Right eye middle bottom eyelid
                "r_ext_mouth": 287,  # Right exterior mouth
                "r_top_eyelid": 386,  # Right eye middle top eyelid
                "r_eye_int": 398,   # Right eye interior
                "r_pupil": 473      # Right pupil
            }
            face_points: list[NormalizedLandmark] = facemark_result.multi_face_landmarks[0].landmark
            nose_point_coord: tuple[float, float, float] | None = landmark_to_list(face_points[face_fields["m_nose_point"]])
            chin_coord: tuple[float, float, float] | None = landmark_to_list(face_points[face_fields["m_chin"]])
            self.m_face_position = nose_point_coord
            if nose_point_coord is None:
                nose_point_coord = (0.0, 0.0, 0.0)
            if chin_coord is None:
                chin_coord = (0.0, 0.0, 0.0)

            scale = get_dist_between_points(
                (nose_point_coord[0], nose_point_coord[1], nose_point_coord[2]),
                (chin_coord[0], chin_coord[1], chin_coord[2])
            ) or 1.0
            self.m_face_scale = (scale, scale, scale)

            for key, val in face_fields.items():
                tmp = landmark_to_list(face_points[val])
                if tmp is not None:
                    tmp = (
                            (tmp[0] - nose_point_coord[0]) / scale,
                            (tmp[1] - nose_point_coord[1]) / scale,
                            (tmp[2] - nose_point_coord[2]) / scale
                    )

                setattr(self, key, tmp)

            self.m_face_position = (
                self.m_face_position[0] - 0.5,
                self.m_face_position[1] - 0.5,
                self.m_face_position[2] - 0.5,
            )


        if bodymark_result is not None:
            body_points: NormalizedLandmarkList = bodymark_result.pose_landmarks
            body_fields: dict[str, int] = {
                "l_shoulder": mediapipe.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                "l_elbow": mediapipe.solutions.pose.PoseLandmark.LEFT_ELBOW,
                "l_hip": mediapipe.solutions.pose.PoseLandmark.LEFT_HIP,
                "l_knee": mediapipe.solutions.pose.PoseLandmark.LEFT_KNEE,
                "l_ankle": mediapipe.solutions.pose.PoseLandmark.LEFT_ANKLE,
                "l_body_wrist": mediapipe.solutions.pose.PoseLandmark.LEFT_WRIST,
                "r_shoulder": mediapipe.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                "r_elbow": mediapipe.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                "r_hip": mediapipe.solutions.pose.PoseLandmark.RIGHT_HIP,
                "r_knee": mediapipe.solutions.pose.PoseLandmark.RIGHT_KNEE,
                "r_ankle": mediapipe.solutions.pose.PoseLandmark.RIGHT_ANKLE,
                "r_body_wrist": mediapipe.solutions.pose.PoseLandmark.RIGHT_WRIST,
            }
            l_shoulder_coord: tuple[float, float, float] | None = landmark_to_list(
                body_points.landmark[body_fields["l_shoulder"]])
            r_shoulder_coord: tuple[float, float, float] | None = landmark_to_list(
                body_points.landmark[body_fields["r_shoulder"]])
            if l_shoulder_coord is None:
                l_shoulder_coord = (0.0, 0.0, 0.0)
            if r_shoulder_coord is None:
                r_shoulder_coord = (0.0, 0.0, 0.0)
            self.m_body_position = (
                (l_shoulder_coord[0] + r_shoulder_coord[0]) / 2,
                (l_shoulder_coord[1] + r_shoulder_coord[1]) / 2,
                (l_shoulder_coord[2] + r_shoulder_coord[2]) / 2
            )

            scale = get_dist_between_points(
                (l_shoulder_coord[0], l_shoulder_coord[1], l_shoulder_coord[2]),
                (r_shoulder_coord[0], r_shoulder_coord[1], r_shoulder_coord[2])
            ) or 1.0
            self.m_body_scale = (scale, scale, scale)

            for key, val in body_fields.items():
                tmp = landmark_to_list(
                    body_points.landmark[val])
                if tmp is not None:
                    tmp = (
                        (tmp[0] - self.m_body_position[0]) / scale,
                        (tmp[1] - self.m_body_position[1]) / scale,
                        (tmp[2] - self.m_body_position[2]) / scale
                    )

                setattr(self, key, tmp)
            self.m_body_position = (
                self.m_body_position[0] - 0.5,
                self.m_body_position[1] - 0.5,
                self.m_body_position[2] - 0.5,
            )

        return self

    def setPointTo(self, point_field_name: str, x: float, y: float, z: float) -> Self:
        return self.setField(point_field_name, (x, y, z))

    def setPointToZero(self, point_field_name: str) -> Self:
        return self.setPointTo(point_field_name, 0, 0, 0)

    def setPointToRandom(self, point: str) -> Self:
        if point in CACHE_HANDS_POSITION:
            return self.setPointTo(
                point, rand_fix_interval(1), rand_fix_interval(
                    1), rand_fix_interval(1)
            )
        # 0.15 is the max value I can find on hand landmark
        return self.setPointTo(point,
                               rand_fix_interval(0.15),
                               rand_fix_interval(0.15),
                               rand_fix_interval(0.15),
                               )

    def setAllPointsToZero(self) -> Self:
        for field_name in FIELDS:
            self.setPointToZero(field_name)
        return self

    def setAllPointsToRandom(self) -> Self:
        for field_name in FIELDS:
            self.setPointToRandom(field_name)
        return self

    def setNonePointsToZero(self) -> Self:
        for field_name in FIELDS:
            if getattr(self, field_name) is None:
                self.setPointToZero(field_name)
        return self

    def setNonePointsToRandom(self) -> Self:
        for field_name in FIELDS:
            if getattr(self, field_name) is None:
                self.setPointToRandom(field_name)
        return self

    def setNonePointsRandomlyToRandomOrZero(self, proba: float = 0.1) -> Self:
        # Filter fields where the attribute is None
        none_fields: list[str] = [
            field_name for field_name in FIELDS if getattr(self, field_name) is None
        ]

        for field_name in none_fields:
            if random.random() < proba:
                self.setPointToZero(field_name)
            else:
                self.setPointToRandom(field_name)

        return self

    def get1DArray(self, valid_fields: list[str] = FIELDS) -> list[float]:
        tmp = [
            coord
            for field_name in valid_fields
            for coord in (cast(list[float], self.getField(field_name) or [0, 0, 0]))
        ]
        # print(self, "\n")
        # print(tmp, "\n\n")
        return tmp

    def getPoints(self, valid_fields: list[str] = FIELDS) -> list[tuple[float, float, float] | None]:
        """Get the points in a list of list.

        Args:
            valid_fields (list[str], optional): Let you pick which fields should be returned. Defaults to None (All point affected).

        Returns:
            list[list[float] | None]: _description_
        """
        return [self.getField(field_name) for field_name in valid_fields]

    def toNumpy(self, valid_fields: list[str] = FIELDS) -> NDArray[np.float32]:
        return np.array(self.get1DArray(valid_fields), dtype=np.float32)

    def toTensor(
        self,
        valid_fields: list[str] = FIELDS,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.as_tensor(self.get1DArray(valid_fields), dtype=torch.float32).to(
            default_device(device)
        )

    def noise(
        self, range: float = 0.005, valid_fields: list[str] = FIELDS
    ) -> Self:
        """Will randomize the gesture points by doing `new_val = old_val + rand_val(-range, range)` to each selected point.

        Args:
            range (float, optional): Random value will be between -range and range. Defaults to 0.005.
            valid_fields (list[str], optional): Let you pick which fields should be randomized. Defaults to None (All point affected).

        Returns:
            DataSample: Return this class instance for chaining
        """
        for field_name in valid_fields:
            field_value: tuple[float, float,
                               float] | None = getattr(self, field_name)
            if field_value is not None:
                return self.setPointTo(field_name,
                                       field_value[0] +
                                       rand_fix_interval(range),
                                       field_value[1] +
                                       rand_fix_interval(range),
                                       field_value[2] +
                                       rand_fix_interval(range))
        return self

    def mirror(self, x: bool = True, y: bool = False, z: bool = False) -> Self:
        inv_x: int = -1 if x else 1
        inv_y: int = -1 if y else 1
        inv_z: int = -1 if z else 1
        for field_name in FIELDS:
            field_value: tuple[float, float,
                               float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            self.setPointTo(field_name,
                            field_value[0] * inv_x,
                            field_value[1] * inv_y,
                            field_value[2] * inv_z)

        # Mirroring the hand make the hand become the opposite hand
        # This if statement will swap the left hand and right hand data
        if (x + y + z) % 2 == 1:
            self.swapHands()
        return self

    def rotate(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        for field_name in valid_fields:
            field_value: tuple[float, float,
                               float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            field_value = rot_3d_x(field_value, x)
            field_value = rot_3d_y(field_value, y)
            field_value = rot_3d_z(field_value, z)
            setattr(self, field_name, field_value)
        return self

    def scale(
        self,
        x: float = 1,
        y: float = 1,
        z: float = 1,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        for field_name in valid_fields:
            field_value: tuple[float, float,
                               float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            self.setPointTo(field_name,
                            field_value[0] * x,
                            field_value[1] * y,
                            field_value[2] * z)
        return self

    def translate(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        for field_name in valid_fields:
            field_value: tuple[float, float,
                               float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            self.setPointTo(field_name,
                            field_value[0] + x,
                            field_value[1] + y,
                            field_value[2] + z)
        return self

    def swapHands(self) -> Self:
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

        self.r_hand_position, self.l_hand_position = (
            self.l_hand_position,
            self.r_hand_position,
        )
        self.r_hand_velocity, self.l_hand_velocity = (
            self.l_hand_velocity,
            self.r_hand_velocity,
        )

        return self

    def moveToOneSide(self, right_side: bool = True) -> Self:
        dest_side = "r_" if right_side else "l_"
        src_side = "l_" if right_side else "r_"

        for field_name in FIELDS:
            if field_name.startswith(src_side):
                src_side_val: tuple[float, float,
                                    float] | None = getattr(self, field_name)
                opposite_field_name = field_name.replace(
                    src_side, dest_side, 1)
                dest_side_value: tuple[float, float, float] | None = getattr(
                    self, field_name.replace(src_side, dest_side)
                )
                if dest_side_value is None:
                    if src_side_val is not None:
                        src_side_val = (-src_side_val[0],
                                        src_side_val[1], -src_side_val[2])
                    setattr(self, opposite_field_name, src_side_val)
                    setattr(self, field_name, None)
        return self

    def isFullNone(self) -> bool:
        for field_name in FIELDS:
            if getattr(self, field_name) is not None:
                return False
        return True
