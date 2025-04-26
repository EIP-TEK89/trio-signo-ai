import random
import torch
import mediapipe
from dataclasses import dataclass, fields
from typing import Generic, TypeVar, Self, final, cast, Sequence, NamedTuple

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


def landmark_to_list(landmark: Landmark) -> list[float]:
    return [landmark.x, landmark.y, landmark.z]

class FaceLandmarkResult(NamedTuple):
    multi_face_landmarks: list[NormalizedLandmarkList]

class BodyLandmarkResult(NamedTuple):
    pose_landmarks: NormalizedLandmarkList

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
    r_ext_nostril: T | None = None # Exterior right nostril
    r_mid_cheek: T | None = None # Middle right cheek
    r_mid_eyebrow: T | None = None  # Middle right eyebrow
    r_ext_eyebrow: T | None = None  # Right exterior eyebrow
    r_ext_lips: T | None = None  # Exterior right lips
    r_jaw_angle: T | None = None  # Right jaw angle
    r_mid_ext_face: T | None = None # Right middle exterior face
    r_int_eyebrow: T | None = None # Interor right eyebrow
    r_mid_jaw: T | None = None # Middle right jaw
    r_mid_bot_eyelid: T | None = None # Right eye middle bottom eyelid
    r_ext_mouth: T | None = None  # Right exterior mouth
    r_top_eyelid: T | None = None # Right eye middle top eyelid
    r_eye_int: T | None = None # Right eye interior
    r_pupil: T | None = None # Right pupil

    l_shoudler: T | None = None  # Left shoulder
    l_elbow: T | None = None  # Left elbow
    l_hip: T | None = None  # Left hip
    l_knee: T | None = None  # Left knee
    l_ankle: T | None = None  # Left ankle

    r_shoudler: T | None = None  # Right shoulder
    r_elbow: T | None = None  # Right elbow
    r_hip: T | None = None  # Right hip
    r_knee: T | None = None  # Right knee
    r_ankle: T | None = None  # Right ankle


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

LEFT_BODY_POINTS : ActiveGestures = ActiveGestures(
    l_shoudler=True,
    l_elbow=True,
    l_hip=True,
    l_knee=True,
    l_ankle=True,
    l_wrist=True
)

RIGHT_BODY_POINTS : ActiveGestures = ActiveGestures(
    r_shoudler=True,
    r_elbow=True,
    r_hip=True,
    r_knee=True,
    r_ankle=True,
    r_wrist=True
)

BODY_POINTS : ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_BODY_POINTS, RIGHT_BODY_POINTS]
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

FACE_POINTS: ActiveGestures = ActiveGestures.buildWithPreset(
    [LEFT_FACE_POINTS, RIGHT_FACE_POINTS]
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
}

CACHE_HANDS_POINTS: list[str] = HANDS_POINTS.getActiveFields()
CACHE_HANDS_POSITION: list[str] = HANDS_POSITION.getActiveFields()


@final
@dataclass
class DataGestures(Gestures[list[float] | None]):
    @classmethod
    def buildFromLandmarkerResult(
        cls,
        landmark_result: HandLandmarkerResult | None = None,
        facemark_result: FaceLandmarkResult | None = None,
        bodymark_result: BodyLandmarkResult | None = None,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        tmp = cls()
        tmp.setHandsFromHandLandmarkerResult(landmark_result, facemark_result, bodymark_result, valid_fields)
        return tmp

    @classmethod
    def from1DArray(
        cls, array: list[float], valid_fields: list[str] = FIELDS
    ) -> Self:
        tmp = cls()
        for i, field_name in enumerate(valid_fields):
            setattr(
                tmp, field_name, array[i *
                                       FIELD_DIMENSION: (i + 1) * FIELD_DIMENSION]
            )
        return tmp

    def setHandsFromHandLandmarkerResult(
        self,
        landmark_result: HandLandmarkerResult | None = None,
        facemark_result: FaceLandmarkResult | None = None,
        bodymark_result: BodyLandmarkResult | None = None,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        """Convert the HandLandmarkerResult object into a DataGestures object.

        HandLandmarkerResult.hand_landmark represent the position of the hand in the image.
        HandLandmarkerResult.hand_world_landmarks represent a normalized hand that is not altered by the position or distance of the camera.

        Args:
            landmark_result (HandLandmarkerResult): _description_
        """

        if landmark_result is not None:
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
                "pinky_tip",
            ]

            for i in range(len(landmark_result.hand_world_landmarks)):
                handlandmark: list[NormalizedLandmark] = landmark_result.hand_landmarks[i]
                handworldlandmark: list[Landmark] = (
                    landmark_result.hand_world_landmarks[i]
                )
                prefix: str = "l_" if landmark_result.handedness[i][0].category_name == "Left" else "r_"

                """
                We use the wrist position to get the hand location
                then we substract 0.5 to center the hand since
                handlandmark elements store their position in a range of 0 to 1.
                Doing so will ease operation such as mirroring or rotation.
                """
                if valid_fields or f"{prefix}hand_position" in valid_fields:
                    self.r_hand_position = [
                        (handlandmark[0].x if handlandmark[0].x else 0) - 0.5,
                        (handlandmark[0].y if handlandmark[0].y else 0) - 0.5,
                        (handlandmark[0].z if handlandmark[0].z else 0) - 0.5,
                    ]

                # Adding position of each finger articulation
                for j, field_name in enumerate(hand_fields):
                    if f"{prefix}{field_name}" in valid_fields:
                        setattr(
                            self,
                            f"{prefix}{field_name}",
                            [
                                handworldlandmark[j].x,
                                handworldlandmark[j].y,
                                handworldlandmark[j].z,
                            ],
                        )

        if facemark_result is not None and len(facemark_result.multi_face_landmarks) > 0:
            face_points: NormalizedLandmarkList = facemark_result.multi_face_landmarks[0].landmark

            self.m_nose_point = landmark_to_list(face_points[1])
            self.m_top_nose = landmark_to_list(face_points[6])
            self.m_eyebrows = landmark_to_list(face_points[9])
            self.m_forehead = landmark_to_list(face_points[10])
            self.m_top_chin = landmark_to_list(face_points[18])
            self.m_bot_up_lip = landmark_to_list(face_points[13])
            self.m_top_low_lip = landmark_to_list(face_points[14])
            self.m_bot_nose = landmark_to_list(face_points[141])
            self.m_chin = landmark_to_list(face_points[152])
            self.m_nose = landmark_to_list(face_points[197])

            self.l_eye_exterior = landmark_to_list(face_points[7])
            self.l_temple = landmark_to_list(face_points[21])
            self.l_mid_chin = landmark_to_list(face_points[32])
            self.l_up_lip = landmark_to_list(face_points[39])
            self.l_ext_nostril = landmark_to_list(face_points[48])
            self.l_mid_cheek = landmark_to_list(face_points[50])
            self.l_mid_eyebrow = landmark_to_list(face_points[52])
            self.l_ext_eyebrow = landmark_to_list(face_points[53])
            self.l_ext_lips = landmark_to_list(face_points[57])
            self.l_jaw_angle = landmark_to_list(face_points[58])
            self.l_mid_ext_face = landmark_to_list(face_points[93])
            self.l_int_eyebrow = landmark_to_list(face_points[107])
            self.l_mid_jaw = landmark_to_list(face_points[136])
            self.l_mid_bot_eyelid = landmark_to_list(face_points[145])
            self.l_ext_mouth = landmark_to_list(face_points[146])
            self.l_top_eyelid = landmark_to_list(face_points[159])
            self.l_eye_int = landmark_to_list(face_points[173])
            self.l_pupil = landmark_to_list(face_points[468])

            self.r_eye_exterior = landmark_to_list(face_points[359])
            self.r_temple = landmark_to_list(face_points[251])
            self.r_mid_chin = landmark_to_list(face_points[262])
            self.r_up_lip = landmark_to_list(face_points[269])
            self.r_ext_nostril = landmark_to_list(face_points[331])
            self.r_mid_cheek = landmark_to_list(face_points[280])
            self.r_mid_eyebrow = landmark_to_list(face_points[283])
            self.r_ext_eyebrow = landmark_to_list(face_points[282])
            self.r_ext_lips = landmark_to_list(face_points[273])
            self.r_jaw_angle = landmark_to_list(face_points[288])
            self.r_mid_ext_face = landmark_to_list(face_points[323])
            self.r_int_eyebrow = landmark_to_list(face_points[336])
            self.r_mid_jaw = landmark_to_list(face_points[365])
            self.r_mid_bot_eyelid = landmark_to_list(face_points[374])
            self.r_ext_mouth = landmark_to_list(face_points[287])
            self.r_top_eyelid = landmark_to_list(face_points[386])
            self.r_eye_int = landmark_to_list(face_points[398])
            self.r_pupil = landmark_to_list(face_points[473])

        if bodymark_result is not None:
            body_points: NormalizedLandmarkList = bodymark_result.pose_landmarks.landmark

            self.l_shoudler = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_SHOULDER])
            self.l_elbow = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_ELBOW])
            self.l_hip = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_HIP])
            self.l_knee = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_KNEE])
            self.l_ankle = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_ANKLE])
            # self.r_wrist = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_WRIST])

            self.r_shoudler = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
            self.r_elbow = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_ELBOW])
            self.r_hip = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_HIP])
            self.r_knee = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_KNEE])
            self.r_ankle = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.RIGHT_ANKLE])
            # self.l_wrist = landmark_to_list(body_points[mediapipe.solutions.pose.PoseLandmark.LEFT_WRIST])

        return self

    def setPointTo(self, point_field_name: str, x: float, y: float, z: float) -> Self:
        setattr(self, point_field_name, [x, y, z])
        return self

    def setPointToZero(self, point_field_name: str) -> Self:
        return self.setPointTo(point_field_name, 0, 0, 0)

    def setPointToRandom(self, point: str) -> Self:
        if point in CACHE_HANDS_POSITION:
            return self.setPointTo(
                point, rand_fix_interval(1), rand_fix_interval(
                    1), rand_fix_interval(1)
            )
        # 0.15 is the max value I can find on hand landmark
        return self.setPointTo(
            point,
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
        none_fields = [
            field_name for field_name in FIELDS if getattr(self, field_name) is None
        ]

        for field_name in none_fields:
            if random.random() < proba:
                setattr(
                    self, field_name, [0, 0, 0]
                )  # Replace setPointToZero with direct set to 0
            else:
                self.setPointToRandom(field_name)

        return self

    def get1DArray(self, valid_fields: list[str] = FIELDS) -> list[float]:
        tmp = [
            coord
            for field_name in valid_fields
            for coord in (cast(list[float], getattr(self, field_name, [0, 0, 0])) or [0, 0, 0])
        ]
        # print(self, "\n")
        # print(tmp, "\n\n")
        return tmp

    def getPoints(self, valid_fields: list[str] = FIELDS) -> list[list[float] | None]:
        """Get the points in a list of list.

        Args:
            valid_fields (list[str], optional): Let you pick which fields should be returned. Defaults to None (All point affected).

        Returns:
            list[list[float] | None]: _description_
        """
        return [getattr(self, field_name) for field_name in valid_fields]

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
            DataSample2: Return this class instance for chaining
        """
        for field_name in valid_fields:
            field_value: list[float] | None = getattr(self, field_name)
            if field_value is not None:
                field_value[0] += rand_fix_interval(range)
                field_value[1] += rand_fix_interval(range)
                field_value[2] += rand_fix_interval(range)
                setattr(self, field_name, field_value)
        return self

    def mirror(self, x: bool = True, y: bool = False, z: bool = False) -> Self:
        for field_name in FIELDS:
            field_value: list[float] | None = getattr(self, field_name)
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

    def rotate(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        for field_name in valid_fields:
            field_value: list[float] | None = getattr(self, field_name)
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
            field_value: list[float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            field_value[0] *= x
            field_value[1] *= y
            field_value[2] *= z
            setattr(self, field_name, field_value)
        return self

    def translate(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        valid_fields: list[str] = FIELDS,
    ) -> Self:
        for field_name in valid_fields:
            field_value: list[float] | None = getattr(self, field_name)
            if field_value is None:
                continue
            field_value[0] += x
            field_value[1] += y
            field_value[2] += z
            setattr(self, field_name, field_value)
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
                src_side_val: list[float] | None = getattr(self, field_name)
                opposite_field_name = field_name.replace(
                    src_side, dest_side, 1)
                dest_side_value: list[float] | None = getattr(
                    self, field_name.replace(src_side, dest_side)
                )
                if dest_side_value is None:
                    if src_side_val is not None:
                        src_side_val[0] *= -1
                        # src_side_val[1] *= -1
                        src_side_val[2] *= -1
                    setattr(self, opposite_field_name, src_side_val)
                    setattr(self, field_name, None)
        return self
