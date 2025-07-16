# typings/mediapipe/tasks/python/vision/hand_landmarker.pyi

from typing import Any, List
from typings.mediapipe.tasks.python.components.containers import landmark as landmark_module
from typings.mediapipe.tasks.python.components.containers import category as category_module

class HandLandmarkerResult:
    handedness: List[List[category_module.Category]]
    hand_landmarks: List[List[landmark_module.NormalizedLandmark]]
    hand_world_landmarks: List[List[landmark_module.Landmark]]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

