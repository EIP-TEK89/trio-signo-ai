from typing import override
from mediapipe.tasks.python.components.containers.landmark import Landmark, NormalizedLandmark
#
# class Landmark:
#     def __init__(self, 
#                  x: float = 0.0, 
#                  y: float = 0.0,
#                  z: float = 0.0, 
#                  visibility: float = 0.0,
#                  presence: float = 0.0):
#         self.x: float = x
#         self.y: float = y
#         self.z: float = z
#         self.visibility: float = visibility
#         self.presence: float = presence
#
#     @override
#     def __repr__(self):
#         return f"Landmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility}, presence={self.presence})"
#
class LandmarkList:
    def __init__(self):
        self.landmarks: list[Landmark] = []

    def add_landmark(self, landmark: Landmark):
        self.landmarks.append(landmark)

    @override
    def __repr__(self):
        return f"LandmarkList(landmarks={self.landmarks})"

class LandmarkListCollection:
    def __init__(self):
        self.landmark_lists: list[LandmarkList] = []

    def add_landmark_list(self, landmark_list: LandmarkList):
        self.landmark_lists.append(landmark_list)

    @override
    def __repr__(self):
        return f"LandmarkListCollection(landmark_lists={self.landmark_lists})"

# class NormalizedLandmark:
#     def __init__(self,
#                  x: float = 0.0,
#                  y: float = 0.0,
#                  z: float = 0.0,
#                  visibility: float = 0.0,
#                  presence: float = 0.0):
#         self.x: float = x
#         self.y: float = y
#         self.z: float = z
#         self.visibility: float = visibility
#         self.presence: float = presence
#
#     @override
#     def __repr__(self):
#         return f"NormalizedLandmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility}, presence={self.presence})"

class NormalizedLandmarkList:
    def __init__(self):
        self.landmark: list[NormalizedLandmark] = []

    def add_normalized_landmark(self, normalized_landmark: NormalizedLandmark):
        self.landmark.append(normalized_landmark)

    @override
    def __repr__(self):
        return f"NormalizedLandmarkList(landmarks={self.landmark})"

class NormalizedLandmarkListCollection:
    def __init__(self):
        self.landmark_lists: list[NormalizedLandmarkList] = []

    def add_normalized_landmark_list(self, normalized_landmark_list: NormalizedLandmarkList):
        self.landmark_lists.append(normalized_landmark_list)

    @override
    def __repr__(self):
        return f"NormalizedLandmarkListCollection(landmark_lists={self.landmark_lists})"
