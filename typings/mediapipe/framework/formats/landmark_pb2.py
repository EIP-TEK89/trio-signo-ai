class Landmark:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

    def __repr__(self):
        return f"Landmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility}, presence={self.presence})"

class LandmarkList:
    def __init__(self):
        self.landmarks = []

    def add_landmark(self, landmark):
        self.landmarks.append(landmark)

    def __repr__(self):
        return f"LandmarkList(landmarks={self.landmarks})"

class LandmarkListCollection:
    def __init__(self):
        self.landmark_lists = []

    def add_landmark_list(self, landmark_list):
        self.landmark_lists.append(landmark_list)

    def __repr__(self):
        return f"LandmarkListCollection(landmark_lists={self.landmark_lists})"

class NormalizedLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

    def __repr__(self):
        return f"NormalizedLandmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility}, presence={self.presence})"

class NormalizedLandmarkList:
    def __init__(self):
        self.landmark = []

    def add_normalized_landmark(self, normalized_landmark):
        self.landmark.append(normalized_landmark)

    def __repr__(self):
        return f"NormalizedLandmarkList(landmarks={self.landmark})"

class NormalizedLandmarkListCollection:
    def __init__(self):
        self.landmark_lists = []

    def add_normalized_landmark_list(self, normalized_landmark_list):
        self.landmark_lists.append(normalized_landmark_list)

    def __repr__(self):
        return f"NormalizedLandmarkListCollection(landmark_lists={self.landmark_lists})"
