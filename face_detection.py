from datetime import datetime
import mediapipe as mp
import json
import cv2
import os

from typing import NamedTuple
from types import GenericAlias
import mediapipe.python.solutions.face_mesh as fm
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions, FaceLandmarker, FaceLandmarkerResult
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *
from typings.mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

IMPORTANT_LANDMARKS: set[int] = set(

    # MID FACE SET
    list(range(1,2)) + # Middle nose point
    list(range(6, 7)) + # Middle Top nose
    list(range(9, 10)) + # Middle of eyebrows
    list(range(10, 11)) + # Middle forehead
    list(range(18, 19)) + # Top chin
    list(range(13, 15)) + # Bottom upper lip, Top lower lip
    list(range(141, 142)) + # Bottom nose
    list(range(152, 153)) + # Middle chin
    list(range(197, 198)) + # Middle nose

    # LEFT FACE SET
    list(range(7, 8)) + # Left eye exterior
    list(range(21, 22)) + # Left temple
    list(range(32, 33)) + # Left middle chin
    list(range(39, 40)) + # Left upper lip
    list(range(48, 49)) + # Exterior left nostril
    list(range(50, 51)) + # Middle left cheek
    list(range(52, 54)) + # Middle left eyebrow
    list(range(57, 59)) + # Exterior left lips, Left jaw angle
    # list(range(93, 94)) + # Left middle exterior face
    list(range(107, 108)) + # Interor left eyebrow
    list(range(136, 137)) + # Middle left jaw
    list(range(145, 147)) + # Left eye middle bottom eyelid, Left exterior mouth
    list(range(159, 160)) + # Left eye middle top eyelid
    list(range(173, 174)) + # Left eye interior
    list(range(468, 469)) + # Left pupil

    # RIGHT FACE SET
    list(range(251, 252)) + # Right temple
    list(range(262, 263)) + # Right middle chin
    list(range(269, 270)) + # Right upper lip
    list(range(273, 274)) + # Right exterior lips
    list(range(280, 281)) + # Right middle cheek
    list(range(282, 284)) + # Right exterior eyebrow, right middle eyebrow
    list(range(287, 288)) + # Right mouth exterior
    list(range(288, 289)) + # Right jaw angle
    list(range(323, 324)) + # Right mid exteriior face
    list(range(331, 332)) + # Right exterior nostril
    list(range(336, 337)) + # Right interior eyebrow
    list(range(359, 360)) + # Right exterior eye
    list(range(365, 366)) + # Middle right jaw
    list(range(374, 375)) + # Bottom right eye
    list(range(386, 387)) + # Top right eye
    list(range(398, 399)) + # Right eye interior
    list(range(473, 474)) # Right pupil
)

_face_mesh: fm.FaceMesh | None = None

def get_face_mesh() -> fm.FaceMesh:
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh

from src.gesture import DataGestures

def track_face(frame):
    face_mesh = get_face_mesh()
    results: NamedTuple = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # print(type(results), results, type(results["muti_face_landmarks"]))
    if not results.multi_face_landmarks:
        return frame, {}

    gest: DataGestures = DataGestures.buildFromLandmarkerResult(facemark_result=results)

    # print(gest)
    # print(IMPORTANT_LANDMARKS)
    face_landmarks: NormalizedLandmarkList = results.multi_face_landmarks[0]
    # print(dir(face_landmarks), vars(face_landmarks))
    height, width, _ = frame.shape
    # points = {}
    for idx, lm in enumerate(gest.getPoints()):
        # print(idx, lm)
        if lm is not None:
            # print("ah", len(IMPORTANT_LANDMARKS))
            # print(f"Landmark {type(idx)}: {lm.x}, {lm.y}, {lm.z}, {lm.presence}, {lm.visibility}")
            # points[str(idx)] = {'x': lm[0], 'y': lm[1], 'z': lm[2]}
            cx, cy = int(lm[0] * width), int(lm[1] * height)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    return frame, results

def adapt_landmarks_to_json(points: dict[str, dict[str, float]]):
    print(points)
    adapted_points = {
        str(idx): [pt['x'], pt['y'], pt['z']] for idx, pt in points.items()
    }
    return adapted_points

def save_face_points_to_json(points_data, output_dir, label, framerate=30, mirrorable=True, invalid=False):
    final_dir = os.path.join(output_dir, label)
    os.makedirs(final_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(final_dir, f'{label}_face_points_{timestamp}.json')

    gestures = []

    gestures.append(adapt_landmarks_to_json(points_data))

    data = {
        'label': label,
        'gestures': gestures,
        'framerate': framerate,
        'mirrorable': mirrorable,
        'invalid': invalid
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=0)

    print(f"✅ JSON saved to {json_path}")
