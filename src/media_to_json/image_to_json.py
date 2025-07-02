import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
import os
from src.datasample import *
from src.gesture import HandLandmarkerResult, BodyLandmarkResult, FaceLandmarkResult
from src.video_recorder.face_detection import track_face
from src.video_recorder.body_detection import track_body
from src.video_recorder.cv_drawer import CVDrawer
from typing import Any
from src.draw_gestures import draw_gestures

from src.run_model import load_hand_landmarker, track_hand, draw_land_marks

# Open the video file

handland_marker: HandLandmarker = load_hand_landmarker(1)

def image_to_json(path: str, label: str,
                  hand_landmarker: HandLandmarker | None = None,
                  body_landmarker: bool = False,
                  face_landmarker: bool = False) -> DataSample | None:
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to load {path}")
        return None

    cv_drawer: CVDrawer = CVDrawer(0, 0)
    image_sample = DataSample(label, [])

    hand_result: HandLandmarkerResult | None = track_hand(image, handland_marker)[0] if hand_landmarker else None
    body_result: BodyLandmarkResult | None = track_body(image)[1] if body_landmarker else None
    face_result: FaceLandmarkResult | None = track_face(image)[1] if face_landmarker else None

    image_sample.insertGestureFromLandmarks(0, hand_result, face_result, body_result)

    cv_drawer.set_frame_dim(image.shape[1], image.shape[0])
    cv_drawer.update_frame(image)
    draw_gestures(image_sample.gestures[0], cv_drawer.draw_line,
                  cv_drawer.draw_point)
    cv2.imshow('Frame', image)
    cv2.waitKey(1)

    return image_sample
