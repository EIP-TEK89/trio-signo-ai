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



FRAME_ELAPSED: int = 0

def video_to_json(path: str, label: str, frame_elapsed: list[int],
                  hand_landmarker: HandLandmarker | None = None,
                  body_landmarker: bool = False,
                  face_landmarker: bool = False
                  ) -> DataSample | None:
    cap = cv2.VideoCapture(path)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video framerate: {fps} FPS")

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.", path)
        return None

    data_sample: DataSample = DataSample(label, [])
    # Loop over the frames

    cv_drawer: CVDrawer = CVDrawer(0, 0)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # If a frame was returned (not end of video)
        if ret:
            cv_drawer.set_frame_dim(cap.get(3), cap.get(4))
            cv_drawer.update_frame(frame)
            # Process the frame here (e.g., display or save)
            hand_result: HandLandmarkerResult | None = track_hand(frame, hand_landmarker, int((1000 / fps) * frame_elapsed[0]))[0] if hand_landmarker else None
            body_result: BodyLandmarkResult | None = track_body(frame)[1] if body_landmarker else None
            face_result: FaceLandmarkResult | None = track_face(frame)[1] if face_landmarker else None
            data_sample.insertGestureFromLandmarks(0, hand_result, face_result, body_result)
            draw_gestures(data_sample.gestures[0], cv_drawer.draw_line,
                                  cv_drawer.draw_point)
            cv2.imshow('Frame', frame)
            # Wait for a key press (1 ms), break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # End of video
            break
        frame_elapsed[0] += 1
    frame_elapsed[0] += fps * 10
    # Release the video capture object and close display window
    cap.release()
    return data_sample
