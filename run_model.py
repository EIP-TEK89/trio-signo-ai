# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run gesture recognition."""

import argparse
import sys
import os
import glob
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import *
from mediapipe.tasks.python.components.containers.category import *
from mediapipe.tasks.python.components.containers.landmark import *

from src.datasample import *
from src.model_class.sign_recognizer_v1 import *
from src.model_class.sign_recognizer_v2 import *

HAND_TRACKING_MODEL_PATH = "models/hand_tracking/google/hand_landmarker.task"

def load_sign_recognizer(model_dir: str) -> tuple[SignRecognizerV1, list[str]]:
    json_files = glob.glob(f"{model_dir}/*.json")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json file found in {model_dir}")
    pth_files = glob.glob(f"{model_dir}/*.pth")
    if len(pth_files) == 0:
        raise FileNotFoundError(f"No .pth file found in {model_dir}")
    with open(json_files[0], "r") as f:
        model_info: dict[str, list[str] | str] = json.load(f)
    match model_info["model_version"]:
        case "v1":
            tmp: ModelInfoV1 = ModelInfoV1(**model_info)
            model = SignRecognizerV1(len(tmp.labels))
            model.loadModel(pth_files[0])
        case "v2":
            tmp: ModelInfoV2 = ModelInfoV2(**model_info)
            model = SignRecognizerV2(len(tmp.labels))
            model.loadModel(pth_files[0])
        case _:
            raise ValueError(f"Model version {model_info['model_version']} not supported")
    print(model_info)
    return model, tmp.labels

def load_hand_landmarker(num_hand: int) -> HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=HAND_TRACKING_MODEL_PATH)
    options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options,
                                                                  num_hands=num_hand)
    recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(options)
    return recognizer

def track_hand(image: cv2.typing.MatLike, hand_tracker: HandLandmarker) -> tuple[HandLandmarkerResult, float]:
    start_time = time.time()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return hand_tracker.detect(mp_image), time.time() - start_time

def draw_land_marks(image: cv2.typing.MatLike, hand_landmarks: HandLandmarkerResult) -> cv2.typing.MatLike:
    img_cpy: cv2.typing.MatLike = image.copy()

    for hand_index, hand_landmarks in enumerate(hand_landmarks.hand_landmarks):

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Calculate the bounding box of the hand
        x_min = min([landmark.x for landmark in hand_landmarks])
        y_min = min([landmark.y for landmark in hand_landmarks])
        y_max = max([landmark.y for landmark in hand_landmarks])

        # Convert normalized coordinates to pixel values
        frame_height, frame_width = img_cpy.shape[:2]
        x_min_px = int(x_min * frame_width)
        y_min_px = int(y_min * frame_height)
        y_max_px = int(y_max * frame_height)

        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        mp_drawing.draw_landmarks(
          img_cpy,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    return img_cpy

def recognize_sign(hand_landmarks: HandLandmarkerResult, sign_recognition_model: SignRecognizerV1 | SignRecognizerV2) -> tuple[int, float]:
    start_time = time.time()
    if type(sign_recognition_model) == SignRecognizerV1:
        if len(hand_landmarks.hand_world_landmarks) != 0:
            return sign_recognition_model.use(LandmarksTo1DArray(hand_landmarks.hand_world_landmarks[0])), time.time() - start_time
    elif type(sign_recognition_model) == SignRecognizerV2:
        # sign_recognition_model.add_frame(hand_landmarks)
        # return sign_recognition_model.use(sign_recognition_model.input_data), time.time() - start_time
        if len(hand_landmarks.hand_world_landmarks) != 0:
            return sign_recognition_model.use(DataSample.from_handlandmarker(hand_landmarks, 0, 0).samples_to_1d_array()), time.time() - start_time

    return -1, time.time() - start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Folder where the model is (The folder should contain a .pth file for the model and a .json to get the info about the model).',
        required=True)
    parser.add_argument(
        '--numHands',
        help='Max number of hands that can be detected by the hand tracker.',
        required=False,
        default=2)

  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    args = parser.parse_args()

    print("Loading sign recognition model...")
    sign_rec_model = load_sign_recognizer(args.model)

    print("Loading hand landmarker...")
    hand_tracker: HandLandmarker = load_hand_landmarker(args.numHands)

    print("Initializing camera...")
    # Start capturing video input from the camera
    cap: cv2.VideoCapture = cv2.VideoCapture(args.cameraId)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    cap.set(cv2.CAP_PROP_FPS, 60)

    handtrack_times = []
    sign_rec_times = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        hand_landmarks: HandLandmarkerResult = None

        hand_landmarks, handtrack_time = track_hand(image, hand_tracker)
        handtrack_times.append(handtrack_time)
        if len(handtrack_times) > 10:
            handtrack_times.pop(0)
        handtrack_time = sum(handtrack_times) / len(handtrack_times)

        recognized_sign, sign_rec_time = recognize_sign(hand_landmarks, sign_rec_model[0])
        sign_rec_times.append(sign_rec_time)
        if len(sign_rec_times) > 10:
            sign_rec_times.pop(0)
        sign_rec_time = sum(sign_rec_times) / len(sign_rec_times)

        image = draw_land_marks(image, hand_landmarks)

        text = "undefined"
        if recognized_sign != -1:
            text = sign_rec_model[1][recognized_sign]
        cv2.putText(image, text, (49, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.01, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        print(f"\r\033[KTrack time: {(handtrack_time * 1000):.3f}ms Recognition time: {(sign_rec_time * 1000):.3f}ms Output: {text}", end=" ")
        cv2.imshow('Run model', image)


        if cv2.waitKey(1) == 27:
            break

    hand_tracker.close()
    cap.release()
    cv2.destroyAllWindows()
print()