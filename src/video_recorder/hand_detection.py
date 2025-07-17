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
from src.model_class.transformer_sign_recognizer import *

HAND_TRACKING_MODEL_PATH = "models/hand_tracking/google/hand_landmarker.task"


def load_hand_landmarker(num_hand: int, running_mode = vision.RunningMode.VIDEO) -> HandLandmarker:
    base_options = python.BaseOptions(
        model_asset_path=HAND_TRACKING_MODEL_PATH)
    # print("========>", vision.RunningMode.VIDEO, vision.RunningMode.IMAGE, running_mode, HAND_TRACKING_MODEL_PATH)
    options: HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=base_options,
                                                                  num_hands=num_hand,
                                                                  running_mode=running_mode)
    recognizer: HandLandmarker = vision.HandLandmarker.create_from_options(
        options)
    return recognizer

def track_hand(image: cv2.typing.MatLike, hand_tracker: HandLandmarker, time_stamp: int) -> tuple[HandLandmarkerResult, float]:
    start_time = time.time()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return hand_tracker.detect_for_video(mp_image, time_stamp), time.time() - start_time

def track_hand_image(image: cv2.typing.MatLike, hand_tracker: HandLandmarker) -> tuple[HandLandmarkerResult, float]:
    start_time = time.time()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return hand_tracker.detect(mp_image), time.time() - start_time

#TODO: remove this function, use the one from src.misc.draw_gestures
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


def recognize_sign(sample: DataSample, sign_recognition_model: SignRecognizerTransformer, valid_fields: list[str] = None) -> tuple[int, float]:
    start_time = time.time()
    out = sign_recognition_model.predict(sample.toTensor(
        sign_recognition_model.info.memory_frame, valid_fields))
    return out, time.time() - start_time
