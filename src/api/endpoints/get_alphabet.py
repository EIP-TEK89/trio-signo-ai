import os

import cv2
from flask import request, jsonify

import numpy as np
import io
from PIL import Image, ImageOps

from src.alphabet_recognizer import *

import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import *

def rescale_from_smallest_side(image, target_size):
    original_height, original_width = image.shape[:2]

    target_width = target_size  # Specify the desired width
    target_height = target_size  # Specify the desired height

    if original_width > original_height:
        target_height = int(original_height / original_width * target_width)
    else:
        target_width = int(original_width / original_height * target_height)

    return cv2.resize(image, (target_width, target_height))

def display_hand_tracked_image(image, recognition_result: HandLandmarkerResult):
    from mediapipe.framework.formats import landmark_pb2

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # print(recognition_result)

    current_frame = image
    for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):

        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('gesture_recognition', current_frame)
    cv2.waitKeyEx(1000)
    cv2.destroyAllWindows()

def get_alpahabet(hand_tracker: HandLandmarker, alphabet_recognizer: LSFAlphabetRecognizer):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the file into memory
        in_memory_file = io.BytesIO(file.read())
        pil_image = Image.open(in_memory_file)

        # pil_image.save("test.jpg")

        # Convert PIL image to OpenCV format
        pil_image = ImageOps.exif_transpose(pil_image)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        image = rescale_from_smallest_side(image, 480)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        recognition_result: HandLandmarkerResult = hand_tracker.detect(mp_image)

        # display_hand_tracked_image(image, recognition_result)

        if len(recognition_result.hand_world_landmarks) < 1:
            return jsonify({'message': None}), 200
        return jsonify({'message': LABEL_MAP.id[alphabet_recognizer.use(LandmarksTo1DArray(recognition_result.hand_world_landmarks[0]))]}), 200