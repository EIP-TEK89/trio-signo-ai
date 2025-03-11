import os
import ipaddress

import cv2
from flask import request, jsonify

import numpy as np
import io
from PIL import Image, ImageOps
from run_model import track_hand, recognize_sign

from src.model_class.transformer_sign_recognizer import *

import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import *

def get_alphabet2(alphabet_recognizer: SignRecognizerTransformer, sample_history: dict[int, DataSample2]):
    try:
        ip: int = ipaddress.ip_address(request.remote_addr)
    except:
        return jsonify({'error': 'Invalid IP address'}), 400

    gest: DataGestures = DataGestures.fromDict(request.get_json())

    if sample_history.get(ip) is None:
        sample_history[ip] = DataSample2("", [])
    sample_history[ip].gestures.insert(0, gest)
    while len(sample_history[ip].gestures) > alphabet_recognizer.info.memory_frame:
        sample_history[ip].gestures.pop(-1)
    sample_history[ip].computeHandVelocity()
    if alphabet_recognizer.info.one_side:
        sample_history[ip].move_to_one_side()
    sign, _ = recognize_sign(sample_history[ip], alphabet_recognizer, alphabet_recognizer.info.active_gestures.getActiveFields())
    # print(f"Sign: {alphabet_recognizer.info.labels[sign]}")
    return jsonify({'message': f"{alphabet_recognizer.info.labels[sign]}"}), 200
