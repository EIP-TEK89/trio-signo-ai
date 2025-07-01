from datetime import datetime
import mediapipe as mp
import json
import cv2
import os
from typing import NamedTuple

IMPORTANT_LANDMARKS = set([
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
    mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
])

mp_pose = mp.solutions.pose
_pose_tracker = None

def get_pose_tracker():
    global _pose_tracker
    if _pose_tracker is None:
        _pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _pose_tracker

def extract_body_landmarks_from_frame(frame, pose_tracker):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results: NamedTuple = pose_tracker.process(image)
    # print(type(results))

    if not results.pose_landmarks:
        return None, None

    # print(type(results.pose_landmarks))
    landmarks = results.pose_landmarks.landmark
    points = {
        idx: {
            'x': landmarks[idx].x,
            'y': landmarks[idx].y,
            'z': landmarks[idx].z,
            'visibility': landmarks[idx].visibility
        } for idx in IMPORTANT_LANDMARKS if idx < len(landmarks)
    }
    return results, points

def track_body(frame):
    pose_tracker = get_pose_tracker()
    results, points = extract_body_landmarks_from_frame(frame, pose_tracker)

    if points is None:
        return frame, None

    # height, width, _ = frame.shape
    # for pt in points.values():
    #     cx, cy = int(pt['x'] * width), int(pt['y'] * height)
    #     cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    return frame, results

def adapt_landmarks_to_json(points):
    adapted_points = {
        str(idx): [pt['x'], pt['y'], pt['z']] for idx, pt in points.items()
    }
    return adapted_points

def save_body_points_to_json(points_data, output_dir, label, framerate=30, mirrorable=True, invalid=False):
    final_dir = os.path.join(output_dir, label)
    os.makedirs(final_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(final_dir, f'{label}_body_points_{timestamp}.json')

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
