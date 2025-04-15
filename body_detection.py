import cv2
import mediapipe as mp
import json
import time
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Record pose landmarks and save them in JSON format.")
parser.add_argument("--label", type=str, default="undefined", help="Label for the recorded gesture.")
parser.add_argument("--output", type=str, default="datasets/body_data", help="Directory to save the JSON files.")
parser.add_argument("--fps", type=int, default=30, help="Frame rate for recording.")
parser.add_argument("--mirrorable", action="store_true", help="Whether the gesture is mirrorable.")
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

SELECTED_LANDMARKS = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

CUSTOM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

def draw_custom_landmarks(image, landmarks, shape):
    h, w, _ = shape
    for idx in SELECTED_LANDMARKS.values():
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 6, (0, 255, 0), -1)

    for connection in CUSTOM_CONNECTIONS:
        start_idx, end_idx = connection
        start = landmarks.landmark[start_idx]
        end = landmarks.landmark[end_idx]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

if not os.path.exists(args.output):
    os.makedirs(args.output)

pose_data = []
frame_interval = 1.0 / args.fps
last_save_time = time.time()

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            draw_custom_landmarks(image, results.pose_landmarks, image.shape)

            if current_time - last_save_time >= frame_interval:
                frame_landmarks = {}
                for name, idx in SELECTED_LANDMARKS.items():
                    lm = results.pose_landmarks.landmark[idx]
                    if lm.visibility < 0.5:
                        frame_landmarks[name] = None
                    else:
                        frame_landmarks[name] = [lm.x, lm.y, lm.z]

                pose_data.append(frame_landmarks)
                last_save_time = current_time

        cv2.imshow("Body Detection (ESC or close to quit)", image)

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Body Detection (ESC or close to quit)", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()

final_data = {
    "label": args.label,
    "gestures": pose_data,
    "fps": args.fps,
    "mirrorable": args.mirrorable
}
current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
filename = f"{args.output}/pose_data_{current_datetime}.json"

with open(filename, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"[✔] Saved landmark data to {filename}")
