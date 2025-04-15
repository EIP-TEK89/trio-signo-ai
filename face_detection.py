from datetime import datetime
import mediapipe as mp
import json
import time
import cv2
import os
import argparse

FPS = 30
JSON_DIR = "datasets/face_data"

IMPORTANT_LANDMARKS = set(

    # MID FACE SET
    list(range(1)) + # Middle upper lip
    list(range(1,2)) + # Middle nose point
    list(range(6, 7)) + # Middle Top nose
    list(range(9, 10)) + # Middle of eyebrows
    list(range(10, 11)) + # Middle forehead
    list(range(18, 19)) + # Top chin
    list(range(13, 15)) + # Bottom upper lip, Top lower lip
    list(range(16, 17)) + # Bottom lower lip
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
    list(range(93, 94)) + # Left middle exterior face
    list(range(107, 108)) + # Interor left eyebrow
    list(range(136, 137)) + # Middle left jaw
    list(range(145, 147)) + # Left eye middle bottom eyelid, Left exterior mouth
    list(range(159, 160)) + # Left eye middle top eyelid
    list(range(173, 174)) + # Left eye interior
    list(range(468, 469)) + # Left pupil

    # RIGHT FACE SET
    list(range(251, 252)) + # Right temple
    list(range(262, 263)) + # Right middle chin, right eye exterior
    list(range(269, 270)) + # Right upper lip
    list(range(273, 274)) + # Right exterior mouth
    list(range(280, 281)) + # Right middle cheek
    list(range(282, 284)) + # Right exterior eyebrow, right middle eyebrow
    list(range(287, 288)) + # Right mouth exterior
    list(range(288, 289)) + # Right jaw angle
    list(range(323, 324)) + # Right ear
    list(range(331, 332)) + # Right exterior nostril
    list(range(336, 337)) + # Right interior eyebrow
    list(range(359, 360)) + # Right exterior eye
    list(range(365, 366)) + # Middle right jaw
    list(range(374, 375)) + # Bottom right eye
    list(range(386, 387)) + # Top right eye
    list(range(398, 399)) + # Right eye interior
    list(range(473, 474)) # Right pupil
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, required=True, help='Label for the gesture')
args = parser.parse_args()

# Création du dossier de sortie
os.makedirs(JSON_DIR, exist_ok=True)

# Initialisation caméra et variables
cap = cv2.VideoCapture(0)
gestures = []
frame_interval = 1.0 / FPS

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    last_frame_time = time.time()

    while cap.isOpened():
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            continue
        last_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = frame.shape

                # Extraction des points importants uniquement
                points = {
                    f"face_{idx}": [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]
                    for idx, landmark in enumerate(face_landmarks.landmark)
                    if idx in IMPORTANT_LANDMARKS
                }

                gestures.append(points)

                # Dessin
                for pt in points.values():
                    cx, cy = int(pt[0] * width), int(pt[1] * height)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        cv2.imshow('Face recording', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()

# Sauvegarde JSON
if gestures:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_data = {
        "label": args.label,
        "gestures": gestures,
        "framerate": FPS,
        "mirrorable": False
    }
    json_path = os.path.join(JSON_DIR, f"face_points_{args.label}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"✅ JSON saved at {json_path}")
