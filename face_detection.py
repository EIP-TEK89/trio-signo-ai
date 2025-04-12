from datetime import datetime
import mediapipe as mp
import json
import time
import cv2
import os

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


def save_points_to_json(points, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(output_dir, f'face_points_{timestamp}.json')

    with open(json_path, 'w') as json_file:
        json.dump(points, json_file, indent=4)

    print(f'✅ JSON file saved : {json_path}')


def main():
    os.makedirs(JSON_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    points_data = []
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
        frame_id = 0
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
                    points = {
                        idx: {'x': pt.x, 'y': pt.y, 'z': pt.z}
                        for idx, pt in enumerate(face_landmarks.landmark) if idx in IMPORTANT_LANDMARKS
                    }

                    points_data.append({
                        'frame_id': frame_id,
                        'points': points
                    })

                    # Draw points on screen
                    for idx, pt in points.items():
                        cx, cy = int(pt['x'] * width), int(pt['y'] * height)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                    frame_id += 1

            cv2.imshow('Face recording', frame)
            if cv2.waitKey(1) & 0xFF == 27: # esc key
                break

    cap.release()
    cv2.destroyAllWindows()

    if points_data:
        save_points_to_json(points_data, JSON_DIR)


if __name__ == '__main__':
    main()