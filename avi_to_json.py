import os
import cv2
import json
import argparse
from datetime import datetime
from src.datasample import DataSample
from run_model import load_hand_landmarker, track_hand
from face_detection import track_face
from body_detection import track_body

parser = argparse.ArgumentParser(description="Process AVI videos from a folder and extract landmarks.")
parser.add_argument("--folder", required=True, help="Folder containing AVI videos.")
parser.add_argument("--label", required=True, help="Label to assign to the processed videos.")
parser.add_argument("--face", action='store_true', help="Enable face tracking.")
parser.add_argument("--body", action='store_true', help="Enable body tracking.")
parser.add_argument("--counter-example", action='store_true', help="Save as counter example.")
args = parser.parse_args()

save_folder = "datasets/"
output_path = os.path.join(save_folder, args.label, "test", "counter_example" if args.counter_example else "")
os.makedirs(output_path, exist_ok=True)

# Load hand landmark model
print("Loading hand landmark model...")
handland_marker = load_hand_landmarker(2)

# Prepare label.json
temp_folder = os.path.join(save_folder, args.label, 'temp')
os.makedirs(temp_folder, exist_ok=True)
label_json_path = os.path.join(temp_folder, 'label.json')
if not os.path.exists(label_json_path):
    with open(label_json_path, 'w') as f:
        json.dump([], f)

# Process all .avi videos in the folder
video_files = [f for f in os.listdir(args.folder) if f.lower().endswith(".avi")]
print(f"Found {len(video_files)} AVI videos to process.")

for vid_file in video_files:
    full_path = os.path.join(args.folder, vid_file)
    print(f"Processing {vid_file}...")

    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        print(f"Failed to open {vid_file}")
        continue

    data_sample = DataSample(args.label, [])
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        hand_result, _ = track_hand(frame, handland_marker)
        face_result = track_face(frame)[1] if args.face else None
        body_result = track_body(frame)[1] if args.body else None

        data_sample.insertGestureFromLandmarks(0, hand_result, face_result, body_result)

    cap.release()

    json_filename = os.path.splitext(vid_file)[0] + ".json"
    json_path = os.path.join(output_path, json_filename)
    data_sample.toJsonFile(json_path)

    update_entry = {"filename": vid_file, "label": args.label}
    with open(label_json_path, 'r') as f:
        label_data = json.load(f)
    label_data.append(update_entry)
    with open(label_json_path, 'w') as f:
        json.dump(label_data, f, indent=4)

    print(f"Saved JSON: {json_filename} with {frame_count} frames.")

print("Video processing complete.")
