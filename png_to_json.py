import os
import cv2
import json
import argparse
import numpy as np
from datetime import datetime
from src.datasample import DataSample
from src.datasample import dataclass
from run_model import load_hand_landmarker, track_hand
from face_detection import track_face
from body_detection import track_body

parser = argparse.ArgumentParser(description="Process PNG images from a folder and extract landmarks.")
parser.add_argument("--folder", required=True, help="Folder containing PNG images.")
parser.add_argument("--label", required=True, help="Label to assign to the processed images.")
parser.add_argument("--model", required=True, help="Path to the sign recognition model (needed for structure).")
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

# Process all .png images in the folder
image_files = [f for f in os.listdir(args.folder) if f.lower().endswith(".png")]
print(f"Found {len(image_files)} PNG images to process.")

for img_file in image_files:
    full_path = os.path.join(args.folder, img_file)
    print(f"Processing {img_file}...")

    image = cv2.imread(full_path)
    if image is None:
        print(f"Failed to load {img_file}")
        continue

    image_sample = DataSample(args.label, [])
    
    hand_result, _ = track_hand(image, handland_marker)
    face_result = track_face(image)[1] if args.face else None
    body_result = track_body(image)[1] if args.body else None

    image_sample.insertGestureFromLandmarks(0, hand_result, face_result, body_result)
    
    """
    for elem in image_sample.gestures:
        print(elem)
        for f in dataclass.fields(elem):
            if f is None:
                json_filename = os.path.splitext(img_file)[0] + ".json"
                json_path = os.path.join(output_path, json_filename)
                image_sample.toJsonFile(json_path)

                update_entry = {"filename": img_file, "label": args.label}
                with open(label_json_path, 'r') as f:
                    label_data = json.load(f)
                label_data.append(update_entry)
                with open(label_json_path, 'w') as f:
                    json.dump(label_data, f, indent=4)
                    """
    json_filename = os.path.splitext(img_file)[0] + ".json"
    json_path = os.path.join(output_path, json_filename)
    image_sample.toJsonFile(json_path)

    update_entry = {"filename": img_file, "label": args.label}
    with open(label_json_path, 'r') as f:
        label_data = json.load(f)
    label_data.append(update_entry)
    with open(label_json_path, 'w') as f:
        json.dump(label_data, f, indent=4)

print("Processing complete.")
