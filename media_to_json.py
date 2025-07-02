import os
import cv2
import json
import argparse
import numpy as np
import time
from datetime import datetime
from src.datasample import DataSample
from src.datasample import dataclass
from src.run_model import load_hand_landmarker, track_hand
from src.video_recorder.face_detection import track_face
from src.video_recorder.body_detection import track_body
from src.media_to_json.video_to_json import video_to_json
from src.media_to_json.image_to_json import image_to_json
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker



parser = argparse.ArgumentParser(description="Process PNG images from a folder and extract landmarks.")
parser.add_argument("--folder", required=True, help="Folder containing PNG images.")
parser.add_argument("--output", required=True, help="Output folder for processed images.")
# parser.add_argument("--label", required=True, help="Label to assign to the processed images.")
#parser.add_argument("--model", required=True, help="Path to the sign recognition model (needed for structure).")
parser.add_argument("--face", action='store_true', help="Enable face tracking.")
parser.add_argument("--body", action='store_true', help="Enable body tracking.")
parser.add_argument("--counter-example", action='store_true', help="Save as counter example.")
args = parser.parse_args()

src_folder: str = args.folder
save_folder: str = args.output
# output_path = os.path.join(save_folder, args.label, "test", "counter_example" if args.counter_example else "")
# os.makedirs(output_path, exist_ok=True)

# Load hand landmark model
print("Loading hand landmark model...")
hand_landmarker = load_hand_landmarker(2)

def convert_file_to_json(src_path: str, files: list[str], label: str, output_path: str,
                         hand_landmarker: HandLandmarker | None = None, body: bool = False, face: bool = False) -> None:
    """
    Convert files to JSON format and save them in the specified output path.
    """
    os.makedirs(output_path, exist_ok=True)
    for file in files:
        full_path = os.path.join(src_path, file)
        data_sample: DataSample | None = None
        print(f"Processing {file}...")
        if file.lower().endswith(".png") or file.lower().endswith(".jpg"):
            print(f"Converting image {file} to JSON...")
            data_sample = image_to_json(full_path, label, hand_landmarker, body, face)
        elif file.lower().endswith(".avi"):
            print(f"Converting video {file} to JSON...")
            data_sample = video_to_json(full_path, label, hand_landmarker, body, face)
        if data_sample:
            # json_filename: str = f"{label}_{round(time.time() * 1000000000)}.json"
            json_filename: str = f"{file.replace(".avi", "").replace(".png", "").replace(".jpg", "")}.json"
            json_path = os.path.join(output_path, json_filename)
            data_sample.toJsonFile(json_path)


for label_folder in os.listdir(src_folder):
    full_path: str = os.path.join(src_folder, label_folder)
    if not os.path.isdir(full_path):
        print(f"Skipping non-directory item: {label_folder}")
        continue
    print(f"Processing label folder: {full_path}")
    # Create output directory for the label

    for sub_folder in ["valid", "counter_examples"]:
        files: list[str] = []
        try:
            src_path: str = os.path.join(full_path, sub_folder)
            put_path: str = os.path.join(save_folder, label_folder, sub_folder)
            files: list[str] = os.listdir(src_path)

            print(f"Found {len(files)} files in {src_path} for label {label_folder}.")

            convert_file_to_json(src_path, files, label_folder, put_path,
                                 hand_landmarker=hand_landmarker,
                                 body=args.body, face=args.face)

            # ajouter la recup des images/video ici
        except FileNotFoundError:
            print(f"No {sub_folder} folder found for label: {label_folder}")

# # Prepare label.json
# temp_folder = os.path.join(save_folder, args.label, 'temp')
# os.makedirs(temp_folder, exist_ok=True)
# label_json_path = os.path.join(temp_folder, 'label.json')
# if not os.path.exists(label_json_path):
#     with open(label_json_path, 'w') as f:
#         json.dump([], f)

print("Processing complete.")
