import cv2
from mediapipe.tasks.python.vision.hand_landmarker import *
import os
from src.datasample import *

from src.run_model import load_hand_landmarker, track_hand, draw_land_marks

# Path to your video file
label = "j"
videos_dir = f"datasets/{label}/temp/"

# Open the video file

handland_marker: HandLandmarker = load_hand_landmarker(1)

for video_file in os.listdir(videos_dir):
    cap = cv2.VideoCapture(videos_dir + video_file)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.", video_file)
    else:
        data_sample: DataSample = DataSample(label, [])
        # Loop over the frames
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()


            # If a frame was returned (not end of video)
            if ret:
                # Process the frame here (e.g., display or save)
                result, _ = track_hand(frame, handland_marker)
                data_sample.pushfront_gesture_from_landmarks(result)
                frame = draw_land_marks(frame, result)
                cv2.imshow('Frame', frame)

                # Wait for a key press (1 ms), break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # End of video
                break
        data_sample.toJsonFile(f"datasets/{label}/{video_file}.json")

        # Release the video capture object and close display window
        cap.release()
        cv2.destroyAllWindows()
