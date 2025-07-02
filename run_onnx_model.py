from src.model_class.transformer_sign_recognizer import *
from src.run_model import *
import cv2

sign_rec: SignRecognizerTransformerONNX = SignRecognizerTransformerONNX("onnx_models")

hand_landmarker: HandLandmarker = load_hand_landmarker(1)

def list_available_cameras(max_cameras=10):
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)  # Essaye d'ouvrir la caméra i
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()  # Libère la caméra après le test

    return available_cameras

available_cameras = list_available_cameras()

# Video setup
record = cv2.VideoCapture(available_cameras[0])
frame_width = int(record.get(3))
frame_height = int(record.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

frame_history: DataSample = DataSample("", [])
prev_sign = -1
prev_display = -1

while True:
    ret, frame = record.read()
    if not ret:
        print("Video error.")
        break

    og_frame = cv2.flip(frame, 1)
    frame = copy.deepcopy(og_frame)
    result, _ = track_hand(frame, hand_landmarker)
    frame = draw_land_marks(frame, result)

    frame_history.insertGestureFromLandmarks(0, result)
    while len(frame_history.gestures) > sign_rec.info.memory_frame:
        frame_history.gestures.pop(-1)
    if sign_rec.info.one_side:
        frame_history.move_to_one_side()

    recognized_sign = sign_rec.predict(frame_history.to_onnx_tensor(sign_rec.info.memory_frame, sign_rec.info.active_gestures.getActiveFields()))

    text = "undefined"

    if prev_sign != recognized_sign:
        prev_display = prev_sign
        prev_sign = recognized_sign
    if recognized_sign != -1:
        text = f"{sign_rec.info.labels[recognized_sign]} prev({sign_rec.info.labels[prev_display]})"
    cv2.putText(frame, text, (49, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.01, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("ONNX runner", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
