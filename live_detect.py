import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time

# Load Models
mask_model = load_model("mask_detector.h5")
face_model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(0)

# --- Performance Tweaks ---
FRAME_SKIP = 3  # Only run AI every 3 frames
frame_count = 0
cached_results = [] # To store detections for skipped frames

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Downscale for processing speed (640x480 is plenty)
    proc_frame = cv2.resize(frame, (640, 480))
    h, w = proc_frame.shape[:2]
    
    # 2. Only run Inference on specific frames
    if frame_count % FRAME_SKIP == 0:
        cached_results = [] 
        results = face_model(proc_frame, stream=True, verbose=False, conf=0.5)
        
        face_list = []
        temp_coords = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in boxes:
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                face = proc_frame[y1:y2, x1:x2]
                
                if face.size > 0:
                    face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))
                    face_list.append(face.astype("float32") / 255.0)
                    temp_coords.append((x1, y1, x2, y2))

        if len(face_list) > 0:
            preds = mask_model.predict(np.array(face_list), verbose=0)
            for i, pred_val in enumerate(preds):
                is_no_mask = pred_val[0] > 0.5
                cached_results.append({
                    "box": temp_coords[i],
                    "label": "No Mask" if is_no_mask else "Mask",
                    "color": (0, 0, 255) if is_no_mask else (0, 255, 0)
                })

    # 3. Draw using cached results (keeps visuals smooth)
    for res in cached_results:
        x1, y1, x2, y2 = res["box"]
        cv2.rectangle(proc_frame, (x1, y1), (x2, y2), res["color"], 2)
        cv2.putText(proc_frame, res["label"], (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, res["color"], 2)

    frame_count += 1
    cv2.imshow("Turbo Mask Detect", proc_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()