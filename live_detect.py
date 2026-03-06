import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
from ultralytics import YOLO


PROC_WIDTH, PROC_HEIGHT = 320, 240 # Smaller = much faster
FRAME_SKIP = 3 
TARGET_FPS = 60


mask_model = load_model("mask_detector.h5")
face_model = YOLO("yolov8n-face.pt")

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
         
        self.stream.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


vs = WebcamStream(src=0).start()

cached_results = []
frame_count = 0

while True:
    frame = vs.read()
    if frame is None: break
    
    
    display_frame = frame.copy()
    h_orig, w_orig = frame.shape[:2]

    
    if frame_count % FRAME_SKIP == 0:
        
        proc_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
        scale_x, scale_y = w_orig / PROC_WIDTH, h_orig / PROC_HEIGHT
        
        results = face_model(proc_frame, stream=True, verbose=False, conf=0.5)
        
        new_results = []
        face_list = []
        temp_coords = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in boxes:
                          f_x1, f_y1, f_x2, f_y2 = map(int, [x1, y1, x2, y2])
                face = proc_frame[f_y1:f_y2, f_x1:f_x2]
                
                if face.size > 0:
                    face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))
                    face_list.append(face.astype("float32") / 255.0)
                    
                    temp_coords.append((int(x1*scale_x), int(y1*scale_y), 
                                        int(x2*scale_x), int(y2*scale_y)))

        if face_list:
            preds = mask_model.predict(np.array(face_list), verbose=0)
            for i, pred_val in enumerate(preds):
                is_no_mask = pred_val[0] > 0.5
                new_results.append({
                    "box": temp_coords[i],
                    "label": "No Mask" if is_no_mask else "Mask",
                    "color": (0, 0, 255) if is_no_mask else (0, 255, 0)
                })
        cached_results = new_results

    
    for res in cached_results:
        x1, y1, x2, y2 = res["box"]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), res["color"], 2)
        cv2.putText(display_frame, res["label"], (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, res["color"], 2)

    frame_count += 1
    cv2.imshow("60 FPS Turbo Detect", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
