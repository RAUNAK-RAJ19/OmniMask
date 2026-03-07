import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import deque

mask_model = load_model("mask_detector.h5")
face_model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FRAME_SKIP = 3
frame_count = 0

# prediction history buffer
pred_buffer = deque(maxlen=10)

last_boxes = []
last_labels = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:

        results = face_model(frame, verbose=False)

        boxes_temp = []
        labels_temp = []

        for result in results:

            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face,(224,224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                pred = mask_model.predict(face, verbose=0)[0][0]

                pred_buffer.append(pred)

                avg_pred = np.mean(pred_buffer)

                if avg_pred > 0.5:
                    label = "No Mask"
                    color = (0,0,255)
                else:
                    label = "Mask"
                    color = (0,255,0)

                boxes_temp.append((x1,y1,x2,y2,color))
                labels_temp.append(label)

        last_boxes = boxes_temp
        last_labels = labels_temp

    for (box,label) in zip(last_boxes,last_labels):

        x1,y1,x2,y2,color = box

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,color,2)

    cv2.imshow("Mask Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
