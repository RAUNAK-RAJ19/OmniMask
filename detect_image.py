import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO


try:
    mask_model = load_model("mask_detector.h5")
    face_model = YOLO("yolov8n-face.pt")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()


IMG_PATH = "test.jpg"
INPUT_SIZE = (224, 224)
CONF_THRESHOLD = 0.5  # Ignore weak detections
MIN_FACE_SIZE = 20    # Ignore tiny background noise

image = cv2.imread(IMG_PATH)
if image is None:
    print(f"Could not open {IMG_PATH}")
    exit()

output_img = image.copy()
h, w = image.shape[:2]


results = face_model(image, stream=True)

face_list = []
face_coords = []


for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for (x1, y1, x2, y2) in boxes:
        # Convert to int and clip to image boundaries
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        # Filtering: Skip if box is too small or invalid
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue

        face = image[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, INPUT_SIZE)
        face = face.astype("float32") / 255.0
        
        face_list.append(face)
        face_coords.append((x1, y1, x2, y2))


masked_count = 0
no_mask_count = 0

if len(face_list) > 0:
   
    faces_array = np.array(face_list)
    preds = mask_model.predict(faces_array, verbose=0)

    for i, pred_val in enumerate(preds):
        pred = pred_val[0]
        x1, y1, x2, y2 = face_coords[i]

        
        is_no_mask = pred > 0.5
        label = f"No Mask: {pred*100:.1f}%" if is_no_mask else f"Mask: {(1-pred)*100:.1f}%"
        color = (0, 0, 255) if is_no_mask else (0, 255, 0)
        
        if is_no_mask: no_mask_count += 1 
        else: masked_count += 1

        
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Create a semi-transparent HUD for the counters
overlay = output_img.copy()
cv2.rectangle(overlay, (0, 0), (220, 80), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)

cv2.putText(output_img, f"Masked: {masked_count}", (10, 30), 2, 0.7, (0, 255, 0), 2)
cv2.putText(output_img, f"No Mask: {no_mask_count}", (10, 60), 2, 0.7, (0, 0, 255), 2)

cv2.imshow("Optimized Mask Detection", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()