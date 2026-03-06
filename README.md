OmniMask
Face Mask Detection using CNN + YOLOv8
📌 Project Overview

This project detects whether a person is wearing a face mask or not in real-time using a webcam.

It combines:

YOLOv8 Face Detection → to detect faces in the frame

CNN Mask Classifier → to classify whether the detected face has a mask or not

The system draws:

🟢 Green Bounding Box → Mask Detected

🔴 Red Bounding Box → No Mask

This project demonstrates practical skills in Computer Vision, Deep Learning, and Real-Time Video Processing.

🚀 Features

Real-time webcam mask detection

Multiple people detection in a single frame

Bounding boxes around detected faces

Color-coded predictions

Uses Deep Learning (CNN) for classification

Uses YOLOv8 for fast face detection

🛠️ Tech Stack

Python

TensorFlow / Keras

OpenCV

Ultralytics YOLOv8

NumPy

📂 Project Structure
mask-detection/
│
├── model.ipynb          # Train CNN mask classifier
├──Live_Detect.py
├── detect_mask.py    # Real-time webcam detection
├── mask_detector.h5        # Trained CNN model
├── yolov8n-face.pt         # YOLOv8 face detection model
├── dataset/
│   ├── with_mask/
│   └── without_mask/
└── README.md


The webcam will start and detect mask / no mask in real time.

Press Q to exit.

🧠 Model Details
Face Detection

YOLOv8n-face

Fast and accurate real-time face detection

Mask Classification

CNN model

Input size: 224 × 224

Output classes:

Mask

No Mask

📊 Workflow

Capture frame from webcam

Detect faces using YOLOv8

Crop detected face

Resize to 224×224

Pass to CNN model

Predict mask / no mask

Draw bounding box with label

📸 Example Output

🟢 Green Box → Mask

🔴 Red Box → No Mask

Works with multiple people simultaneously.

🔮 Future Improvements

Add face recognition

Deploy as Android app

Convert to web application

Improve accuracy with MobileNetV2

👨‍💻 Author

Raunak Raj

Aspiring Data Scientist | Machine Learning Developer

Currently working on Computer Vision and Deep Learning projects.

⭐ If you like this project, consider starring the repository!
