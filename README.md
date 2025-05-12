😄 Real-Time Facial Emotion Detection with Keras and OpenCV
This repository provides a simple and effective implementation of facial emotion recognition using a pre-trained Mini-XCEPTION model, OpenCV for face detection, and Keras for model inference.

🔍 Overview
The code performs the following steps:

Installs required dependencies (keras, opencv-python).

Downloads a pre-trained emotion recognition model (Mini-XCEPTION) trained on the FER-2013 dataset.

Detects faces in an uploaded image using Haar Cascade.

Predicts the emotion of each detected face.

Displays the image with bounding boxes and emotion labels.

🧠 Pre-trained Model
We use the Mini-XCEPTION model trained on the FER-2013 dataset. It expects a 64x64 grayscale input and outputs a prediction across 7 emotions:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

⚙️ Installation & Setup
Step 1: Install Required Libraries
bash
Copy
Edit
pip install -q keras opencv-python
Step 2: Download Pre-trained Emotion Model
wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5
🚀 How to Use (in Google Colab)
Upload your image file when prompted using the file uploader.

The script detects faces and predicts emotions using the model.

The resulting image will be displayed with annotations.

Full Script
import cv2
import numpy as np
from keras.models import load_model
from google.colab import files
from google.colab.patches import cv2_imshow

# Load pre-trained emotion detection model
model = load_model("emotion_model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Upload image
uploaded = files.upload()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Process uploaded image
for filename in uploaded.keys():
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(preds)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    cv2_imshow(img)
📌 Notes
This script is designed to run in Google Colab.

It assumes one or more faces may be present in a single image.

You can extend this script for real-time video using a webcam (cv2.VideoCapture) if needed.

📄 License
This project is licensed under the MIT License.
