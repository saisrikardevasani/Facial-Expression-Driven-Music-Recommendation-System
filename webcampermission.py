import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")

# Haar cascade path
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    return np.array(image).reshape(1, 48, 48, 1) / 255.0

# Webcam initialization
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise RuntimeError("Error: Cannot access webcam")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        image = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        pred = model.predict(extract_features(image), verbose=0)
        prediction_label = labels[pred.argmax()]

        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(im, prediction_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    cv2.imshow("Output", im)
    if cv2.waitKey(27) & 0xFF == 27:  # Press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
