import os
import cv2
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mtcnn import MTCNN


# CONFIG

MLFLOW_TRACKING_URI = "https://dagshub.com/Aryanupadhyay23/Emotion-Detection-Deep-Learning.mlflow"

MODEL_NAME = "resnet50"
MODEL_ALIAS = "production"

IMG_SIZE = 224

EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Anger",
    "Neutral"
]

DAGSHUB_USERNAME = "Aryanupadhyay23"
EXPERIMENT_NAME = "Emotion Detection"


# MLflow AUTH

def configure_mlflow():

    token = os.environ.get("DAGSHUB_TOKEN")

    if not token:
        raise ValueError("DAGSHUB_TOKEN not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# LOAD MODEL

def load_model():

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    model = mlflow.tensorflow.load_model(model_uri)

    return model


# PREPROCESS FACE

def preprocess_face(face):

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    face = face.astype("float32")

    face = tf.keras.applications.resnet50.preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    return face


# INITIALIZE

configure_mlflow()

model = load_model()

# MTCNN Face Detector
detector = MTCNN()

cap = cv2.VideoCapture(0)

print("Press q to exit")


# WEBCAM LOOP

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(rgb_frame)

    for result in results:

        x, y, w, h = result['box']

        x = max(0, x)
        y = max(0, y)

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        img = preprocess_face(face)

        preds = model(img, training=False).numpy()

        class_id = np.argmax(preds)

        emotion = EMOTIONS[class_id]

        confidence = preds[0][class_id]

        label = f"{emotion} ({confidence:.2f})"

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # draw label
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()