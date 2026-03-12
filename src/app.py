import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from PIL import Image
import pandas as pd
from mtcnn import MTCNN


# CONFIGURATION

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


# MLflow AUTHENTICATION

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


# INITIALIZE FACE DETECTOR

detector = MTCNN()


# PREPROCESS FACE

def preprocess_face(face):

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    face = face.astype("float32")

    face = tf.keras.applications.resnet50.preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    return face


# FACE DETECTION

def detect_face(image):

    img = np.array(image)

    results = detector.detect_faces(img)

    if len(results) == 0:
        return None, img

    # choose face with highest confidence
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    x, y, w, h = results[0]['box']

    x = max(0, x)
    y = max(0, y)

    face = img[y:y+h, x:x+w]

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    return face, img


# PREDICTION

def predict(image):

    if image is None:
        return "No image uploaded", 0, None, None

    face, boxed_img = detect_face(image)

    if face is None:
        return "No face detected", 0, None, boxed_img

    img = preprocess_face(face)

    preds = model(img, training=False).numpy()

    predicted_class = np.argmax(preds)

    emotion = EMOTIONS[predicted_class]

    confidence = float(preds[0][predicted_class])

    df = pd.DataFrame({
        "Emotion": EMOTIONS,
        "Probability": preds[0]
    })

    return emotion, confidence, df, boxed_img


# INITIALIZATION

configure_mlflow()

model = load_model()


# GRADIO UI

with gr.Blocks(theme=gr.themes.Soft(), title="Emotion Detection AI") as demo:

    gr.Markdown(
        """
        # Emotion Detection using Deep Learning
        Upload an image. The system will automatically detect the face and predict the emotion.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )

            predict_btn = gr.Button("Predict Emotion")

        with gr.Column():

            detected_image = gr.Image(label="Detected Face")

            emotion_output = gr.Textbox(label="Predicted Emotion")

            confidence_output = gr.Number(label="Confidence")

    prob_chart = gr.BarPlot(
        x="Emotion",
        y="Probability",
        title="Emotion Probabilities"
    )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            emotion_output,
            confidence_output,
            prob_chart,
            detected_image
        ]
    )


demo.launch()