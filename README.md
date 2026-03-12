# Face Emotion Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-blue)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning system for **facial emotion recognition** that detects faces in images and predicts emotional expressions using a **ResNet50 convolutional neural network**.

The project integrates **MTCNN face detection**, **ResNet50 emotion classification**, **MLflow experiment tracking**, and a **Gradio-based interactive interface**.

---

# Project Overview

This project builds an end-to-end **AI pipeline for emotion recognition from facial images**.

The system automatically:

1. Detects faces in images  
2. Extracts the facial region  
3. Preprocesses the image for the neural network  
4. Predicts the emotional expression  

Supported emotion classes:

- Surprise  
- Fear  
- Disgust  
- Happy  
- Sad  
- Anger  
- Neutral  

---

# Demo

You can run the interactive interface locally to test the model.

The application allows users to upload an image and instantly receive emotion predictions along with probability distributions.

---

# Dataset

The model is trained using the **RAF-DB (Real-world Affective Faces Database)**.

Dataset link:

https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

RAF-DB is a widely used dataset for facial expression recognition containing thousands of real-world images with diverse expressions, lighting conditions, and head poses.

---

# Model Architecture

The emotion recognition model uses **ResNet50**, a deep convolutional neural network architecture widely used for computer vision tasks.

Pipeline architecture:

Face Image  
↓  
Face Detection (MTCNN)  
↓  
Image Preprocessing  
↓  
ResNet50 Model  
↓  
Emotion Prediction

Key components:

Face Detection  
MTCNN is used to detect faces and extract the facial region before feeding the image to the model.

Preprocessing  
Faces are resized to **224 × 224 pixels** and preprocessed using the ResNet50 preprocessing pipeline.

Classification  
The processed image is passed to the trained ResNet50 model which outputs probabilities for each emotion class.

---

# Experiment Tracking with MLflow

Model training experiments are tracked using **MLflow**.

MLflow logs:

- model parameters  
- training metrics  
- experiment runs  
- model artifacts  
- model versioning  

The model is stored in the **MLflow Model Registry** and deployed using alias-based versioning.

Example:

```
resnet50@production
```

This allows the production model to be updated without modifying inference code.

---

# Features

- Deep learning-based emotion recognition
- Automatic face detection using MTCNN
- ResNet50 CNN architecture
- MLflow experiment tracking and model registry
- Gradio interactive web interface
- Real-time webcam emotion detection
- Modular training experiments

---

# Project Structure

```
project
│
├── app.py
├── webcam_app.py
├── requirements.txt
├── README.md
│
├── training_notebooks
│   ├── resnet50-model-class-weights.ipynb
│   ├── resnet50-model-cutmix-mixup.ipynb
│   ├── convnext-tiny-model-mlflow.ipynb
│   ├── efficient-net-v2-s-class-weights.ipynb
│   └── efficient-net-v2-s-mixup-cutmix.ipynb
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Running the Application

Run the Gradio application

```
python app.py
```

The web interface will open in your browser where you can upload images and receive emotion predictions.

---

# Real-Time Webcam Emotion Detection

To run the webcam-based emotion recognition system:

```
python webcam_app.py
```

The system will detect faces in real time and display predicted emotions on the video stream.

Press **Q** to exit.

---

# Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- MTCNN
- MLflow
- OpenCV
- Gradio
- NumPy
- Pandas

---

# Future Improvements

- Face alignment for improved prediction accuracy
- Temporal smoothing for video emotion recognition
- Multi-face emotion tracking
- Optimized inference pipeline
- Deployment using Docker and CI/CD pipelines

---
