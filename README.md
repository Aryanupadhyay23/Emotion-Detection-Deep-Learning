# Face Emotion Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-blue)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning system for **facial emotion recognition** that detects faces in images and predicts emotional expressions using a **ResNet50 convolutional neural network**.

The project integrates **MTCNN face detection**, **ResNet50 emotion classification**, **MLflow experiment tracking**, and a **Gradio-based interactive interface**.

---

# Live Demo

Try the deployed application here:

https://huggingface.co/spaces/Aryan2301/Face_Emotion_Recognition_System

Upload an image containing a face and the system will automatically detect the face and predict the emotion.

---

# Project Overview

This project builds an end-to-end **AI pipeline for emotion recognition from facial images**.

Pipeline:

Face Image  
↓  
Face Detection (MTCNN)  
↓  
Image Preprocessing  
↓  
ResNet50 Model  
↓  
Emotion Prediction

The system predicts the following **7 emotion classes**:

- Surprise  
- Fear  
- Disgust  
- Happy  
- Sad  
- Anger  
- Neutral  

---

# Dataset

The model was trained using the **RAF-DB (Real-world Affective Faces Database)**.

Dataset link:

https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

RAF-DB contains thousands of facial images collected from the internet with diverse:

- expressions
- lighting conditions
- head poses
- occlusions

It is widely used for **facial expression recognition research**.

---

# Experiments

Multiple deep learning architectures and training strategies were explored.

All experiments were **tracked using MLflow**, allowing systematic comparison of models.

### Experiment Notebooks

| Notebook | Description |
|--------|--------|
| `custom-cnn-model-mlflow.ipynb` | Baseline CNN architecture |
| `convnext-tiny-model-mlflow.ipynb` | ConvNeXt Tiny transfer learning experiment |
| `efficient-net-v2-s-class-weigths.ipynb` | EfficientNetV2-S with class weights |
| `efficient-net-v2-s-mixup-cutmix.ipynb` | EfficientNetV2-S with MixUp and CutMix |
| `resnet50-model-class-weigths.ipynb` | ResNet50 with class weights |
| `resnet50-model-cutmix-mixup.ipynb` | ResNet50 with MixUp and CutMix |

Each experiment logged:

- training metrics
- validation metrics
- hyperparameters
- model artifacts
- experiment runs

using **MLflow experiment tracking**.

---

# Best Model

The best performing model was:

**ResNet50 trained with MixUp + CutMix augmentation**

### Overall Performance

| Metric | Score |
|------|------|
| Train Accuracy | **0.9172** |
| Validation Accuracy | **0.8644** |
| Test Accuracy | **0.8673** |
| ROC-AUC (Macro) | **0.9690** |
| ROC-AUC (Micro) | **0.9805** |

---

### ROC-AUC per Emotion Class

| Emotion | ROC-AUC |
|------|------|
| Surprise | 0.9859 |
| Fear | 0.9297 |
| Disgust | 0.9540 |
| Happy | 0.9865 |
| Sad | 0.9780 |
| Anger | 0.9805 |
| Neutral | 0.9686 |

---

# Model Architecture

The deployed model uses **ResNet50 Transfer Learning**.

Architecture:

Input Image (224×224)  
↓  
ResNet50 Feature Extractor  
↓  
Global Average Pooling  
↓  
Dense Layers  
↓  
Softmax Output (7 emotions)

---

# Training Techniques Used

To improve generalization and performance:

- Transfer Learning
- MixUp Data Augmentation
- CutMix Data Augmentation
- Class Weight Balancing
- Early Stopping
- MLflow Experiment Tracking

---

# Experiment Tracking with MLflow

MLflow was used for:

- experiment tracking
- model versioning
- artifact storage
- metric logging
- reproducibility

The final production model was registered as:

```
resnet50@production
```

---

# Features

- Deep learning-based emotion recognition
- Automatic face detection using **MTCNN**
- Multiple model experiments
- **MLflow experiment tracking**
- Gradio interactive web interface
- Real-time webcam emotion detection
- Hugging Face deployment

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
├── notebooks
│   ├── custom-cnn-model-mlflow.ipynb
│   ├── convnext-tiny-model-mlflow.ipynb
│   ├── efficient-net-v2-s-class-weigths.ipynb
│   ├── efficient-net-v2-s-mixup-cutmix.ipynb
│   ├── resnet50-model-class-weigths.ipynb
│   └── resnet50-model-cutmix-mixup.ipynb
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

Run the Gradio interface

```
python app.py
```

Open the browser and upload an image to predict emotions.

---

# Real-Time Webcam Emotion Detection

Run:

```
python webcam_app.py
```

The system will detect faces and display predicted emotions on the video stream.

Press **Q** to exit.

---

# Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- EfficientNetV2
- ConvNeXt
- MTCNN
- MLflow
- OpenCV
- Gradio
- NumPy
- Pandas

---

# Future Improvements

- Face alignment
- Temporal smoothing for video predictions
- Multi-face emotion tracking
- Model quantization for faster inference
- Dockerized deployment

---

# Author

Aryan Upadhyay

AI / Machine Learning Developer focused on **Deep Learning, Computer Vision, and MLOps**.