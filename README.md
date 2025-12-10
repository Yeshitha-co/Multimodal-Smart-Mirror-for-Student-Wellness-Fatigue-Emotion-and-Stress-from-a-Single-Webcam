# Smart Mirror - Emotion, Fatigue & rPPG Detection

A real-time multimodal system that detects emotion, fatigue, and heart rate (rPPG) from video feeds using deep learning and computer vision. Built with Streamlit for an interactive interface and designed for both student wellness monitoring and driver/emotional state assessment.

## Project Overview

This project integrates three independent AI models for real-time analysis of:

•⁠  ⁠*Facial expressions* (emotion recognition)
•⁠  ⁠*Drowsiness / fatigue* (fatigue detection)
•⁠  ⁠*Heart rate* (rPPG-based BPM estimation)

A simple reinforcement-learning (RL) agent can optionally use these signals to suggest gentle interventions (e.g., breaks, hydration reminders).

## Project Structure


Smart Mirror/
├── app.py                              # Main Streamlit application integrating all models
├── Emotion/
│   ├── Emotion_Novel.ipynb             # Emotion model training on AffectNet YOLO dataset
│   ├── Finetuned_Novel_Emotion.ipynb   # Fine-tuning and optimization notebook
│   ├── best_emotion_model_b3_novel.pth # Trained emotion model weights
│   └── emotion_model_novel_finetuned.pth # Fine-tuned model weights
├── Fatigue/
│   ├── fatigue_detection_module.ipynb  # Fatigue detection training notebook
│   ├── fatigue_module.py               # Facial landmark extraction and feature computation
│   └── fatigue_mlp.joblib              # Trained MLP classifier for drowsiness
└── Stress/
    ├── rppg_bpm.ipynb                  # rPPG model training on UBFC-2 dataset
    └── best_model.pth                  # Trained heart rate estimation model


## Technical Stack

•⁠  ⁠*Framework*: Streamlit (web interface)
•⁠  ⁠*Deep Learning*: PyTorch, timm
•⁠  ⁠*Computer Vision*: OpenCV, MediaPipe (facial landmarks)
•⁠  ⁠*Data Processing*: NumPy
•⁠  ⁠*ML Models*: Scikit-learn (MLP classifier)

## Implemented Models

### 1. Emotion Detection Model

*Architecture*: NovelEmotionModel with EfficientNet-B3 Backbone

•⁠  ⁠*Backbone*: EfficientNet-B3 (pretrained, feature dimension: 1536)
•⁠  ⁠*Key Components*:
  - *MicroExpEnhancer*: Micro-expression enhancement module to amplify subtle facial changes (e.g., ⁠ x_out = x + factor × Conv2(ReLU(Conv1(x))) ⁠)
  - *DynamicEmotionRouting*: Two-expert gated network biased toward high-arousal and low-arousal emotions
  - *Multi-task Heads*: Emotion classifier (8 classes), Valence regressor, Arousal regressor

*Dataset*: AffectNet YOLO Format
•⁠  ⁠Training split: Training/validation split from AffectNet YOLO dataset
•⁠  ⁠8 emotion classes: Neutral, Happy, Sad, Angry, Surprise, Disgust, Fear, Tired
•⁠  ⁠Image size: 224×224
•⁠  ⁠Data augmentation: Horizontal flip, brightness/contrast adjustment, color jitter, motion blur, Gaussian noise, random shadows

*Training Details*:
•⁠  ⁠Base Model: ⁠ best_emotion_model_b3_novel.pth ⁠
•⁠  ⁠Fine-tuned Model: ⁠ emotion_model_novel_finetuned.pth ⁠
•⁠  ⁠Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
•⁠  ⁠Loss function: CrossEntropyLoss (with class weights)
•⁠  ⁠Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
•⁠  ⁠Batch size: 16
•⁠  ⁠Epochs: ~15 (with early stopping)
•⁠  ⁠Class weights: Inverse frequency weighting for imbalanced classes
•⁠  ⁠Normalization: Mean (0.5, 0.5, 0.5), Std (0.5, 0.5, 0.5)
•⁠  ⁠Mixed precision: Automatic mixed precision (AMP) enabled for GPU

### 2. Fatigue Detection Model

*Architecture*: MLP Classifier on Handcrafted Geometric Features

•⁠  ⁠*Face & Landmarks*: MediaPipe Face Mesh for 2D facial landmarks
•⁠  ⁠*Input Features to MLP (5-D)*:
  - Eye Aspect Ratio (EAR)
  - Mouth Aspect Ratio (MAR)
  - Yawn detection flag (binary)
  - Head tilt angle (degrees)
  - Pitch ratio (normalized head pitch)

*Dataset*: Driver Drowsiness Dataset (DDD)
•⁠  ⁠Classes: Drowsy, Non-drowsy
•⁠  ⁠Split: 70% train / 15% validation / 15% test
•⁠  ⁠Original image size: 224×224 (used only to run MediaPipe and compute features)
•⁠  ⁠Final training is done on tabular 5-D feature vectors, not raw pixels

*MLP Classifier*:
•⁠  ⁠Input: 5-D feature vector
•⁠  ⁠Hidden Layer 1: 32 units, ReLU
•⁠  ⁠Hidden Layer 2: 16 units, ReLU
•⁠  ⁠Output Layer: 2 logits (Drowsy vs Non-drowsy) + softmax
•⁠  ⁠Saved model: ⁠ fatigue_mlp.joblib ⁠ (scikit-learn / joblib export of the trained MLP)

*Training Details*:
•⁠  ⁠Loss function: Cross-entropy on (Drowsy, Non-drowsy) labels
•⁠  ⁠Optimizer: Adam
•⁠  ⁠Batch size: 64
•⁠  ⁠Epochs: ~10–20 with early stopping based on validation accuracy
•⁠  ⁠Reported test accuracy: ≈93% on held-out DDD test split

### 3. Heart Rate (rPPG) Detection Model

*Architecture*: UltraLightNet (Lightweight CNN)

•⁠  ⁠*Feature Extraction*:
  - Conv2d (3 → 16, kernel=5, stride=2)
  - MaxPool (2)
  - Conv2d (16 → 32, kernel=3)
  - MaxPool (3)
•⁠  ⁠*Temporal Processing*: Average pooling across the time dimension on a sliding window of ~180 frames (≈6 seconds at 30 FPS)
•⁠  ⁠*Regression Head*: Fully connected layers mapping flattened features (32×3×3 → 64 → 32 → 1 BPM value)
•⁠  ⁠*Model Size*: ~20K parameters (optimized for lightweight real-time inference)
•⁠  ⁠*Input Preprocessing*: Center crop (skipping face detection for speed), resized to 36×36

*Dataset*: UBFC-2
•⁠  ⁠Subject count: Multiple subjects with ground truth BPM
•⁠  ⁠Ground truth: BPM values from ground_truth.txt files
•⁠  ⁠BPM range: Validated to 40–200 BPM
•⁠  ⁠Frame sampling: 180 frames per video window (≈6 seconds at 30 FPS)
•⁠  ⁠Image size: 36×36 (ultra-small for lightweight inference)

*Training Details*:
•⁠  ⁠Optimizer: Adam (lr=1e-3)
•⁠  ⁠Loss function: Mean Squared Error (MSE)
•⁠  ⁠Validation metric: Mean Absolute Error (MAE)
•⁠  ⁠Epochs: up to ~100 (with early stopping)
•⁠  ⁠Batch size: 2
•⁠  ⁠Frame depth: 180 frames
•⁠  ⁠Video processing: No face detection, center crop only for speed

*Stress Score Calculation* (derived from BPM):
•⁠  ⁠Resting (60–80 BPM): Low stress (0–33%)
•⁠  ⁠Elevated (80–100 BPM): Moderate stress (33–83%)
•⁠  ⁠High (100+ BPM): High stress (83–100%)

These scores can be mapped to LOW / MODERATE / HIGH stress levels in the UI.

## Installation

### Prerequisites

•⁠  ⁠Python 3.8+
•⁠  ⁠CUDA 11.0+ (optional, for GPU acceleration)
•⁠  ⁠Webcam or video source

### Setup

1.⁠ ⁠Clone the repository:
   ⁠ bash
   git clone <your_repo_url>.git
   cd "Smart Mirror"
    ⁠

2.⁠ ⁠Install dependencies:
   ⁠ bash
   pip install streamlit torch torchvision timm opencv-python mediapipe numpy scikit-learn joblib
    ⁠

3.⁠ ⁠(Optional) For training and experimentation:
   ⁠ bash
   pip install jupyter albumentations grad-cam kagglehub kaggle
    ⁠

## Usage

### Run the Application

⁠ bash
streamlit run app.py
 ⁠

The application will open in your browser at ⁠ http://localhost:8501 ⁠.

You should see:
•⁠  ⁠Live video feed (from webcam)
•⁠  ⁠Emotion label and confidence
•⁠  ⁠Fatigue bar (drowsiness probability)
•⁠  ⁠Stress bar (from rPPG BPM)
•⁠  ⁠(Optionally) RL-related info about last suggested action

### Training Models

Training notebooks are available for reproducing results:

•⁠  ⁠*Emotion Model*: ⁠ Emotion/Emotion_Novel.ipynb ⁠ – trains on AffectNet YOLO dataset
•⁠  ⁠*Fatigue Model*: ⁠ Fatigue/fatigue_detection_module.ipynb ⁠ – trains on Driver Drowsiness Dataset (DDD)
•⁠  ⁠*rPPG Model*: ⁠ Stress/rppg_bpm.ipynb ⁠ – trains on UBFC-2 dataset

## Model Performance

### Emotion Model
•⁠  ⁠Backbone: EfficientNet-B3 with micro-expression enhancement and dynamic emotion routing
•⁠  ⁠Classes: 8 emotions
•⁠  ⁠Training: Class-weighted loss for handling imbalance

### Fatigue Model
•⁠  ⁠Binary classification: Drowsy (0) vs Non-drowsy (1)
•⁠  ⁠Reported performance: ≈93% accuracy on 15% held-out DDD test split
•⁠  ⁠5-D feature-based approach for lightweight inference

### rPPG Model
•⁠  ⁠Lightweight CNN architecture (~20K parameters) optimized for real-time performance
•⁠  ⁠Regression-based BPM prediction from small (36×36) frame crops
•⁠  ⁠Stress score mapping for interpretability in the UI

## Key Features

•⁠  ⁠*Multi-modal Analysis*: Simultaneously detects emotion, drowsiness, and heart rate from the same video stream
•⁠  ⁠*Real-time Processing*: Live video stream analysis with Streamlit
•⁠  ⁠*Local-Only Execution*: All inference runs locally; no cloud backend required
•⁠  ⁠*GPU Support*: CUDA acceleration available for faster emotion model inference
•⁠  ⁠*Transfer Learning*: Uses pretrained backbones (EfficientNet-B3, etc.)
•⁠  ⁠*Lightweight Models*: Optimized for edge deployment (rPPG model: ~20K parameters)
•⁠  ⁠*Micro-expression Enhancement*: Specialized module to capture subtle facial expressions
•⁠  ⁠*Dynamic Routing*: Expert-based architecture for emotion classification
•⁠  ⁠*RL-Based Interventions (Prototype)*: Simple RL agent to suggest breaks or hydration when fatigue/stress is high

## System Requirements

•⁠  ⁠*Webcam/Video Input*: Required for real-time analysis
•⁠  ⁠*RAM*: Minimum 4GB (8GB+ recommended)
•⁠  ⁠*GPU*: NVIDIA GPU with CUDA support (optional but recommended)
•⁠  ⁠*Python*: 3.8 or higher
•⁠  ⁠*OS*: Linux, macOS, or Windows

## Datasets Used

•⁠  ⁠*AffectNet YOLO Format* – For emotion detection training
  - 8 emotion classes
  - YOLO-formatted annotations
•⁠  ⁠*Driver Drowsiness Dataset (DDD)* – For fatigue detection
  - Drowsy vs Non-drowsy classification
  - Real-world driving scenarios
•⁠  ⁠*UBFC-2* – For heart rate (rPPG) estimation
  - Multiple subjects with ground truth BPM
  - Video-based heart rate measurement

---

*Last Updated*: December 2025
