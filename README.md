# 🧠 StressGuard AI — Stress Detection in IT Professionals
### Image Processing & Machine Learning System

> A full-stack Django web application that detects stress levels in IT professionals through facial expression analysis using Deep Convolutional Neural Networks (CNN), OpenCV face detection, and real-time webcam monitoring.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Training](#model-training)
7. [Running the Web App](#running-the-web-app)
8. [Features](#features)
9. [API Reference](#api-reference)
10. [Stress Level Mapping](#stress-level-mapping)
11. [Troubleshooting](#troubleshooting)

---

## System Overview

```
Face Image / Live Feed
        │
        ▼
┌───────────────────────┐
│  OpenCV Haar Cascade  │  ← Face Detection
│  Face Crop & Resize   │  ← Preprocessing
│  Grayscale + Normalize│  ← Feature Extraction
└───────────┬───────────┘
            │  48×48 Grayscale Input
            ▼
┌───────────────────────┐
│   Deep CNN (Keras)    │  ← 4 Conv Blocks
│   BatchNorm + Dropout │  ← Regularization
│   GlobalAvgPool       │  ← Feature Map
│   Dense + Softmax     │  ← 7-class output
└───────────┬───────────┘
            │
            ▼
  7 Emotions → Stress Level
  angry/fearful/disgusted → 🔴 High
  sad/surprised           → 🟡 Medium
  happy/neutral           → 🟢 Low
            │
            ▼
  Django Dashboard + DB + Charts
```

---

## Project Structure

```
stress_detection/
├── manage.py                        # Django management
├── setup.py                         # One-click setup script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── stress_project/                  # Django project config
│   ├── settings.py                  # App settings
│   ├── urls.py                      # Root URL config
│   └── wsgi.py
│
├── stress_app/                      # Main Django app
│   ├── models.py                    # DB models (StressRecord, ModelMetrics)
│   ├── views.py                     # All views (dashboard, upload, live, history)
│   ├── urls.py                      # App URL patterns
│   ├── admin.py                     # Admin registration
│   ├── templates/stress_app/
│   │   ├── base.html                # Base layout with sidebar
│   │   ├── dashboard.html           # Main dashboard + charts
│   │   ├── upload.html              # Image upload page
│   │   ├── result.html              # Detection result page
│   │   ├── live.html                # Live camera page
│   │   └── history.html             # Records history
│   └── static/                      # CSS, JS, images
│
├── ml_model/
│   ├── train_model.py               # CNN training pipeline
│   ├── predict.py                   # Inference engine
│   ├── stress_model.h5              # ← Generated after training
│   ├── training_history.png         # ← Generated after training
│   ├── confusion_matrix.png         # ← Generated after training
│   └── metrics.json                 # ← Generated after training
│
├── dataset_utils/
│   └── prepare_dataset.py           # FER2013 CSV → folders
│
└── media/uploads/                   # Uploaded images
```

---

## Tech Stack

| Component        | Technology |
|-----------------|------------|
| Language         | Python 3.12 |
| Environment      | Anaconda |
| Backend          | Django 4.2 |
| Database         | SQLite (dev) / MySQL (prod) |
| Deep Learning    | TensorFlow 2.x + Keras |
| Face Detection   | OpenCV Haar Cascade |
| Image Processing | OpenCV, NumPy |
| Data Viz         | Matplotlib, Seaborn, Chart.js |
| Frontend         | HTML5, CSS3, JavaScript |
| Dataset          | FER2013 / RAF-DB (48×48 grayscale) |

---

## Step-by-Step Deployment

### ✅ STEP 1 — Install Anaconda

Download from: https://www.anaconda.com/products/distribution

```bash
# Verify installation
conda --version
```

---

### ✅ STEP 2 — Create Conda Environment

```bash
conda create -n stress_env python=3.12 -y
conda activate stress_env
```

---

### ✅ STEP 3 — Install Dependencies

```bash
# Core packages via conda
conda install -c conda-forge cmake -y
conda install -c conda-forge dlib -y

# Python packages via pip
pip install django==4.2
pip install tensorflow>=2.13
pip install opencv-python>=4.8
pip install scikit-learn matplotlib seaborn
pip install Pillow numpy
pip install mediapipe

# Optional: MySQL support
pip install mysqlclient
```

**Or install all at once:**
```bash
pip install -r requirements.txt
```

> ⚠️ **dlib installation note:** If pip install fails, use:
> `conda install -c conda-forge dlib`

---

### ✅ STEP 4 — Navigate to Project

```bash
cd stress_detection/
```

---

### ✅ STEP 5 — Run Setup Script

```bash
python setup.py
```

This will:
- Create and apply database migrations
- Create an admin superuser (admin / admin123)
- Create required media directories

---

### ✅ STEP 6 — Prepare Your Dataset

Your dataset should have this structure (48×48 grayscale images):

**Option A: Folder Structure (ready to use)**
```
dataset/
├── train/
│   ├── angry/      *.png / *.jpg
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    ├── angry/
    └── ...
```

**Option B: FER2013 CSV format**
```bash
# Download FER2013 from Kaggle:
# https://www.kaggle.com/datasets/msambare/fer2013

# Convert CSV to folder structure:
python dataset_utils/prepare_dataset.py \
    --csv fer2013.csv \
    --output dataset/

# Verify your dataset:
python dataset_utils/prepare_dataset.py --verify dataset/
```

---

### ✅ STEP 7 — Train the Model

```bash
python ml_model/train_model.py --dataset_path dataset/
```

**Expected output:**
```
============================================================
  Stress Detection – Deep CNN Training Pipeline
============================================================

[1] Loading dataset...
  Loaded 35887 images
  Class distribution: {angry: 4953, disgusted: 547, ...}

[2] Train size: 28709, Test size: 7178

[3] Building Deep CNN model...
Model: "StressDetectionCNN"
...

[4] Training...
Epoch 1/60: accuracy: 0.25 - val_accuracy: 0.30
...
Epoch 45/60: accuracy: 0.68 - val_accuracy: 0.64

[5] Evaluating on test set...
  Test Accuracy : 0.6412 (64.12%)

[6] Saving plots...
  Training history saved to ml_model/training_history.png
  Confusion matrix saved to ml_model/confusion_matrix.png

[7] Model saved to: ml_model/stress_model.h5
```

> ⏱️ Training takes ~30–90 minutes depending on hardware.
> GPU recommended: `pip install tensorflow[and-cuda]`

---

### ✅ STEP 8 — Configure Database (Optional: MySQL)

**SQLite (default, no config needed)**

**MySQL setup:**
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE stress_detection_db;
EXIT;
```

Edit `stress_project/settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'stress_detection_db',
        'USER': 'root',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

Then re-run migrations:
```bash
python manage.py migrate
```

---

### ✅ STEP 9 — Run the Development Server

```bash
python manage.py runserver
```

---

### ✅ STEP 10 — Open the Application

| URL | Page |
|-----|------|
| `http://127.0.0.1:8000/` | Dashboard |
| `http://127.0.0.1:8000/upload/` | Image Upload |
| `http://127.0.0.1:8000/live/` | Live Camera |
| `http://127.0.0.1:8000/history/` | History |
| `http://127.0.0.1:8000/admin/` | Admin Panel |

**Admin credentials:** `admin` / `admin123`

---

## Features

### 📊 Dashboard
- Total detection count with stress level breakdown
- Real-time line chart of stress over time
- Emotion distribution doughnut chart
- Quick action buttons for image upload and live camera
- Model performance metrics (accuracy, recall, F1)

### 🖼️ Image Upload Detection
- Drag & drop or file browse upload
- Real-time preview before submission
- Multi-face detection support
- Annotated result image with bounding boxes
- Emotion probability radar chart
- Auto-save to database

### 📷 Live Camera Detection
- Browser webcam access via JavaScript MediaDevices API
- Frames sent to Django backend every 1.5 seconds
- Annotated live feed overlay
- Real-time stress timeline chart
- Session statistics (frames analyzed, stress counts)

### 📋 History
- Full record table with filtering by stress level and emotion
- Delete individual records
- Timestamp, confidence, emotion breakdown

---

## CNN Architecture

```
Input: 48×48×1 (grayscale)
    │
    ▼
Block 1: Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout(0.25)
Block 2: Conv(128) → BN → Conv(128) → BN → MaxPool → Dropout(0.25)
Block 3: Conv(256) → BN → Conv(256) → BN → MaxPool → Dropout(0.25)
Block 4: Conv(512) → BN → MaxPool → Dropout(0.25)
    │
    ▼
GlobalAveragePooling2D
Dense(512, relu) → BN → Dropout(0.5)
Dense(256, relu) → Dropout(0.4)
Dense(7, softmax)
    │
    ▼
Output: 7 emotion probabilities
```

**Data Augmentation (training only):**
- Rotation: ±20°
- Width/height shift: ±15%
- Horizontal flip
- Zoom: ±15%
- Shear: ±10%

---

## Stress Level Mapping

| Emotion | Stress Level | Rationale |
|---------|-------------|-----------|
| Angry | 🔴 High | Active negative arousal |
| Disgusted | 🔴 High | Negative emotional state |
| Fearful | 🔴 High | Anxiety/threat response |
| Sad | 🟡 Medium | Passive negative affect |
| Surprised | 🟡 Medium | Moderate arousal |
| Happy | 🟢 Low | Positive emotional state |
| Neutral | 🟢 Low | Baseline / relaxed |

---

## API Reference

### POST `/live/frame/`
Send a webcam frame for real-time analysis.

**Request:**
```json
{
    "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
    "annotated_frame": "data:image/jpeg;base64,...",
    "results": [
        {
            "bbox": [100, 80, 150, 150],
            "emotion": "neutral",
            "stress_level": "Low",
            "confidence": 84.3,
            "scores": {"angry": 2.1, "happy": 8.3, "neutral": 84.3, ...},
            "color": "#3182ce"
        }
    ],
    "face_count": 1
}
```

### GET `/api/timeline/?hours=24`
Fetch stress timeline data for charts.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python ml_model/train_model.py --dataset_path dataset/` |
| `No face detected` | Use a clear, well-lit frontal face photo |
| `dlib install fails` | Use `conda install -c conda-forge dlib` |
| `Camera not working` | Allow browser camera permissions; use HTTPS in production |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `train_model.py` (e.g., 32) |
| `Low accuracy` | Use more data, increase EPOCHS, enable GPU |

---

## Training Tips for Better Accuracy

1. **More data:** FER2013 has ~35k images; RAF-DB has ~29k with cleaner labels
2. **Class imbalance:** FER2013 has very few "disgusted" samples — consider oversampling
3. **Transfer learning:** Load pre-trained VGGFace or ResNet weights for faster convergence
4. **GPU:** Training on GPU is ~10× faster — install `tensorflow[and-cuda]`
5. **Hyperparameters:** Try learning rate 1e-4, batch size 32

---

## License

MIT License — Free for academic and research use.

---

*Built with ❤️ for IT Professional Wellness Monitoring*
#