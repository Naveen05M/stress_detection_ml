"""
============================================================
  Stress Detection - Prediction Engine
  Handles: Image files, live webcam feed
  Face detection via OpenCV Haar Cascade
  Emotion prediction via trained CNN
============================================================
"""
"""
Stress Detection - Prediction Engine (PyTorch)
"""
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
from pathlib import Path

# ── Constants ──────────────────────────────────────────
IMG_SIZE = 48
EMOTION_LABELS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
STRESS_MAP = {
    'angry':     ('High',   '#e53e3e'),
    'disgusted': ('High',   '#c53030'),
    'fearful':   ('High',   '#e67e22'),
    'happy':     ('Low',    '#38a169'),
    'neutral':   ('Low',    '#3182ce'),
    'sad':       ('Medium', '#d69e2e'),
    'surprised': ('Medium', '#805ad5'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Using device: {device}')


# ── Model Architecture ─────────────────────────────────
class StressCNN(nn.Module):
    def __init__(self):
        super(StressCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load Model ─────────────────────────────────────────
class StressPredictor:
    _model = None
    _face_cascade = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            model_path = Path(__file__).parent / 'stress_model_pytorch.pth'
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Please run: python ml_model/train_pytorch.py"
                )
            model = StressCNN().to(device)
            model.load_state_dict(
                torch.load(str(model_path), map_location=device)
            )
            model.eval()
            cls._model = model
            print(f'[INFO] Model loaded from {model_path}')
        return cls._model

    @classmethod
    def get_cascade(cls):
        if cls._face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cls._face_cascade = cv2.CascadeClassifier(cascade_path)
        return cls._face_cascade


# ── Preprocessing ──────────────────────────────────────
def preprocess_face(face_roi):
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


# ── Face Detection ─────────────────────────────────────
def detect_faces(image_bgr):
    cascade = StressPredictor.get_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces


# ── Prediction ─────────────────────────────────────────
def predict_from_image_array(image_bgr):
    model = StressPredictor.get_model()
    faces = detect_faces(image_bgr)
    results = []

    if len(faces) == 0:
        return results

    for (x, y, w, h) in faces:
        face_roi = image_bgr[y:y+h, x:x+w]
        tensor = preprocess_face(face_roi)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        emotion_idx = np.argmax(probs)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(probs[emotion_idx]) * 100
        stress_level, color = STRESS_MAP[emotion]

        results.append({
            'bbox': (int(x), int(y), int(w), int(h)),
            'emotion': emotion,
            'stress_level': stress_level,
            'confidence': round(confidence, 2),
            'scores': {
                EMOTION_LABELS[i]: round(float(probs[i])*100, 2)
                for i in range(len(EMOTION_LABELS))
            },
            'color': color,
        })

    return results


# ── Annotate Image ─────────────────────────────────────
def annotate_image(image_bgr, results):
    annotated = image_bgr.copy()
    for r in results:
        x, y, w, h = r['bbox']
        label = f"{r['emotion'].upper()} | {r['stress_level']} | {r['confidence']:.1f}%"
        color_map = {
            'High': (0, 0, 220),
            'Medium': (0, 165, 255),
            'Low': (0, 200, 0)
        }
        box_color = color_map.get(r['stress_level'], (200, 200, 200))
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(
            annotated, label,
            (x, max(y-10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
        )
    return annotated


# ── Image to Base64 ────────────────────────────────────
def image_to_base64(image_bgr):
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


# ── Full Pipeline ──────────────────────────────────────
def predict_from_file(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return predict_from_image_array(img), img


def predict_and_annotate(image_path):
    results, original = predict_from_file(image_path)
    annotated = annotate_image(original, results)
    annotated_b64 = image_to_base64(annotated)
    return results, annotated_b64, len(results)

