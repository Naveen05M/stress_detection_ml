"""
Stress Detection - Optimized Prediction Engine
Fast face detection + GPU inference + caching
"""
import cv2
import numpy as np
import base64
import torch
import torch.nn as nn
from pathlib import Path

# ── Constants ──────────────────────────────────────────
IMG_SIZE       = 48
EMOTION_LABELS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
STRESS_MAP     = {
    'angry':     ('High',   '#e53e3e'),
    'disgusted': ('High',   '#c53030'),
    'fearful':   ('High',   '#e67e22'),
    'happy':     ('Low',    '#38a169'),
    'neutral':   ('Low',    '#3182ce'),
    'sad':       ('Medium', '#d69e2e'),
    'surprised': ('Medium', '#805ad5'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Predict using: {device}')


# ── Model Architecture (Must match train_pytorch.py) ───
class StressCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(StressCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Optimized Predictor with Caching ──────────────────
class StressPredictor:
    _model        = None
    _face_cascade = None
    _eye_cascade  = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            model_path = Path(__file__).parent / 'stress_model_pytorch.pth'
            if not model_path.exists():
                raise FileNotFoundError(
                    f'Model not found at {model_path}. '
                    'Run: python ml_model/train_pytorch.py'
                )
            model = StressCNN().to(device)
            model.load_state_dict(
                torch.load(str(model_path), map_location=device)
            )
            model.eval()

            # ── Speed optimization: TorchScript compile ──
            try:
                dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
                with torch.no_grad():
                    model(dummy)  # Warmup run
                print('[INFO] Model warmup complete')
            except Exception:
                pass

            cls._model = model
            print(f'[INFO] Model loaded: {model_path}')
        return cls._model

    @classmethod
    def get_cascades(cls):
        if cls._face_cascade is None:
            base = cv2.data.haarcascades
            cls._face_cascade = cv2.CascadeClassifier(
                base + 'haarcascade_frontalface_default.xml'
            )
            cls._eye_cascade = cv2.CascadeClassifier(
                base + 'haarcascade_eye.xml'
            )
        return cls._face_cascade, cls._eye_cascade


# ── Fast Face Detection ────────────────────────────────
def detect_faces_fast(image_bgr):
    """Fast multi-scale face detection"""
    face_cascade, _ = StressPredictor.get_cascades()

    # Resize for faster detection
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if w > 640:
        scale   = 640 / w
        resized = cv2.resize(image_bgr, (640, int(h * scale)))
    else:
        resized = image_bgr

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Equalize histogram for better detection
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.05,   # Smaller = more thorough
        minNeighbors = 4,       # Lower = detect more faces
        minSize      = (25, 25),
        flags        = cv2.CASCADE_SCALE_IMAGE
    )

    # Scale back to original size
    if scale != 1.0 and len(faces) > 0:
        faces = [(
            int(x / scale), int(y / scale),
            int(w2 / scale), int(h2 / scale)
        ) for (x, y, w2, h2) in faces]

    return faces


# ── Batch Preprocessing ────────────────────────────────
def preprocess_faces_batch(image_bgr, faces):
    """Process all faces at once for GPU batch inference"""
    batch = []
    for (x, y, w, h) in faces:
        # Add padding around face
        pad  = int(min(w, h) * 0.1)
        x1   = max(0, x - pad)
        y1   = max(0, y - pad)
        x2   = min(image_bgr.shape[1], x + w + pad)
        y2   = min(image_bgr.shape[0], y + h + pad)

        face_roi = image_bgr[y1:y2, x1:x2]
        gray     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        norm     = resized.astype('float32') / 255.0
        batch.append(norm)

    if not batch:
        return None

    # Stack into batch tensor
    batch_arr    = np.array(batch)
    batch_tensor = torch.FloatTensor(batch_arr).unsqueeze(1).to(device)
    return batch_tensor


# ── Main Prediction Function ───────────────────────────
def predict_from_image_array(image_bgr):
    """Fast prediction for all faces in image"""
    model   = StressPredictor.get_model()
    faces   = detect_faces_fast(image_bgr)
    results = []

    if len(faces) == 0:
        return results

    # Batch process all faces at once
    batch_tensor = preprocess_faces_batch(image_bgr, faces)
    if batch_tensor is None:
        return results

    with torch.no_grad():
        outputs    = model(batch_tensor)
        probs_all  = torch.softmax(outputs, dim=1).cpu().numpy()

    for i, (x, y, w, h) in enumerate(faces):
        probs        = probs_all[i]
        emotion_idx  = int(np.argmax(probs))
        emotion      = EMOTION_LABELS[emotion_idx]
        confidence   = float(probs[emotion_idx]) * 100
        stress_level, color = STRESS_MAP[emotion]

        results.append({
            'bbox':         (int(x), int(y), int(w), int(h)),
            'emotion':      emotion,
            'stress_level': stress_level,
            'confidence':   round(confidence, 2),
            'scores': {
                EMOTION_LABELS[j]: round(float(probs[j]) * 100, 2)
                for j in range(7)
            },
            'color': color,
        })

    return results


# ── Annotate Image ─────────────────────────────────────
def annotate_image(image_bgr, results):
    annotated = image_bgr.copy()
    color_map = {
        'High':   (0,   0, 220),
        'Medium': (0, 165, 255),
        'Low':    (0, 200,   0)
    }
    for r in results:
        x, y, w, h = r['bbox']
        box_color  = color_map.get(r['stress_level'], (200, 200, 200))
        label      = (f"{r['emotion'].upper()} | "
                      f"{r['stress_level']} | "
                      f"{r['confidence']:.1f}%")

        # Draw filled rectangle background for text
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            annotated,
            (x, max(y - th - 12, 0)),
            (x + tw + 8, max(y, th + 12)),
            box_color, -1
        )
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(
            annotated, label,
            (x + 4, max(y - 4, th + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )
    return annotated


# ── Image to Base64 ────────────────────────────────────
def image_to_base64(image_bgr):
    _, buffer = cv2.imencode(
        '.jpg', image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    return base64.b64encode(buffer).decode('utf-8')


# ── Full Pipeline ──────────────────────────────────────
def predict_from_file(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Cannot read image: {image_path}')
    return predict_from_image_array(img), img


def predict_and_annotate(image_path):
    results, original = predict_from_file(image_path)
    annotated         = annotate_image(original, results)
    annotated_b64     = image_to_base64(annotated)
    return results, annotated_b64, len(results)