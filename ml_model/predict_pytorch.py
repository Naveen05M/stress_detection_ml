import torch
import torch.nn as nn
import cv2
import numpy as np

IMG_SIZE = 48
EMOTION_LABELS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
STRESS_MAP = {
    'angry': 'High', 'disgusted': 'High', 'fearful': 'High',
    'happy': 'Low', 'neutral': 'Low',
    'sad': 'Medium', 'surprised': 'Medium'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    model = StressCNN().to(device)
    model.load_state_dict(torch.load('ml_model/stress_model_pytorch.pth', map_location=device))
    model.eval()
    return model


def predict_emotion(face_img, model):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape)==3 else face_img
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor = torch.FloatTensor(resized/255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    emotion = EMOTION_LABELS[np.argmax(probs)]
    return emotion, STRESS_MAP[emotion], float(np.max(probs))*100, {EMOTION_LABELS[i]: round(float(probs[i])*100,2) for i in range(7)}