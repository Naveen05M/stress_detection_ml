"""
============================================================
  Stress Detection - Maximum Accuracy Training
  FER2013 + RAF-DB Combined
  All Accuracy Methods:
  1. 120 Epochs + Early Stopping
  2. Heavy Data Augmentation
  3. Cosine Annealing LR
  4. Label Smoothing
  5. Weight Decay
  6. Gradient Clipping
  7. Class Weights for Imbalance
  8. Deeper Architecture
============================================================
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── GPU Setup ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Constants ──────────────────────────────────────────
IMG_SIZE     = 48
BATCH_SIZE   = 64
EPOCHS       = 120
LR           = 0.0003
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
NUM_CLASSES  = 7

EMOTION_LABELS = [
    'angry','disgusted','fearful',
    'happy','neutral','sad','surprised'
]

# ── Dataset with Heavy Augmentation ───────────────────
class FaceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [2])
            # Random brightness
            if torch.rand(1) > 0.4:
                x = x * (0.7 + torch.rand(1) * 0.6)
            # Random contrast
            if torch.rand(1) > 0.4:
                mean = x.mean()
                x = (x - mean) * (0.7 + torch.rand(1) * 0.6) + mean
            # Gaussian noise
            if torch.rand(1) > 0.5:
                x = x + torch.randn_like(x) * 0.04
            # Random erasing
            if torch.rand(1) > 0.6:
                h, w = x.shape[1], x.shape[2]
                eh = torch.randint(5, 18, (1,)).item()
                ew = torch.randint(5, 18, (1,)).item()
                sy = torch.randint(0, h - eh, (1,)).item()
                sx = torch.randint(0, w - ew, (1,)).item()
                x[:, sy:sy+eh, sx:sx+ew] = 0
            # Random shift
            if torch.rand(1) > 0.6:
                shift_h = torch.randint(-4, 4, (1,)).item()
                shift_w = torch.randint(-4, 4, (1,)).item()
                x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
            x = torch.clamp(x, 0, 1)
        return x, self.y[idx]


# ── Load Dataset ───────────────────────────────────────
def load_dataset(dataset_path):
    X, y = [], []
    dataset_path = Path(dataset_path)
    print('\n  Images per emotion:')

    for label_idx, label in enumerate(EMOTION_LABELS):
        loaded = 0
        for split in ['train', 'test', '']:
            path = dataset_path / split / label
            if not path.exists():
                continue
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_file in path.glob(ext):
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label_idx)
                    loaded += 1
        print(f'  {label:12s}: {loaded}')

    X = np.array(X, dtype='float32') / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    print(f'\n  Total: {len(X)} images')
    return X, y


# ── CNN Model (Improved Architecture) ─────────────────
class StressCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(StressCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 4
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

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Early Stopping ─────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter  = 0
        self.best_acc = 0
        self.stop     = False

    def __call__(self, val_acc, model, path):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
            torch.save(model.state_dict(), path)
            print(f'  ✅ Model saved! Best: {self.best_acc*100:.2f}%')
        else:
            self.counter += 1
            print(f'  No improvement {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.stop = True


# ── Plot History ───────────────────────────────────────
def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History - FER2013 + RAF-DB', fontsize=14)

    axes[0].plot(history['train_acc'], label='Train', color='#2196F3')
    axes[0].plot(history['val_acc'],   label='Val',   color='#FF5722')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(history['train_loss'], label='Train', color='#4CAF50')
    axes[1].plot(history['val_loss'],   label='Val',   color='#F44336')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved: {save_path}')


# ── Main Training ──────────────────────────────────────
def train(dataset_path='dataset_combined/'):
    print('='*60)
    print('  Stress Detection - Maximum Accuracy Training')
    print(f'  Device     : {device}')
    print(f'  Epochs     : {EPOCHS}')
    print(f'  Batch Size : {BATCH_SIZE}')
    print(f'  LR         : {LR}')
    print(f'  Patience   : {PATIENCE}')
    print('='*60)

    # Load dataset
    print('\n[1] Loading dataset...')
    X, y = load_dataset(dataset_path)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\n[2] Train: {len(X_train)}  Test: {len(X_test)}')

    # Class weights for imbalanced data
    class_counts = Counter(y_train)
    total = len(y_train)
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts.get(i, 1))
        for i in range(NUM_CLASSES)
    ]).to(device)
    print('\n  Class weights (for imbalanced data):')
    for i, w in enumerate(class_weights):
        print(f'  {EMOTION_LABELS[i]:12s}: {w.item():.3f}')

    # Weighted sampler for balanced training
    sample_weights = [class_weights[y_train[i]].item() for i in range(len(y_train))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders
    train_loader = DataLoader(
        FaceDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )
    test_loader = DataLoader(
        FaceDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    print('\n[3] Building model...')
    model = StressCNN().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {params:,}')

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    # Early stopping
    early_stop = EarlyStopping(patience=PATIENCE)

    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }

    os.makedirs('ml_model', exist_ok=True)
    model_path = 'ml_model/stress_model_pytorch.pth'

    print('\n[4] Training...')
    print('-'*60)

    for epoch in range(EPOCHS):
        # ── Train Phase ──────────────────────────────
        model.train()
        train_loss = train_correct = total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            total         += len(y_batch)

        train_acc  = train_correct / total
        train_loss = train_loss / len(train_loader)

        # ── Validation Phase ─────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total   += len(y_batch)

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(test_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(
            f'Epoch {epoch+1:3d}/{EPOCHS}: '
            f'loss={train_loss:.4f} acc={train_acc:.4f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} '
            f'lr={current_lr:.6f}'
        )

        early_stop(val_acc, model, model_path)
        if early_stop.stop:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print('\n' + '='*60)
    print(f'  Best Accuracy : {early_stop.best_acc*100:.2f}%')
    print(f'  Model saved   : {model_path}')
    print('='*60)

    plot_history(history, 'ml_model/training_history.png')
    print('\nTraining Complete! 🎉')


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'dataset_combined/'
    train(dataset)