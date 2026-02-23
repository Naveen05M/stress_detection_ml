import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── GPU Setup ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')

# ── Constants ──────────────────────────────────────────
IMG_SIZE    = 48
BATCH_SIZE  = 64
EPOCHS      = 100
NUM_CLASSES = 7
LR          = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE    = 15
EMOTION_LABELS = [
    'angry','disgusted','fearful',
    'happy','neutral','sad','surprised'
]


# ── Dataset with Augmentation ──────────────────────────
class FaceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [2])
            # Random noise
            if torch.rand(1) > 0.5:
                x = x + torch.randn_like(x) * 0.05
            # Random brightness
            if torch.rand(1) > 0.5:
                x = x * (0.8 + torch.rand(1) * 0.4)
            # Random erasing
            if torch.rand(1) > 0.7:
                x = self.random_erase(x)
            x = torch.clamp(x, 0, 1)
        return x, self.y[idx]

    def random_erase(self, x):
        h, w = x.shape[1], x.shape[2]
        eh = torch.randint(5, 15, (1,)).item()
        ew = torch.randint(5, 15, (1,)).item()
        sy = torch.randint(0, h - eh, (1,)).item()
        sx = torch.randint(0, w - ew, (1,)).item()
        x[:, sy:sy+eh, sx:sx+ew] = 0
        return x


# ── Load Dataset ───────────────────────────────────────
def load_dataset(dataset_path):
    X, y = [], []
    dataset_path = Path(dataset_path)

    for label_idx, label in enumerate(EMOTION_LABELS):
        loaded = 0
        for split in ['train', 'test', '']:
            path = dataset_path / split / label
            if not path.exists():
                continue
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_file in path.glob(ext):
                    img = cv2.imread(
                        str(img_file), cv2.IMREAD_GRAYSCALE
                    )
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label_idx)
                    loaded += 1
        print(f'  {label:12s}: {loaded} images')

    X = np.array(X, dtype='float32') / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    print(f'\n  Total: {len(X)} images loaded')
    return X, y


# ── CNN Model ──────────────────────────────────────────
class StressCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(StressCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Early Stopping ─────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.stop = False

    def __call__(self, val_acc, model, path):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f'  Model saved! Best: {self.best_acc:.4f}')
        else:
            self.counter += 1
            print(f'  No improvement {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.stop = True


# ── Plot History ───────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history['train_acc'], label='Train', color='#2196F3')
    axes[0].plot(history['val_acc'], label='Val', color='#FF5722')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_loss'], label='Train', color='#4CAF50')
    axes[1].plot(history['val_loss'], label='Val', color='#F44336')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ml_model/training_history.png', dpi=120)
    plt.close()
    print('Training history plot saved!')


# ── Main Training ──────────────────────────────────────
def train():
    print('='*60)
    print('  Stress Detection - PyTorch GPU Training')
    print('  All Accuracy Methods Combined')
    print(f'  Device     : {device}')
    print(f'  Epochs     : {EPOCHS}')
    print(f'  Batch Size : {BATCH_SIZE}')
    print(f'  LR         : {LR}')
    print(f'  Weight Decay: {WEIGHT_DECAY}')
    print('='*60)

    # Load data
    print('\n[1] Loading dataset...')
    X, y = load_dataset('dataset/')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\n[2] Train: {len(X_train)}  Test: {len(X_test)}')

    # DataLoaders with augmentation
    train_loader = DataLoader(
        FaceDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        FaceDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Model
    print('\n[3] Building model...')
    model = StressCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {total_params:,}')

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
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
        # ── Train Phase ──
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_acc = train_correct / total
        train_loss = train_loss / len(train_loader)

        # ── Validation Phase ──
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / val_total
        val_loss = val_loss / len(test_loader)

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Current LR
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch {epoch+1:3d}/{EPOCHS}: '
            f'loss={train_loss:.4f} acc={train_acc:.4f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} '
            f'lr={current_lr:.6f}'
        )

        # Early stopping check
        early_stop(val_acc, model, model_path)
        if early_stop.stop:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print('\n' + '='*60)
    print(f'  Best Accuracy : {early_stop.best_acc:.4f} ({early_stop.best_acc*100:.2f}%)')
    print(f'  Model saved   : {model_path}')
    print('='*60)

    # Save plots
    plot_history(history)
    print('\nTraining Complete!')


if __name__ == '__main__':
    train()