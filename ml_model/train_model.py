"""
============================================================
  Stress Detection - Deep CNN Model Training
  Dataset: FER2013 / RAF-DB compatible (48x48 grayscale)
  Categories: angry, disgusted, fearful, happy, neutral, sad, surprised
  Author: Stress Detection System
  Usage: python train_model.py --dataset_path /path/to/dataset
============================================================
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 60
NUM_CLASSES = 7
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


def load_dataset_from_directory(dataset_path):
    """
    Load images from folder structure:
        dataset/
            train/
                angry/    *.png / *.jpg
                happy/    ...
            test/
                angry/
                ...
    OR flat structure:
        dataset/
            angry/
            happy/
            ...
    """
    X, y = [], []

    # Check if train/test split already exists
    if os.path.isdir(os.path.join(dataset_path, 'train')):
        for split in ['train', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue
            for label_idx, label in enumerate(EMOTION_LABELS):
                label_path = os.path.join(split_path, label)
                if not os.path.isdir(label_path):
                    print(f"  [WARN] Missing folder: {label_path}")
                    continue
                for fname in os.listdir(label_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_path, fname)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X.append(img)
                        y.append(label_idx)
    else:
        # Flat structure
        for label_idx, label in enumerate(EMOTION_LABELS):
            label_path = os.path.join(dataset_path, label)
            if not os.path.isdir(label_path):
                print(f"  [WARN] Missing folder: {label_path}")
                continue
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(label_path, fname)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label_idx)

    X = np.array(X, dtype='float32') / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    print(f"  Loaded {len(X)} images from {dataset_path}")
    return X, y


def build_deep_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Deep CNN architecture optimized for 48×48 grayscale facial expression images.
    Architecture: 4 Conv Blocks → GAP → Dense → Softmax
    """
    inputs = Input(shape=input_shape)

    # --- Block 1 ---
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # --- Block 2 ---
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # --- Block 3 ---
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # --- Block 4 ---
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # --- Classifier Head ---
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='StressDetectionCNN')
    return model


def plot_training_history(history, save_dir='ml_model'):
    """Plot and save training/validation accuracy & loss curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Acc', color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='#FF5722')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='#4CAF50')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#F44336')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Training history saved to {save_dir}/training_history.png")


def plot_confusion_matrix(y_true, y_pred, save_dir='ml_model'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        linewidths=0.5
    )
    plt.title('Confusion Matrix – Emotion Classification', fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_dir}/confusion_matrix.png")


def train(dataset_path, model_save_path='ml_model/stress_model.h5'):
    """Main training pipeline."""
    print("=" * 60)
    print("  Stress Detection – Deep CNN Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading dataset...")
    X, y = load_dataset_from_directory(dataset_path)
    print(f"  Dataset shape: {X.shape}, Labels shape: {y.shape}")
    print(f"  Class distribution: { {EMOTION_LABELS[i]: int(np.sum(y==i)) for i in range(NUM_CLASSES)} }")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    print(f"\n[2] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest',
    )
    datagen.fit(X_train)

    # 4. Build model
    print("\n[3] Building Deep CNN model...")
    model = build_deep_cnn()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Callbacks
    os.makedirs('ml_model', exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # 6. Train
    print("\n[4] ...")
    history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=None,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

    # 7. Evaluate
    print("\n[5] Evaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))

    # 8. Save visualizations
    print("\n[6] Saving plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)

    # 9. Save metrics to file
    metrics = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'epochs_trained': len(history.history['accuracy']),
    }
    import json
    with open('ml_model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[7] Model saved to: {model_save_path}")
    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)
    return model, history, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stress Detection CNN')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset folder (with emotion subfolders)')
    parser.add_argument('--model_save', type=str, default='ml_model/stress_model.h5',
                        help='Path to save trained model')
    args = parser.parse_args()
    train(args.dataset_path, args.model_save)
