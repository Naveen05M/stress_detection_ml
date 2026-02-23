"""
============================================================
  Dataset Preparation Utility
  Converts FER2013 CSV format to folder structure
  FER2013 CSV format: emotion,pixels,Usage
  Usage: python dataset_utils/prepare_dataset.py --csv fer2013.csv --output dataset/
============================================================
"""
import os
import csv
import argparse
import numpy as np
import cv2
from pathlib import Path

EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_SIZE = 48


def fer2013_csv_to_folders(csv_path, output_dir):
    """
    Convert FER2013 CSV (emotion, pixels, Usage) to folder structure:
        output_dir/
            train/
                angry/  001.png  002.png ...
                happy/  ...
            test/
                angry/
                ...
    """
    output_dir = Path(output_dir)
    counts = {'Training': {em: 0 for em in EMOTION_LABELS},
              'PublicTest': {em: 0 for em in EMOTION_LABELS}}

    split_map = {'Training': 'train', 'PublicTest': 'test', 'PrivateTest': 'test'}

    # Create directories
    for split in ['train', 'test']:
        for em in EMOTION_LABELS:
            (output_dir / split / em).mkdir(parents=True, exist_ok=True)

    print(f"Reading: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            emotion_idx = int(row['emotion'])
            if emotion_idx >= len(EMOTION_LABELS):
                continue

            emotion = EMOTION_LABELS[emotion_idx]
            usage = row.get('Usage', 'Training')
            split = split_map.get(usage, 'train')

            pixels = list(map(int, row['pixels'].split()))
            if len(pixels) != IMG_SIZE * IMG_SIZE:
                continue

            img = np.array(pixels, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
            save_path = output_dir / split / emotion / f"{idx:06d}.png"
            cv2.imwrite(str(save_path), img)

            if usage in counts:
                counts[usage][emotion] += 1

            if idx % 1000 == 0:
                print(f"  Processed {idx} images...")

    print("\nDataset created successfully!")
    print("\nTrain set:")
    for em, cnt in counts['Training'].items():
        print(f"  {em:12s}: {cnt}")
    print("\nTest set:")
    for em, cnt in counts['PublicTest'].items():
        print(f"  {em:12s}: {cnt}")
    print(f"\nSaved to: {output_dir}")


def verify_dataset(dataset_path):
    """Verify dataset structure and count images."""
    dataset_path = Path(dataset_path)
    print(f"\nVerifying dataset at: {dataset_path}")
    total = 0
    for split in ['train', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            # Try flat structure
            split_path = dataset_path
        print(f"\n  {split.upper()}:")
        for em in EMOTION_LABELS:
            em_path = split_path / em
            if em_path.exists():
                count = len(list(em_path.glob('*.png')) + list(em_path.glob('*.jpg')))
                print(f"    {em:12s}: {count} images")
                total += count
    print(f"\n  TOTAL: {total} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to fer2013.csv')
    parser.add_argument('--output', type=str, default='dataset', help='Output directory')
    parser.add_argument('--verify', type=str, help='Verify dataset at given path')
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.verify)
    elif args.csv:
        fer2013_csv_to_folders(args.csv, args.output)
    else:
        print("Usage examples:")
        print("  python prepare_dataset.py --csv fer2013.csv --output dataset/")
        print("  python prepare_dataset.py --verify dataset/")
