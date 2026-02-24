"""
Merge FER2013 + RAF-DB datasets for better accuracy
"""
import os
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
FER2013_PATH = r"C:\Users\NAVEENKUMAR\stress_detection\dataset"
RAFDB_PATH   = r"C:\Users\NAVEENKUMAR\Downloads\RAF-DB"
OUTPUT_PATH  = r"C:\Users\NAVEENKUMAR\stress_detection\dataset_combined"

EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

# RAF-DB uses different names — map them
RAFDB_MAP = {
    'anger':    'angry',
    'disgust':  'disgusted',
    'fear':     'fearful',
    'happiness':'happy',
    'neutral':  'neutral',
    'sadness':  'sad',
    'surprise': 'surprised',
}

def merge():
    print('='*50)
    print('  Merging FER2013 + RAF-DB')
    print('='*50)

    # Create output folders
    for split in ['train', 'test']:
        for emotion in EMOTIONS:
            os.makedirs(f'{OUTPUT_PATH}/{split}/{emotion}', exist_ok=True)

    counts = {e: 0 for e in EMOTIONS}

    # Copy FER2013
    print('\n[1] Copying FER2013...')
    for split in ['train', 'test']:
        for emotion in EMOTIONS:
            src = Path(FER2013_PATH) / split / emotion
            dst = Path(OUTPUT_PATH) / split / emotion
            if not src.exists():
                continue
            for i, img in enumerate(src.glob('*.png')):
                shutil.copy(img, dst / f'fer_{img.name}')
                counts[emotion] += 1
            for i, img in enumerate(src.glob('*.jpg')):
                shutil.copy(img, dst / f'fer_{img.name}')
                counts[emotion] += 1

    # Copy RAF-DB
    print('[2] Copying RAF-DB...')
    rafdb = Path(RAFDB_PATH)
    for split in ['train', 'test']:
        split_path = rafdb / split
        if not split_path.exists():
            split_path = rafdb / 'basic' / split
        if not split_path.exists():
            continue
        for rafdb_name, emotion in RAFDB_MAP.items():
            src = split_path / rafdb_name
            if not src.exists():
                src = split_path / rafdb_name.capitalize()
            if not src.exists():
                continue
            dst = Path(OUTPUT_PATH) / split / emotion
            for img in src.glob('*.jpg'):
                shutil.copy(img, dst / f'raf_{img.name}')
                counts[emotion] += 1
            for img in src.glob('*.png'):
                shutil.copy(img, dst / f'raf_{img.name}')
                counts[emotion] += 1

    print('\n[3] Dataset Merge Complete!')
    print('='*50)
    total = 0
    for emotion, count in counts.items():
        print(f'  {emotion:12s}: {count} images')
        total += count
    print(f'\n  Total: {total} images')
    print(f'  Output: {OUTPUT_PATH}')
    print('='*50)

if __name__ == '__main__':
    merge()
