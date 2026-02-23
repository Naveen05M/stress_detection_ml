print('='*50)
print('  StressGuard AI - Package Check')
print('='*50)

packages = {
    'numpy': 'Array Processing',
    'tensorflow': 'Deep Learning',
    'torch': 'PyTorch GPU',
    'cv2': 'Face Detection',
    'django': 'Web Framework',
    'sklearn': 'ML Utilities',
    'matplotlib': 'Graphs',
    'seaborn': 'Confusion Matrix',
    'PIL': 'Image Handling',
    'mediapipe': 'Face Landmarks',
    'pymongo': 'MongoDB Driver',
    'djongo': 'MongoDB Django',
}

passed = 0
failed = 0

for package, purpose in packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'OK')
        print(f'  OK      {package:20s} {version:15s} {purpose}')
        passed += 1
    except ImportError:
        print(f'  MISSING {package:20s} NOT INSTALLED   {purpose}')
        failed += 1

print('='*50)
print(f'  Passed : {passed}')
print(f'  Failed : {failed}')
if failed == 0:
    print('  All packages ready!')
else:
    print(f'  Install {failed} missing packages!')
print('='*50)