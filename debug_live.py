# Place this in: C:\Users\NAVEENKUMAR\stress_detection\stress_app\views.py
# Replace the user_live_frame function with this debug version

# JUST RUN THIS TO TEST FACE DETECTION:
# python debug_live.py

import sys, os
sys.path.insert(0, 'ml_model')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stress_project.settings')

import cv2
import numpy as np

print("Testing face detection with webcam...")
print("Press SPACE to capture and detect, press Q to quit")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

from predict import predict_from_image_array
print("Model loaded OK")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    # Show live feed
    cv2.imshow('StressNet Test - Press SPACE to detect, Q to quit', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("Detecting...")
        results = predict_from_image_array(frame)
        print(f"Faces detected: {len(results)}")
        for r in results:
            print(f"  Emotion: {r['emotion']}, Stress: {r['stress_level']}, Confidence: {r['confidence']}%")
            x,y,w,h = r['bbox']
            color = (0,255,0) if r['stress_level']=='Low' else (0,165,255) if r['stress_level']=='Medium' else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{r['emotion']} {r['confidence']:.0f}%", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if len(results) == 0:
            print("  No face detected - make sure face is visible and well lit")
        cv2.imshow('Result', frame)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()