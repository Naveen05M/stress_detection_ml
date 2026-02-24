"""
Django Views for Stress Detection System
Complete optimized version
"""
import os
import sys
import json
import base64
import tempfile
import cv2
import numpy as np
from pathlib import Path
from datetime import timedelta

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .models import StressRecord, ModelMetrics

# Add ml_model to sys.path
sys.path.insert(0, str(Path(settings.BASE_DIR) / 'ml_model'))


# ─────────────────────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────────────────────
def dashboard(request):
    records = StressRecord.objects.all().order_by('-timestamp')[:50]
    total   = StressRecord.objects.count()

    high_count   = StressRecord.objects.filter(stress_level='High').count()
    medium_count = StressRecord.objects.filter(stress_level='Medium').count()
    low_count    = StressRecord.objects.filter(stress_level='Low').count()

    emotion_counts = {}
    for em in settings.EMOTION_CLASSES:
        emotion_counts[em] = StressRecord.objects.filter(emotion=em).count()

    timeline = list(
        StressRecord.objects.order_by('-timestamp')[:20].values(
            'timestamp', 'stress_level', 'emotion', 'confidence'
        )
    )
    timeline.reverse()
    for t in timeline:
        t['timestamp'] = t['timestamp'].strftime('%H:%M:%S')

    metrics = ModelMetrics.objects.first()

    context = {
        'records':        records,
        'total':          total,
        'high_count':     high_count,
        'medium_count':   medium_count,
        'low_count':      low_count,
        'emotion_counts': json.dumps(emotion_counts),
        'timeline':       json.dumps(timeline),
        'metrics':        metrics,
    }
    return render(request, 'stress_app/dashboard.html', context)


# ─────────────────────────────────────────────────────────────
#  Image Upload Detection
# ─────────────────────────────────────────────────────────────
def upload_detect(request):
    if request.method == 'GET':
        return render(request, 'stress_app/upload.html')

    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, 'No image file provided.')
            return render(request, 'stress_app/upload.html')

        uploaded_file = request.FILES['image']

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            img_bgr = cv2.imread(tmp_path)

            # Resize large images for faster prediction
            if img_bgr is not None:
                max_size = 800
                h, w     = img_bgr.shape[:2]
                if w > max_size or h > max_size:
                    scale   = max_size / max(w, h)
                    img_bgr = cv2.resize(
                        img_bgr,
                        (int(w * scale), int(h * scale))
                    )
                cv2.imwrite(tmp_path, img_bgr)

            from predict import predict_and_annotate
            results, annotated_b64, face_count = predict_and_annotate(tmp_path)

        except FileNotFoundError as e:
            os.unlink(tmp_path)
            messages.error(request, str(e))
            return render(request, 'stress_app/upload.html',
                          {'model_missing': True})
        except Exception as e:
            os.unlink(tmp_path)
            messages.error(request, f'Prediction error: {str(e)}')
            return render(request, 'stress_app/upload.html')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        saved_records = []
        for r in results:
            record = StressRecord.objects.create(
                emotion         = r['emotion'],
                stress_level    = r['stress_level'],
                confidence      = r['confidence'],
                source          = 'image',
                angry_score     = r['scores'].get('angry',     0),
                disgusted_score = r['scores'].get('disgusted', 0),
                fearful_score   = r['scores'].get('fearful',   0),
                happy_score     = r['scores'].get('happy',     0),
                neutral_score   = r['scores'].get('neutral',   0),
                sad_score       = r['scores'].get('sad',       0),
                surprised_score = r['scores'].get('surprised', 0),
            )
            saved_records.append(record)

        context = {
            'results':       results,
            'annotated_b64': annotated_b64,
            'face_count':    face_count,
            'records':       saved_records,
        }
        return render(request, 'stress_app/result.html', context)


# ─────────────────────────────────────────────────────────────
#  Live Camera Detection
# ─────────────────────────────────────────────────────────────
def live_detect(request):
    return render(request, 'stress_app/live.html')


@csrf_exempt
def live_frame(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        body       = json.loads(request.body)
        frame_data = body.get('frame', '')

        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JsonResponse({'error': 'Invalid image'}, status=400)

        img_bgr = cv2.resize(img_bgr, (640, 480))

        from predict import predict_from_image_array, annotate_image, image_to_base64

        results   = predict_from_image_array(img_bgr)
        annotated = annotate_image(img_bgr, results)
        ann_b64   = image_to_base64(annotated)

        for r in results:
            StressRecord.objects.create(
                emotion         = r['emotion'],
                stress_level    = r['stress_level'],
                confidence      = r['confidence'],
                source          = 'live',
                angry_score     = r['scores'].get('angry',     0),
                disgusted_score = r['scores'].get('disgusted', 0),
                fearful_score   = r['scores'].get('fearful',   0),
                happy_score     = r['scores'].get('happy',     0),
                neutral_score   = r['scores'].get('neutral',   0),
                sad_score       = r['scores'].get('sad',       0),
                surprised_score = r['scores'].get('surprised', 0),
            )

        return JsonResponse({
            'annotated_frame': 'data:image/jpeg;base64,' + ann_b64,
            'results':         results,
            'face_count':      len(results),
        })

    except FileNotFoundError as e:
        return JsonResponse({
            'error':  'Model not found. Train first.',
            'detail': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─────────────────────────────────────────────────────────────
#  History & Reports
# ─────────────────────────────────────────────────────────────
def history(request):
    records        = StressRecord.objects.all().order_by('-timestamp')
    filter_level   = request.GET.get('level',   '')
    filter_emotion = request.GET.get('emotion', '')

    if filter_level:
        records = records.filter(stress_level=filter_level)
    if filter_emotion:
        records = records.filter(emotion=filter_emotion)

    return render(request, 'stress_app/history.html', {
        'records':        records,
        'emotion_labels': settings.EMOTION_CLASSES,
        'filter_level':   filter_level,
        'filter_emotion': filter_emotion,
    })


def api_stress_timeline(request):
    hours   = int(request.GET.get('hours', 24))
    since   = timezone.now() - timedelta(hours=hours)
    records = StressRecord.objects.filter(
        timestamp__gte=since
    ).order_by('timestamp')

    data = [
        {
            'timestamp':    r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'stress_level': r.stress_level,
            'emotion':      r.emotion,
            'confidence':   r.confidence,
        }
        for r in records
    ]
    return JsonResponse({'data': data, 'count': len(data)})


def delete_record(request, record_id):
    try:
        record = StressRecord.objects.get(pk=record_id)
        record.delete()
        messages.success(request, 'Record deleted.')
    except StressRecord.DoesNotExist:
        messages.error(request, 'Record not found.')
    return redirect('history')