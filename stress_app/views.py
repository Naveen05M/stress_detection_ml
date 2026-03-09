import json
import base64
import sys
import os
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.db.models import Count, Q
from .models import Employee, StressRecord, Notification
from .forms import EmployeeRegistrationForm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model'))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_admin(user):
    return user.is_authenticated and (user.is_staff or user.is_superuser)

def get_employee(user):
    try:
        return Employee.objects.get(user=user)
    except Employee.DoesNotExist:
        return None

def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login_view')
        if not is_admin(request.user):
            return redirect('user_dashboard')
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

def employee_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login_view')
        if is_admin(request.user):
            return redirect('admin_dashboard')
        emp = get_employee(request.user)
        if not emp:
            logout(request)
            return redirect('login_view')
        if emp.status == 'pending':
            return render(request, 'stress_app/pending.html', {'emp': emp})
        if emp.status == 'rejected':
            return render(request, 'stress_app/rejected.html', {'emp': emp})
        return view_func(request, *args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

# ─── Auth Views ───────────────────────────────────────────────────────────────

def login_view(request):
    # If already logged in, redirect to correct page
    if request.user.is_authenticated:
        if is_admin(request.user):
            return redirect('admin_dashboard')
        emp = get_employee(request.user)
        if emp:
            if emp.status == 'approved':
                return redirect('user_dashboard')
            elif emp.status == 'pending':
                return render(request, 'stress_app/pending.html', {'emp': emp})
            else:
                return render(request, 'stress_app/rejected.html', {'emp': emp})
        return redirect('user_dashboard')

    error = None
    if request.method == 'POST':
        identifier = request.POST.get('identifier', '').strip()
        password   = request.POST.get('password', '').strip()

        # Try email login
        user = None
        try:
            u = User.objects.get(email=identifier)
            user = authenticate(request, username=u.username, password=password)
        except User.DoesNotExist:
            pass

        # Try employee_id login
        if not user:
            try:
                emp = Employee.objects.get(employee_id=identifier)
                user = authenticate(request, username=emp.user.username, password=password)
            except Employee.DoesNotExist:
                pass

        # Try username login
        if not user:
            user = authenticate(request, username=identifier, password=password)

        if user:
            login(request, user)
            # Update last active for employees
            emp = get_employee(user)
            if emp:
                emp.last_active = timezone.now()
                emp.save(update_fields=['last_active'])

            # Redirect based on role
            if is_admin(user):
                return redirect('admin_dashboard')
            if emp:
                if emp.status == 'approved':
                    return redirect('user_dashboard')
                elif emp.status == 'pending':
                    return render(request, 'stress_app/pending.html', {'emp': emp})
                else:
                    return render(request, 'stress_app/rejected.html', {'emp': emp})
            return redirect('user_dashboard')
        else:
            error = 'Invalid credentials. Please check your Employee ID / Email and password.'

    return render(request, 'stress_app/login.html', {'error': error})


def register_view(request):
    if request.user.is_authenticated:
        return redirect('login_view')

    form = EmployeeRegistrationForm()
    if request.method == 'POST':
        form = EmployeeRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data['email'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name'],
            )
            emp = Employee.objects.create(
                user=user,
                employee_id=form.cleaned_data['employee_id'],
                department=form.cleaned_data['department'],
                designation=form.cleaned_data.get('designation', ''),
                phone=form.cleaned_data.get('phone', ''),
                date_of_join=form.cleaned_data.get('date_of_join'),
                profile_photo=request.FILES.get('profile_photo'),
                status='pending',
            )
            Notification.objects.create(
                message=f"New registration: {emp.full_name} ({emp.employee_id}) is waiting for approval.",
                link=f"/admin-panel/employees/{emp.id}/"
            )
            return render(request, 'stress_app/register.html', {
                'form': EmployeeRegistrationForm(),
                'success': True
            })
    return render(request, 'stress_app/register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login_view')


# ─── Employee Views ───────────────────────────────────────────────────────────

@employee_required
def user_dashboard(request):
    emp = get_employee(request.user)
    records = StressRecord.objects.filter(employee=emp).order_by('-timestamp')[:50]
    total   = StressRecord.objects.filter(employee=emp).count()
    high    = StressRecord.objects.filter(employee=emp, stress_level='High').count()
    medium  = StressRecord.objects.filter(employee=emp, stress_level='Medium').count()
    low     = StressRecord.objects.filter(employee=emp, stress_level='Low').count()

    # Timeline data for chart
    timeline = []
    for r in reversed(list(StressRecord.objects.filter(employee=emp).order_by('-timestamp')[:20])):
        timeline.append({
            'timestamp': r.timestamp.strftime('%H:%M'),
            'stress_level': r.stress_level,
            'emotion': r.emotion,
        })

    # Emotion counts
    emotion_counts = {}
    for r in StressRecord.objects.filter(employee=emp):
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    return render(request, 'stress_app/user_dashboard.html', {
        'emp': emp,
        'records': records,
        'total': total, 'high': high, 'medium': medium, 'low': low,
        'timeline': json.dumps(timeline),
        'emotion_counts': json.dumps(emotion_counts),
    })


@employee_required
def user_live(request):
    emp = get_employee(request.user)
    return render(request, 'stress_app/user_live.html', {'emp': emp})


@employee_required
@require_POST
def user_live_frame(request):
    emp = get_employee(request.user)
    try:
        data     = json.loads(request.body)
        frame_b64 = data.get('frame', '')
        if not frame_b64:
            return JsonResponse({'error': 'No frame'}, status=400)

        # Decode base64 frame
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        frame_bytes = base64.b64decode(frame_b64)

        import numpy as np
        import cv2
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({'error': 'Invalid image'}, status=400)

        # Run prediction
        from predict import predict_from_image_array
        results = predict_from_image_array(img)
        face_count = len(results)

        # Annotate frame
        import base64 as b64mod
        annotated = img.copy()
        for r in results:
            x, y, w, h = r['bbox']
            sl = r.get('stress_level', 'Low')
            color = (0, 0, 255) if sl == 'High' else (0, 165, 255) if sl == 'Medium' else (0, 255, 0)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            label = f"{r['emotion']} {r['confidence']:.0f}%"
            cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buf = cv2.imencode('.jpg', annotated)
        annotated_frame_b64 = 'data:image/jpeg;base64,' + b64mod.b64encode(buf).decode()

        # Save records
        if emp and face_count > 0:
            for r in results:
                StressRecord.objects.create(
                    employee=emp,
                    emotion=r.get('emotion', 'neutral'),
                    stress_level=r.get('stress_level', 'Low'),
                    confidence=r.get('confidence', 0),
                    source='live',
                    angry_score=r.get('scores', {}).get('angry', 0),
                    happy_score=r.get('scores', {}).get('happy', 0),
                    neutral_score=r.get('scores', {}).get('neutral', 0),
                    sad_score=r.get('scores', {}).get('sad', 0),
                    fearful_score=r.get('scores', {}).get('fearful', 0),
                    disgusted_score=r.get('scores', {}).get('disgusted', 0),
                    surprised_score=r.get('scores', {}).get('surprised', 0),
                )
            emp.last_active = timezone.now()
            emp.save(update_fields=['last_active'])

        emp.last_active = timezone.now()
        emp.save(update_fields=['last_active'])

        return JsonResponse({
            'results': results,
            'annotated_frame': annotated_frame_b64,
            'face_count': face_count,
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@employee_required
def user_history(request):
    emp     = get_employee(request.user)
    records = StressRecord.objects.filter(employee=emp).order_by('-timestamp')
    return render(request, 'stress_app/user_history.html', {'emp': emp, 'records': records})


@employee_required
def user_profile(request):
    emp = get_employee(request.user)
    return render(request, 'stress_app/user_profile.html', {'emp': emp})


# ─── Admin Views ──────────────────────────────────────────────────────────────

@admin_required
def admin_dashboard(request):
    total_emp   = Employee.objects.count()
    approved    = Employee.objects.filter(status='approved').count()
    pending     = Employee.objects.filter(status='pending').count()
    rejected    = Employee.objects.filter(status='rejected').count()
    total_rec   = StressRecord.objects.count()

    since_24h   = timezone.now() - timedelta(hours=24)
    high_today  = StressRecord.objects.filter(stress_level='High', timestamp__gte=since_24h).count()

    # Timeline last 24h
    timeline = []
    for r in StressRecord.objects.filter(timestamp__gte=since_24h).order_by('timestamp')[:50]:
        timeline.append({
            'timestamp': r.timestamp.strftime('%H:%M'),
            'stress_level': r.stress_level,
        })

    # Emotion distribution
    emotion_counts = {}
    for r in StressRecord.objects.all():
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    # Department stress
    departments = Employee.objects.filter(status='approved').values_list('department', flat=True).distinct()
    dept_stress = []
    for dept in departments:
        emps = Employee.objects.filter(department=dept, status='approved')
        emp_ids = emps.values_list('id', flat=True)
        total_d = StressRecord.objects.filter(employee_id__in=emp_ids).count()
        high_d  = StressRecord.objects.filter(employee_id__in=emp_ids, stress_level='High').count()
        pct     = round(high_d * 100 / total_d) if total_d > 0 else 0
        dept_stress.append({'dept': dept, 'total': total_d, 'high': high_d, 'pct': pct})
    dept_stress.sort(key=lambda x: x['pct'], reverse=True)

    recent = StressRecord.objects.select_related('employee__user').order_by('-timestamp')[:20]
    unread = Notification.objects.filter(is_read=False).count()

    return render(request, 'stress_app/admin_dashboard.html', {
        'total_emp': total_emp, 'approved': approved,
        'pending': pending, 'rejected': rejected,
        'total_rec': total_rec, 'high_today': high_today,
        'timeline': json.dumps(timeline),
        'emotion_counts': json.dumps(emotion_counts),
        'dept_stress': dept_stress,
        'recent': recent,
        'unread': unread,
    })


@admin_required
def admin_employees(request):
    status_filter = request.GET.get('status', '')
    dept_filter   = request.GET.get('dept', '')
    search        = request.GET.get('search', '')

    employees = Employee.objects.select_related('user').all()
    if status_filter:
        employees = employees.filter(status=status_filter)
    if dept_filter:
        employees = employees.filter(department=dept_filter)
    if search:
        employees = employees.filter(
            Q(user__first_name__icontains=search) |
            Q(user__last_name__icontains=search)  |
            Q(employee_id__icontains=search)
        )
    employees = employees.order_by('-registered_at')

    # Attach latest record to each employee
    for emp in employees:
        emp.latest_record = StressRecord.objects.filter(employee=emp).order_by('-timestamp').first()

    dept_choices = Employee._meta.get_field('department').choices or []
    unread       = Notification.objects.filter(is_read=False).count()

    return render(request, 'stress_app/admin_employees.html', {
        'employees': employees,
        'status_filter': status_filter,
        'dept_filter': dept_filter,
        'search': search,
        'dept_choices': dept_choices,
        'unread': unread,
    })


@admin_required
def admin_approve(request, emp_id):
    emp = get_object_or_404(Employee, id=emp_id)
    emp.status      = 'approved'
    emp.approved_by = request.user
    emp.approved_at = timezone.now()
    emp.rejection_reason = ''
    emp.save()
    return redirect('admin_employees')


@admin_required
def admin_reject(request, emp_id):
    emp = get_object_or_404(Employee, id=emp_id)
    if request.method == 'POST':
        emp.status           = 'rejected'
        emp.rejection_reason = request.POST.get('reason', '')
        emp.save()
    return redirect('admin_employees')


@admin_required
def admin_employee_detail(request, emp_id):
    emp     = get_object_or_404(Employee, id=emp_id)
    records = StressRecord.objects.filter(employee=emp).order_by('-timestamp')

    total  = records.count()
    high   = records.filter(stress_level='High').count()
    medium = records.filter(stress_level='Medium').count()
    low    = records.filter(stress_level='Low').count()
    summary = {'total': total, 'high': high, 'medium': medium, 'low': low}

    timeline = []
    for r in reversed(list(records[:30])):
        timeline.append({'timestamp': r.timestamp.strftime('%H:%M'), 'stress_level': r.stress_level})

    emotion_counts = {}
    for r in records:
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    unread = Notification.objects.filter(is_read=False).count()

    return render(request, 'stress_app/admin_employee_detail.html', {
        'emp': emp,
        'records': records,
        'summary': summary,
        'timeline': json.dumps(timeline),
        'emotion_counts': json.dumps(emotion_counts),
        'unread': unread,
    })


@admin_required
def admin_realtime(request):
    employees = Employee.objects.filter(status='approved').select_related('user')
    for emp in employees:
        emp.latest_record = StressRecord.objects.filter(employee=emp).order_by('-timestamp').first()
        emp.is_online = emp.last_active and (timezone.now() - emp.last_active).seconds < 120
    unread = Notification.objects.filter(is_read=False).count()
    return render(request, 'stress_app/admin_realtime.html', {
        'employees': employees,
        'unread': unread,
    })


@admin_required
def admin_notifications(request):
    Notification.objects.filter(is_read=False).update(is_read=True)
    notifs = Notification.objects.all().order_by('-created_at')
    return render(request, 'stress_app/admin_notifications.html', {'notifs': notifs, 'unread': 0})


# ─── Admin API ────────────────────────────────────────────────────────────────

@admin_required
def api_all_employees_live(request):
    employees = Employee.objects.filter(status='approved').select_related('user')
    data = []
    for emp in employees:
        latest = StressRecord.objects.filter(employee=emp).order_by('-timestamp').first()
        is_online = emp.last_active and (timezone.now() - emp.last_active).seconds < 120
        data.append({
            'id':           emp.id,
            'name':         emp.full_name,
            'employee_id':  emp.employee_id,
            'department':   emp.department,
            'is_online':    is_online,
            'last_active':  emp.last_active.strftime('%H:%M:%S') if emp.last_active else '--',
            'emotion':      latest.emotion      if latest else '--',
            'stress_level': latest.stress_level if latest else '--',
            'confidence':   latest.confidence   if latest else 0,
            'timestamp':    latest.timestamp.strftime('%H:%M:%S') if latest else '--',
        })
    return JsonResponse({'employees': data})


@admin_required
def api_employee_records(request, emp_id):
    """API: Get latest records for a specific employee (for admin live watch)"""
    emp     = get_object_or_404(Employee, id=emp_id)
    records = StressRecord.objects.filter(employee=emp).order_by('-timestamp')[:20]
    latest  = records.first()
    is_online = emp.last_active and (timezone.now() - emp.last_active).seconds < 120

    data = {
        'emp_id':      emp.id,
        'name':        emp.full_name,
        'employee_id': emp.employee_id,
        'department':  emp.department,
        'is_online':   is_online,
        'last_active': emp.last_active.strftime('%H:%M:%S') if emp.last_active else '--',
        'latest': {
            'emotion':      latest.emotion      if latest else '--',
            'stress_level': latest.stress_level if latest else '--',
            'confidence':   latest.confidence   if latest else 0,
            'timestamp':    latest.timestamp.strftime('%d %b %Y %H:%M:%S') if latest else '--',
            'angry':        latest.angry_score    if latest else 0,
            'happy':        latest.happy_score    if latest else 0,
            'neutral':      latest.neutral_score  if latest else 0,
            'sad':          latest.sad_score      if latest else 0,
            'fearful':      latest.fearful_score  if latest else 0,
            'disgusted':    latest.disgusted_score if latest else 0,
            'surprised':    latest.surprised_score if latest else 0,
        } if latest else None,
        'history': [
            {
                'timestamp':   r.timestamp.strftime('%H:%M:%S'),
                'emotion':     r.emotion,
                'stress_level': r.stress_level,
                'confidence':  r.confidence,
            } for r in records
        ]
    }
    return JsonResponse(data)
