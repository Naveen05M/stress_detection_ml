"""
Models for StressNet Authentication System
- Employee (user profile with approval)
- StressRecord (emotion detection history)
"""
from django.db import models
from django.contrib.auth.models import User


DEPARTMENT_CHOICES = [
    ('Engineering',      'Engineering'),
    ('Software Dev',     'Software Development'),
    ('QA & Testing',     'QA & Testing'),
    ('DevOps',           'DevOps'),
    ('Data Science',     'Data Science'),
    ('Product',          'Product Management'),
    ('Design',           'Design & UX'),
    ('HR',               'Human Resources'),
    ('Finance',          'Finance'),
    ('Sales',            'Sales & Marketing'),
    ('Support',          'Customer Support'),
    ('Management',       'Management'),
    ('Other',            'Other'),
]

STATUS_CHOICES = [
    ('pending',  'Pending Approval'),
    ('approved', 'Approved'),
    ('rejected', 'Rejected'),
]

STRESS_CHOICES = [
    ('High',   'High'),
    ('Medium', 'Medium'),
    ('Low',    'Low'),
]

EMOTION_CHOICES = [
    ('angry',     'Angry'),
    ('disgusted', 'Disgusted'),
    ('fearful',   'Fearful'),
    ('happy',     'Happy'),
    ('neutral',   'Neutral'),
    ('sad',       'Sad'),
    ('surprised', 'Surprised'),
]


class Employee(models.Model):
    """Extended user profile for employees"""
    user          = models.OneToOneField(User, on_delete=models.CASCADE,
                                         related_name='employee')
    employee_id   = models.CharField(max_length=20, unique=True,
                                     verbose_name='Employee ID')
    department    = models.CharField(max_length=50, choices=DEPARTMENT_CHOICES,
                                     default='Engineering')
    designation   = models.CharField(max_length=100, blank=True,
                                     verbose_name='Designation / Role')
    phone         = models.CharField(max_length=15, blank=True,
                                     verbose_name='Phone Number')
    date_of_join  = models.DateField(null=True, blank=True,
                                     verbose_name='Date of Joining')
    profile_photo = models.ImageField(upload_to='profile_photos/',
                                      null=True, blank=True)
    status        = models.CharField(max_length=10, choices=STATUS_CHOICES,
                                     default='pending')
    approved_by   = models.ForeignKey(User, null=True, blank=True,
                                      on_delete=models.SET_NULL,
                                      related_name='approved_employees')
    approved_at   = models.DateTimeField(null=True, blank=True)
    rejection_reason = models.TextField(blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)
    last_active   = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-registered_at']
        verbose_name = 'Employee'
        verbose_name_plural = 'Employees'

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.employee_id})"

    @property
    def full_name(self):
        return self.user.get_full_name() or self.user.username

    @property
    def is_approved(self):
        return self.status == 'approved'

    @property
    def total_detections(self):
        return self.stressrecord_set.count()

    @property
    def stress_summary(self):
        records = self.stressrecord_set.all()
        total   = records.count()
        if total == 0:
            return {'high': 0, 'medium': 0, 'low': 0, 'total': 0}
        return {
            'high':   records.filter(stress_level='High').count(),
            'medium': records.filter(stress_level='Medium').count(),
            'low':    records.filter(stress_level='Low').count(),
            'total':  total,
        }

    @property
    def latest_emotion(self):
        record = self.stressrecord_set.order_by('-timestamp').first()
        return record.emotion if record else 'N/A'

    @property
    def latest_stress(self):
        record = self.stressrecord_set.order_by('-timestamp').first()
        return record.stress_level if record else 'N/A'


class StressRecord(models.Model):
    """Individual emotion detection record"""
    employee      = models.ForeignKey(Employee, on_delete=models.CASCADE,
                                      null=True, blank=True)
    timestamp     = models.DateTimeField(auto_now_add=True)
    emotion       = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    stress_level  = models.CharField(max_length=10, choices=STRESS_CHOICES)
    confidence    = models.FloatField(default=0.0)
    source        = models.CharField(max_length=10, default='live',
                                     choices=[('live','Live'),('image','Image')])
    # Individual emotion scores
    angry_score     = models.FloatField(default=0.0)
    disgusted_score = models.FloatField(default=0.0)
    fearful_score   = models.FloatField(default=0.0)
    happy_score     = models.FloatField(default=0.0)
    neutral_score   = models.FloatField(default=0.0)
    sad_score       = models.FloatField(default=0.0)
    surprised_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        emp = self.employee.full_name if self.employee else 'Unknown'
        return f"{emp} - {self.emotion} ({self.stress_level}) @ {self.timestamp:%H:%M}"


class Notification(models.Model):
    """Admin notifications for new registrations"""
    message    = models.TextField()
    is_read    = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    link       = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.message[:50]