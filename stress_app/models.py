from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class StressRecord(models.Model):
    """Model to store stress detection results."""
    STRESS_LEVELS = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High'),
    ]
    EMOTION_CHOICES = [
        ('angry', 'Angry'),
        ('disgusted', 'Disgusted'),
        ('fearful', 'Fearful'),
        ('happy', 'Happy'),
        ('neutral', 'Neutral'),
        ('sad', 'Sad'),
        ('surprised', 'Surprised'),
    ]
    SOURCE_CHOICES = [
        ('image', 'Image Upload'),
        ('live', 'Live Camera'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    stress_level = models.CharField(max_length=10, choices=STRESS_LEVELS)
    confidence = models.FloatField(default=0.0)
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default='image')
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    notes = models.TextField(blank=True, null=True)
    angry_score = models.FloatField(default=0.0)
    disgusted_score = models.FloatField(default=0.0)
    fearful_score = models.FloatField(default=0.0)
    happy_score = models.FloatField(default=0.0)
    neutral_score = models.FloatField(default=0.0)
    sad_score = models.FloatField(default=0.0)
    surprised_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.emotion} - {self.stress_level}"

    @property
    def emotion_scores(self):
        return {
            'angry': self.angry_score,
            'disgusted': self.disgusted_score,
            'fearful': self.fearful_score,
            'happy': self.happy_score,
            'neutral': self.neutral_score,
            'sad': self.sad_score,
            'surprised': self.surprised_score,
        }


class ModelMetrics(models.Model):
    """Store ML model training metrics."""
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    training_date = models.DateTimeField(auto_now_add=True)
    epochs = models.IntegerField(default=0)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-training_date']

    def __str__(self):
        return f"{self.model_name} - Acc: {self.accuracy:.2%}"
