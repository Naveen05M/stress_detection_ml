"""
Django settings for Stress Detection in IT Professionals project.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-stress-detection-it-professionals-2024-change-in-production'

DEBUG = True

ALLOWED_HOSTS = ['*', 'localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'stress_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'stress_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'stress_project.wsgi.application'

# Database - SQLite (default), can switch to MySQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# MySQL configuration (uncomment to use MySQL instead of SQLite)
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'stress_detection_db',
#         'USER': 'root',
#         'PASSWORD': 'your_mysql_password',
#         'HOST': 'localhost',
#         'PORT': '3306',
#     }
# }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'stress_app' / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ML Model path
MODEL_PATH = BASE_DIR / 'ml_model' / 'stress_model.h5'
HAARCASCADE_PATH = BASE_DIR / 'ml_model' / 'haarcascade_frontalface_default.xml'

# Emotion to Stress Mapping
EMOTION_STRESS_MAP = {
    'angry': 'High',
    'disgusted': 'High',
    'fearful': 'High',
    'happy': 'Low',
    'neutral': 'Low',
    'sad': 'Medium',
    'surprised': 'Medium',
}

EMOTION_CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
