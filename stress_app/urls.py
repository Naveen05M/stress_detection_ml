from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_detect, name='upload_detect'),
    path('live/', views.live_detect, name='live_detect'),
    path('live/frame/', views.live_frame, name='live_frame'),
    path('history/', views.history, name='history'),
    path('history/delete/<int:record_id>/', views.delete_record, name='delete_record'),
    path('api/timeline/', views.api_stress_timeline, name='api_timeline'),
]
