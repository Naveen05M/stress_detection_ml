from django.contrib import admin
from .models import StressRecord, ModelMetrics


@admin.register(StressRecord)
class StressRecordAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'emotion', 'stress_level', 'confidence', 'source')
    list_filter = ('stress_level', 'emotion', 'source')
    search_fields = ('emotion', 'stress_level')
    ordering = ('-timestamp',)


@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'accuracy', 'recall', 'f1_score', 'training_date')
    ordering = ('-training_date',)
