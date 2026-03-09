from django.contrib import admin
from .models import Employee, StressRecord, Notification


@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ['employee_id', 'full_name', 'department',
                    'designation', 'status', 'registered_at']
    list_filter   = ['status', 'department']
    search_fields = ['employee_id', 'user__first_name',
                     'user__last_name', 'user__email']
    readonly_fields = ['registered_at', 'approved_at', 'last_active']


@admin.register(StressRecord)
class StressRecordAdmin(admin.ModelAdmin):
    list_display = ['employee', 'emotion', 'stress_level',
                    'confidence', 'source', 'timestamp']
    list_filter   = ['stress_level', 'emotion', 'source']
    search_fields = ['employee__user__first_name',
                     'employee__employee_id']
    readonly_fields = ['timestamp']


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['message', 'is_read', 'created_at']
    list_filter  = ['is_read']