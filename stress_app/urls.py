from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('',              views.login_view,    name='login_view'),
    path('register/',     views.register_view, name='register_view'),
    path('logout/',       views.logout_view,   name='logout_view'),

    # Employee
    path('dashboard/',    views.user_dashboard, name='user_dashboard'),
    path('live/',         views.user_live,      name='user_live'),
    path('live/frame/',   views.user_live_frame, name='live_frame'),
    path('history/',      views.user_history,   name='user_history'),
    path('profile/',      views.user_profile,   name='user_profile'),

    # Admin panel
    path('admin-panel/',                              views.admin_dashboard,       name='admin_dashboard'),
    path('admin-panel/employees/',                    views.admin_employees,       name='admin_employees'),
    path('admin-panel/employees/<int:emp_id>/',       views.admin_employee_detail, name='admin_employee_detail'),
    path('admin-panel/employees/<int:emp_id>/approve/', views.admin_approve,       name='admin_approve'),
    path('admin-panel/employees/<int:emp_id>/reject/',  views.admin_reject,        name='admin_reject'),
    path('admin-panel/realtime/',                     views.admin_realtime,        name='admin_realtime'),
    path('admin-panel/notifications/',                views.admin_notifications,   name='admin_notifications'),

    # Admin APIs
    path('api/employees/live/',                       views.api_all_employees_live,  name='api_employees_live'),
    path('api/employees/<int:emp_id>/records/',       views.api_employee_records,    name='api_employee_records'),
]
