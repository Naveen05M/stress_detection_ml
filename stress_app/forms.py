"""
Forms for StressNet Registration
"""
from django import forms
from .models import DEPARTMENT_CHOICES


class EmployeeRegistrationForm(forms.Form):
    # Personal Info
    first_name  = forms.CharField(max_length=50,  label='First Name')
    last_name   = forms.CharField(max_length=50,  label='Last Name')
    email       = forms.EmailField(label='Email Address')

    # Employee Info
    employee_id  = forms.CharField(max_length=20, label='Employee ID')
    department   = forms.ChoiceField(choices=DEPARTMENT_CHOICES, label='Department')
    designation  = forms.CharField(max_length=100, required=False,
                                   label='Designation / Role')
    phone        = forms.CharField(max_length=15, required=False,
                                   label='Phone Number')
    date_of_join = forms.DateField(required=False, label='Date of Joining',
                                   widget=forms.DateInput(attrs={'type': 'date'}))

    # Profile Photo
    profile_photo = forms.ImageField(required=False, label='Profile Photo')

    # Account
    password  = forms.CharField(widget=forms.PasswordInput, label='Password',
                                min_length=6)
    password2 = forms.CharField(widget=forms.PasswordInput,
                                label='Confirm Password')

    def clean_email(self):
        from django.contrib.auth.models import User
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('This email is already registered.')
        return email

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get('password')
        p2 = cleaned.get('password2')
        if p1 and p2 and p1 != p2:
            raise forms.ValidationError('Passwords do not match.')
        return cleaned