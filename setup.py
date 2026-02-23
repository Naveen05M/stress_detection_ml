#!/usr/bin/env python
"""
One-shot setup script for Stress Detection project.
Run this after installing requirements.
"""
import os
import subprocess
import sys


def run(cmd, cwd=None):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  [WARNING] Command returned {result.returncode}")
    return result.returncode == 0


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("  StressGuard AI - Project Setup")
    print("=" * 60)

    os.chdir(base_dir)

    print("\n[1] Creating migrations...")
    run("python manage.py makemigrations stress_app")

    print("\n[2] Applying migrations...")
    run("python manage.py migrate")

    print("\n[3] Creating superuser (admin/admin)...")
    cmd = (
        "python -c \""
        "import django, os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','stress_project.settings'); "
        "django.setup(); "
        "from django.contrib.auth.models import User; "
        "User.objects.filter(username='admin').exists() or User.objects.create_superuser('admin','admin@stress.local','admin123')"
        "\""
    )
    run(cmd)

    print("\n[4] Collecting static files...")
    run("python manage.py collectstatic --noinput")

    print("\n[5] Verifying media directories...")
    os.makedirs("media/uploads", exist_ok=True)
    os.makedirs("ml_model", exist_ok=True)

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Prepare dataset:")
    print("     python dataset_utils/prepare_dataset.py --csv fer2013.csv --output dataset/")
    print("\n  2. Train the model:")
    print("     python ml_model/train_model.py --dataset_path dataset/")
    print("\n  3. Run the server:")
    print("     python manage.py runserver")
    print("\n  4. Open browser at:  http://127.0.0.1:8000")
    print("     Admin panel:       http://127.0.0.1:8000/admin")
    print("     Login:  admin / admin123")
    print("=" * 60)


if __name__ == '__main__':
    main()
