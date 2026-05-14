from django.apps import AppConfig
from pathlib import Path


class SegmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "segmentation"
    path = str(Path(__file__).resolve().parent)