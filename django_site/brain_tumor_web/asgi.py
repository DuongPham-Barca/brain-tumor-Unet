"""ASGI config for brain_tumor_web project."""

import os

from django.core.asgi import get_asgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brain_tumor_web.settings")

application = get_asgi_application()