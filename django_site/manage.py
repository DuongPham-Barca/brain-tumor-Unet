#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

from pathlib import Path
import os
import sys


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brain_tumor_web.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()