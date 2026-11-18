"""Resolve server resources independently of the terminal's working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resource_path(variable: str, default: str) -> Path:
    path = Path(os.environ.get(variable, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
