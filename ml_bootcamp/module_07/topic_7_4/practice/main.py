"""Точка входа сервиса: uvicorn main:app. Каталог модели задаётся переменной окружения MODEL_DIR."""

import os
from pathlib import Path

from sales_service import create_app

app = create_app(Path(os.environ.get("MODEL_DIR", "model")))
