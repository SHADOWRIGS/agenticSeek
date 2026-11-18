# Образ сервиса прогноза продаж: практика 7.4.4 (разбор — урок 7.4.2).
# Рядом должны лежать pyproject.toml, uv.lock и .python-version вашего проекта и каталог model/.
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1


FROM base AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


FROM base AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" MODEL_DIR=/app/model OMP_NUM_THREADS=1
COPY sales_service.py main.py ./
COPY model/ ./model/
ARG GIT_COMMIT=unknown
ARG MODEL_VERSION=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT \
      ml.model-version=$MODEL_VERSION
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=2s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
