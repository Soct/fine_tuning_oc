FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND=echo \
    MODEL_ID=notebooks/qwen3-medical-dpo-lora

WORKDIR /app

COPY deployment ./deployment
RUN uv sync --project deployment --no-dev

COPY app ./app

EXPOSE 8000

CMD ["uv", "run", "--project", "deployment", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
