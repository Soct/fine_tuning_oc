import json
import logging
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response

from app.inference import build_backend
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse


logger = logging.getLogger("inference_api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

backend = build_backend()
app = FastAPI(title="Fine-tuning OC inference API", version="0.1.0")


def log_json(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=True))


@app.middleware("http")
async def request_response_logger(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    log_json(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json(
            "response",
            request_id=request_id,
            status_code=500,
            duration_ms=duration_ms,
            error=exc.__class__.__name__,
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["x-request-id"] = request_id
    log_json(
        "response",
        request_id=request_id,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend=backend.name, model_id=backend.model_id)


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    text = backend.generate(
        prompt=payload.prompt,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )
    log_json(
        "generation",
        backend=backend.name,
        model_id=backend.model_id,
        prompt_chars=len(payload.prompt),
        output_chars=len(text),
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )
    return GenerateResponse(text=text, backend=backend.name, model_id=backend.model_id)
