import json
import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Body, FastAPI, Request
from fastapi.responses import Response

from app.inference import build_backend
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse, UsageStats

_GENERATE_EXAMPLES = {
    "question_ouverte": {
        "summary": "Question médicale ouverte",
        "value": {
            "prompt": "Quels sont les symptômes les plus courants d'une crise d'asthme ?",
            "max_new_tokens": 2048,
            "temperature": 0.3,
        },
    },
    "qcm": {
        "summary": "QCM médical",
        "value": {
            "prompt": (
                "Question : Quel médicament est le traitement de première ligne "
                "pour l'hypertension artérielle essentielle non compliquée ?\n\n"
                "A. Amoxicilline\nB. Amlodipine\nC. Méthotrexate\nD. Furosémide\n\n"
                "Réponds uniquement par la lettre de la bonne réponse, "
                "suivie d'une courte justification."
            ),
            "max_new_tokens": 2048,
            "temperature": 0.1,
        },
    },
}


logger = logging.getLogger("inference_api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_json(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=True))


def create_app() -> FastAPI:
    backend = build_backend()
    app = FastAPI(title="Fine-tuning OC inference API", version="0.1.0")
    app.state.backend = backend

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
        current_backend = app.state.backend
        return HealthResponse(status="ok", backend=current_backend.name, model_id=current_backend.model_id)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(payload: GenerateRequest = Body(openapi_examples=_GENERATE_EXAMPLES)) -> GenerateResponse:
        current_backend = app.state.backend
        result = current_backend.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        log_json(
            "generation",
            backend=current_backend.name,
            model_id=current_backend.model_id,
            prompt_chars=len(payload.prompt),
            output_chars=len(result.text),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        usage = (
            UsageStats(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                tokens_per_second=result.tokens_per_second,
            )
            if result.prompt_tokens is not None and result.completion_tokens is not None
            else None
        )
        return GenerateResponse(
            text=result.text,
            backend=current_backend.name,
            model_id=current_backend.model_id,
            usage=usage,
        )

    return app


app = create_app()
