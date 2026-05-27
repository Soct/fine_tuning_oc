import dataclasses
import os
import time
from collections.abc import Mapping
from typing import Any, Protocol

import httpx


@dataclasses.dataclass
class GenerateResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None


class InferenceBackend(Protocol):
    name: str
    model_id: str

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerateResult:
        ...


class EchoBackend:
    """Lightweight deterministic backend used for local POC, CI and Docker smoke tests."""

    name = "echo"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerateResult:
        del temperature
        text = prompt.strip()
        if not text:
            return GenerateResult(text="")
        words = text.split()
        truncated = " ".join(words[:max_new_tokens])
        return GenerateResult(
            text=(
                "POC fallback response. "
                "A production deployment should serve the LoRA adapter with vLLM. "
                f"Prompt: {truncated}"
            )
        )


def build_http_client(base_url: str, timeout: float, headers: Mapping[str, str]) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=dict(headers))


class VllmBackend:
    """OpenAI-compatible vLLM client backend used by the FastAPI service."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        system_prompt: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "Tu es un assistant medical expert. Reponds de maniere claire, factuelle et structuree. "
            "Si la question est en anglais, reponds en anglais."
        )
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = build_http_client(base_url=self.base_url, timeout=timeout, headers=headers)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Invalid response from vLLM: missing choices")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Invalid response from vLLM: missing message")

        # content peut être null (Qwen3 thinking mode) : on tombe alors sur reasoning_content
        content = message.get("content") or message.get("reasoning_content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(text_parts).strip()
        raise RuntimeError(
            f"Invalid response from vLLM: unsupported content format "
            f"(type={type(content).__name__}, value={content!r})"
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerateResult:
        t0 = time.perf_counter()
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        elapsed = time.perf_counter() - t0
        response.raise_for_status()
        payload = response.json()
        text = self._extract_text(payload)

        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        tps = round(completion_tokens / elapsed, 1) if completion_tokens and elapsed > 0 else None

        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tps,
        )


def build_backend() -> InferenceBackend:
    backend_name = os.getenv("INFERENCE_BACKEND", "echo").lower()
    model_id = os.getenv("MODEL_ID", "notebooks/qwen3-medical-dpo-lora")
    if backend_name == "echo":
        return EchoBackend(model_id=model_id)
    if backend_name == "vllm":
        return VllmBackend(
            model_id=model_id,
            base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001"),
            api_key=os.getenv("VLLM_API_KEY"),
            timeout=float(os.getenv("VLLM_TIMEOUT_SECONDS", "120")),
            system_prompt=os.getenv("SYSTEM_PROMPT"),
        )
    raise RuntimeError(f"Unsupported INFERENCE_BACKEND={backend_name!r}")
