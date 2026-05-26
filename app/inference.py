import os
from typing import Protocol


class InferenceBackend(Protocol):
    name: str
    model_id: str

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        ...


class EchoBackend:
    """Lightweight deterministic backend used for local POC, CI and Docker smoke tests."""

    name = "echo"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        del temperature
        text = prompt.strip()
        if not text:
            return ""
        words = text.split()
        truncated = " ".join(words[:max_new_tokens])
        return (
            "POC fallback response. "
            "A production deployment should serve the LoRA adapter with vLLM. "
            f"Prompt: {truncated}"
        )


def build_backend() -> InferenceBackend:
    backend_name = os.getenv("INFERENCE_BACKEND", "echo").lower()
    model_id = os.getenv("MODEL_ID", "notebooks/qwen3-medical-dpo-lora")
    if backend_name == "echo":
        return EchoBackend(model_id=model_id)
    if backend_name == "vllm":
        raise RuntimeError(
            "INFERENCE_BACKEND=vllm is documented for production but is not bundled "
            "in this lightweight API image. See docs/vllm.md."
        )
    raise RuntimeError(f"Unsupported INFERENCE_BACKEND={backend_name!r}")
