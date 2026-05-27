from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def echo_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("INFERENCE_BACKEND", "echo")
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_returns_status(echo_client: TestClient) -> None:
    response = echo_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "echo"
    assert "model_id" in body


def test_generate_returns_text(echo_client: TestClient) -> None:
    response = echo_client.post(
        "/generate",
        json={"prompt": "Quels sont les signes d'alerte ?", "max_new_tokens": 16},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "echo"
    assert body["text"]
    assert "Prompt:" in body["text"]


def test_generate_rejects_empty_prompt(echo_client: TestClient) -> None:
    response = echo_client.post("/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_vllm_backend_proxies_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = request.read().decode("utf-8")
        assert "qwen3-medical-dpo-lora" in payload
        assert "Question test" in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Reponse medicale vLLM"
                        }
                    }
                ]
            },
        )

    def fake_build_http_client(base_url: str, timeout: float, headers: dict[str, str]) -> httpx.Client:
        del timeout, headers
        transport = httpx.MockTransport(handler)
        return httpx.Client(base_url=base_url, transport=transport)

    monkeypatch.setenv("INFERENCE_BACKEND", "vllm")
    monkeypatch.setenv("MODEL_ID", "notebooks/qwen3-medical-dpo-lora")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8001")
    monkeypatch.setattr("app.inference.build_http_client", fake_build_http_client)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"prompt": "Question test", "max_new_tokens": 32, "temperature": 0.2},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "vllm"
    assert body["text"] == "Reponse medicale vLLM"
