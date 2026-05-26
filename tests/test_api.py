from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "echo"
    assert "model_id" in body


def test_generate_returns_text() -> None:
    response = client.post(
        "/generate",
        json={"prompt": "Quels sont les signes d'alerte ?", "max_new_tokens": 16},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "echo"
    assert body["text"]
    assert "Prompt:" in body["text"]


def test_generate_rejects_empty_prompt() -> None:
    response = client.post("/generate", json={"prompt": ""})

    assert response.status_code == 422
