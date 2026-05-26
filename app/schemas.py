from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    backend: str
    model_id: str


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_id: str
