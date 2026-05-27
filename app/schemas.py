from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float | None = None


class GenerateResponse(BaseModel):
    text: str
    backend: str
    model_id: str
    usage: UsageStats | None = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_id: str
