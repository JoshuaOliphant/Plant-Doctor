# ABOUTME: Configuration for PlantDoc evaluation runs.
# ABOUTME: Defines paths, models, and evaluation parameters.

from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ClaudeModel(str, Enum):
    """Supported Claude models for evaluation."""
    HAIKU_45 = "claude-haiku-4-5-20251001"
    SONNET_45 = "claude-sonnet-4-5-20250929"


# Pricing per million tokens (input, output) in dollars
MODEL_PRICING: dict[ClaudeModel, tuple[float, float]] = {
    ClaudeModel.HAIKU_45: (1.0, 5.0),
    ClaudeModel.SONNET_45: (3.0, 15.0),
}


class EvalSettings(BaseSettings):
    """Settings for evaluation runs."""
    anthropic_api_key: str

    # Paths
    plantdoc_dataset_path: Path = Field(
        default=Path(__file__).parent.parent.parent / "data" / "PlantDoc-Dataset",
        description="Path to PlantDoc dataset root"
    )
    results_path: Path = Field(
        default=Path(__file__).parent / "results",
        description="Path to store evaluation results"
    )

    # Evaluation parameters
    max_images: int | None = Field(
        default=None,
        description="Maximum images to evaluate (None for all)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class EvalCase(BaseModel):
    """A single evaluation case: one image with its expected label."""
    image_path: Path
    expected_disease: str
    plant_species: str  # Extracted from disease label


class EvalResult(BaseModel):
    """Result of evaluating a single case."""
    case: EvalCase
    predicted_disease: str | None
    predicted_species: str | None
    confidence: float | None
    is_correct: bool
    processing_time_ms: int
    cost_cents: float
    model_used: str
    raw_response: dict | None = None
    error: str | None = None
