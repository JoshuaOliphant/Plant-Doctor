# ABOUTME: Multi-model plant disease classifier using PydanticAI.
# ABOUTME: Supports Claude, OpenAI GPT-4o, and Google Gemini for vision classification.

import os
import time
import json
import re
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

from .prompts import get_prompt, DISEASE_SYNONYMS, DISEASE_TERMS
from .config import EvalCase, EvalResult

# Load environment variables from .env file
load_dotenv()


class ModelProvider(str, Enum):
    """Supported model providers for plant classification."""
    # Anthropic Claude models
    CLAUDE_HAIKU = "anthropic:claude-haiku-4-5-20251001"
    CLAUDE_SONNET = "anthropic:claude-sonnet-4-5-20250929"

    # OpenAI GPT-5 family (latest)
    GPT_5_2 = "openai:gpt-5.2"
    GPT_5 = "openai:gpt-5"
    GPT_5_MINI = "openai:gpt-5-mini"
    GPT_5_NANO = "openai:gpt-5-nano"

    # OpenAI GPT-4 family (legacy but still supported)
    GPT_4O = "openai:gpt-4o"
    GPT_4O_MINI = "openai:gpt-4o-mini"
    GPT_4_1 = "openai:gpt-4.1"
    GPT_4_1_MINI = "openai:gpt-4.1-mini"
    GPT_4_1_NANO = "openai:gpt-4.1-nano"

    # Google Gemini 3 family (latest)
    GEMINI_3_PRO = "google-gla:gemini-3-pro-preview"
    GEMINI_3_FLASH = "google-gla:gemini-3-flash-preview"

    # Google Gemini 2 family
    GEMINI_2_FLASH = "google-gla:gemini-2.0-flash"
    GEMINI_2_PRO = "google-gla:gemini-2.0-pro-exp-02-05"

    # Google Gemini 1.5 family (legacy)
    GEMINI_1_5_FLASH = "google-gla:gemini-1.5-flash"
    GEMINI_1_5_PRO = "google-gla:gemini-1.5-pro"


# Pricing per million tokens (input, output) in dollars
MODEL_PRICING: dict[ModelProvider, tuple[float, float]] = {
    # Claude pricing
    ModelProvider.CLAUDE_HAIKU: (1.0, 5.0),
    ModelProvider.CLAUDE_SONNET: (3.0, 15.0),

    # OpenAI GPT-5 pricing (approximate)
    ModelProvider.GPT_5_2: (3.0, 12.0),
    ModelProvider.GPT_5: (2.5, 10.0),
    ModelProvider.GPT_5_MINI: (0.3, 1.2),
    ModelProvider.GPT_5_NANO: (0.1, 0.4),

    # OpenAI GPT-4 pricing (approximate)
    ModelProvider.GPT_4O: (2.5, 10.0),
    ModelProvider.GPT_4O_MINI: (0.15, 0.6),
    ModelProvider.GPT_4_1: (2.0, 8.0),
    ModelProvider.GPT_4_1_MINI: (0.4, 1.6),
    ModelProvider.GPT_4_1_NANO: (0.1, 0.4),

    # Gemini 3 pricing (approximate - preview pricing)
    ModelProvider.GEMINI_3_PRO: (1.0, 4.0),
    ModelProvider.GEMINI_3_FLASH: (0.5, 3.0),

    # Gemini 2 pricing
    ModelProvider.GEMINI_2_FLASH: (0.1, 0.4),
    ModelProvider.GEMINI_2_PRO: (1.25, 5.0),

    # Gemini 1.5 pricing
    ModelProvider.GEMINI_1_5_FLASH: (0.075, 0.3),
    ModelProvider.GEMINI_1_5_PRO: (1.25, 5.0),
}


class ClassificationResult(BaseModel):
    """Structured output for plant disease classification."""
    species: str = Field(description="The identified plant species")
    disease: str = Field(description="The identified disease or 'Healthy'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str | None = Field(default=None, description="Optional reasoning (for CoT)")


@dataclass
class MultiModelResult:
    """Result from multi-model classification."""
    case: EvalCase
    predicted_disease: str | None
    predicted_species: str | None
    confidence: float | None
    is_correct: bool
    processing_time_ms: int
    cost_cents: float
    model_used: str
    prompt_type: str
    reasoning: str | None = None
    raw_response: dict | None = None
    error: str | None = None


def load_image_bytes(image_path: Path) -> bytes:
    """Load image as bytes for PydanticAI BinaryContent."""
    return image_path.read_bytes()


def get_media_type(image_path: Path) -> str:
    """Determine media type from file extension."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/jpeg")


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)

    # Try to find standalone JSON object
    json_obj_match = re.search(r'\{[\s\S]*\}', text)
    if json_obj_match:
        text = json_obj_match.group(0)

    return json.loads(text.strip())


def check_disease_match(predicted: str, expected: str) -> bool:
    """Check if predicted disease matches expected label with synonym support."""
    if not predicted:
        return False

    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact match
    if pred_lower == exp_lower:
        return True

    # Check synonym matches
    for canonical, alternatives in DISEASE_SYNONYMS.items():
        if canonical in exp_lower:
            for alt in alternatives:
                if alt in pred_lower:
                    return True
        if canonical in pred_lower:
            for alt in alternatives:
                if alt in exp_lower:
                    return True

    # Check if key disease terms match
    for term in DISEASE_TERMS:
        if term in pred_lower and term in exp_lower:
            return True

    # Check if "healthy" matches healthy leaf classes
    if "healthy" in pred_lower:
        disease_indicators = [
            "blight", "spot", "rust", "scab", "virus", "mold",
            "rot", "mildew", "mosaic", "curl", "wilt"
        ]
        if "leaf" in exp_lower and not any(d in exp_lower for d in disease_indicators):
            return True

    # Check for early/late blight distinction
    if "early blight" in pred_lower and "early blight" in exp_lower:
        return True
    if "late blight" in pred_lower and "late blight" in exp_lower:
        return True

    return False


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: ModelProvider
) -> float:
    """Estimate API cost in cents."""
    input_price, output_price = MODEL_PRICING.get(model, (1.0, 5.0))
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return round((input_cost + output_cost) * 100, 4)


async def classify_image(
    case: EvalCase,
    model: ModelProvider,
    prompt_type: str = "baseline"
) -> MultiModelResult:
    """Classify a plant image using the specified model and prompt type.

    Args:
        case: The evaluation case with image path and expected disease
        model: The model provider to use
        prompt_type: One of "baseline", "hybrid", "cot", "structured"

    Returns:
        MultiModelResult with classification results
    """
    start = time.time()
    system_prompt = get_prompt(prompt_type)

    try:
        # Create agent with the specified model
        agent = Agent(
            model=model.value,
            system_prompt=system_prompt,
        )

        # Load image
        image_bytes = load_image_bytes(case.image_path)
        media_type = get_media_type(case.image_path)

        # Run classification
        result = await agent.run([
            "Classify this plant image.",
            BinaryContent(data=image_bytes, media_type=media_type),
        ])

        # Parse response
        response_text = result.output
        parsed = parse_json_response(response_text)

        processing_time_ms = int((time.time() - start) * 1000)

        # Extract token usage if available
        input_tokens = getattr(result, 'input_tokens', 0) or 1000  # Estimate if not available
        output_tokens = getattr(result, 'output_tokens', 0) or 100

        cost_cents = estimate_cost(input_tokens, output_tokens, model)

        predicted_disease = parsed.get("disease", "")
        predicted_species = parsed.get("species", "")
        confidence = parsed.get("confidence", 0.0)
        reasoning = parsed.get("reasoning")

        is_correct = check_disease_match(predicted_disease, case.expected_disease)

        return MultiModelResult(
            case=case,
            predicted_disease=predicted_disease,
            predicted_species=predicted_species,
            confidence=confidence,
            is_correct=is_correct,
            processing_time_ms=processing_time_ms,
            cost_cents=cost_cents,
            model_used=model.value,
            prompt_type=prompt_type,
            reasoning=reasoning,
            raw_response=parsed,
        )

    except Exception as e:
        processing_time_ms = int((time.time() - start) * 1000)
        return MultiModelResult(
            case=case,
            predicted_disease=None,
            predicted_species=None,
            confidence=None,
            is_correct=False,
            processing_time_ms=processing_time_ms,
            cost_cents=0.0,
            model_used=model.value,
            prompt_type=prompt_type,
            error=str(e),
        )


def classify_image_sync(
    case: EvalCase,
    model: ModelProvider,
    prompt_type: str = "baseline"
) -> MultiModelResult:
    """Synchronous wrapper for classify_image."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(classify_image(case, model, prompt_type))
