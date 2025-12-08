# ABOUTME: Core diagnosis task that calls Claude API for plant disease classification.
# ABOUTME: Wraps the main.py diagnosis logic for use in evaluation framework.

import time
import json
import re

import anthropic

from .config import ClaudeModel, MODEL_PRICING, EvalCase, EvalResult
from .dataset_loader import load_image_as_base64


# Few-shot prompt for classification eval - teaches format without constraining to dataset
EVAL_SYSTEM_PROMPT = """You are a plant disease classifier. Analyze plant images and identify the species and any visible disease.

Respond with ONLY valid JSON in this exact format:
{
  "species": "Plant name",
  "disease": "Specific disease name or 'Healthy'",
  "confidence": 0.0-1.0
}

## Examples

Example 1 - Apple with fungal disease:
{
  "species": "Apple",
  "disease": "Apple Scab",
  "confidence": 0.85
}

Example 2 - Tomato with bacterial infection:
{
  "species": "Tomato",
  "disease": "Tomato bacterial spot",
  "confidence": 0.78
}

Example 3 - Corn with fungal leaf disease:
{
  "species": "Corn",
  "disease": "Corn rust",
  "confidence": 0.92
}

Example 4 - Healthy plant with no disease:
{
  "species": "Strawberry",
  "disease": "Healthy",
  "confidence": 0.88
}

Example 5 - Potato with late-stage blight:
{
  "species": "Potato",
  "disease": "Potato late blight",
  "confidence": 0.81
}

## Guidelines
- Include the plant name in the disease when relevant (e.g., "Apple rust" not just "Rust")
- Be specific: "Tomato early blight" not "Blight"
- Use "Healthy" if no disease is visible
- Output ONLY the JSON, no additional text"""


# GEPA-optimized prompt - evolved through reflective mutation for higher accuracy
OPTIMIZED_SYSTEM_PROMPT = """You are an AI Plant Disease Classifier specialized in diagnosing plant diseases through precise image analysis.

TASK OBJECTIVES:
- Accurately identify plant species from input images
- Detect and classify specific plant diseases with high precision
- Provide a confident and accurate diagnostic assessment

DOMAIN-SPECIFIC KNOWLEDGE:
1. Common Plant Disease Patterns:
   - Apple Trees Specific Diseases:
     * Apple Scab Leaf: Characterized by dark, scaly lesions on leaves
     * Apple Cedar Rust: Orange-brown spots, often with distinct patterns
     * Distinguishing between similar diseases critical for accurate diagnosis

2. Diagnostic Criteria:
   - Examine leaf surface for:
     * Color variations
     * Spot patterns
     * Texture changes
     * Lesion characteristics
     * Discoloration edges and types

3. Confidence Assessment Factors:
   - Visual symptom clarity
   - Distinctiveness of disease markers
   - Comprehensiveness of observable symptoms

INPUT FORMAT:
- High-resolution image of plant leaf/affected plant part
- Optional: Suggested plant species name

OUTPUT REQUIREMENTS:
Respond ONLY with JSON in this structured format:
{
  "species": "Exact plant species name",
  "disease": "Specific disease name (Plant Species + Disease Type)",
  "confidence": 0.0-1.0
}

CRITICAL GUIDELINES:
1. Disease Naming Protocol:
   - ALWAYS prefix disease name with plant species
   - Use most specific disease identification possible
   - Prioritize accuracy over generalization

2. Confidence Scoring Methodology:
   - Decimal scale (0.0-1.0)
   - Score based on:
     * Symptom clarity
     * Matching against known disease patterns
     * Comprehensiveness of observable indicators

3. Health Assessment Principles:
   - Default to "Healthy" if NO definitive disease markers
   - Distinguish between natural variation and pathological changes

4. Diagnostic Strategy:
   - Holistic image examination
   - Cross-reference against comprehensive plant disease database
   - Consider subtle, nuanced visual indicators

STRICT OUTPUT CONSTRAINTS:
- JSON format ONLY
- No supplementary text
- Precise, unambiguous diagnostic representation"""


def parse_classification_response(text: str) -> dict:
    """Parse JSON from Claude's response."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)
    return json.loads(text.strip())


def estimate_cost(input_tokens: int, output_tokens: int, model: ClaudeModel) -> float:
    """Estimate API cost in cents."""
    input_price, output_price = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return round((input_cost + output_cost) * 100, 4)


def run_diagnosis(
    case: EvalCase,
    client: anthropic.Anthropic,
    model: ClaudeModel,
    use_optimized: bool = False
) -> EvalResult:
    """Run diagnosis on a single image and return the result."""
    start = time.time()

    # Select prompt based on optimization flag
    system_prompt = OPTIMIZED_SYSTEM_PROMPT if use_optimized else EVAL_SYSTEM_PROMPT

    try:
        # Load image
        image_b64 = load_image_as_base64(case.image_path)

        # Call Claude
        response = client.messages.create(
            model=model.value,
            max_tokens=500,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Classify this plant image."
                    }
                ]
            }]
        )

        # Parse response
        result = parse_classification_response(response.content[0].text)
        processing_time_ms = int((time.time() - start) * 1000)
        cost_cents = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )

        predicted_disease = result.get("disease", "")
        predicted_species = result.get("species", "")
        confidence = result.get("confidence", 0.0)

        # Check if prediction matches expected disease
        is_correct = check_disease_match(predicted_disease, case.expected_disease)

        return EvalResult(
            case=case,
            predicted_disease=predicted_disease,
            predicted_species=predicted_species,
            confidence=confidence,
            is_correct=is_correct,
            processing_time_ms=processing_time_ms,
            cost_cents=cost_cents,
            model_used=model.value,
            raw_response=result
        )

    except Exception as e:
        processing_time_ms = int((time.time() - start) * 1000)
        return EvalResult(
            case=case,
            predicted_disease=None,
            predicted_species=None,
            confidence=None,
            is_correct=False,
            processing_time_ms=processing_time_ms,
            cost_cents=0.0,
            model_used=model.value,
            error=str(e)
        )


def check_disease_match(predicted: str, expected: str) -> bool:
    """Check if predicted disease matches expected label.

    Uses fuzzy matching since Claude may phrase things differently.
    """
    if not predicted:
        return False

    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact match
    if pred_lower == exp_lower:
        return True

    # Check if key disease terms match
    disease_terms = [
        "scab", "rust", "blight", "bacterial spot", "mosaic virus",
        "yellow virus", "mold", "septoria", "powdery mildew", "black rot",
        "gray leaf spot", "healthy", "leaf spot"
    ]

    for term in disease_terms:
        if term in pred_lower and term in exp_lower:
            return True

    # Check if "healthy" matches healthy leaf classes
    if "healthy" in pred_lower:
        # Classes like "Apple leaf", "Tomato leaf" without disease = healthy
        if "leaf" in exp_lower and not any(
            d in exp_lower for d in ["blight", "spot", "rust", "scab", "virus", "mold", "rot", "mildew"]
        ):
            return True

    # Check for early/late blight distinction
    if "early blight" in pred_lower and "early blight" in exp_lower:
        return True
    if "late blight" in pred_lower and "late blight" in exp_lower:
        return True

    return False
