# ABOUTME: Core diagnosis task that calls Claude API for plant disease classification.
# ABOUTME: Wraps the main.py diagnosis logic for use in evaluation framework.

import time
import json
import re

import anthropic

from .config import ClaudeModel, MODEL_PRICING, EvalCase, EvalResult
from .dataset_loader import load_image_as_base64
from .prompts import BASELINE_PROMPT, HYBRID_PROMPT, DISEASE_SYNONYMS, DISEASE_TERMS


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


# GEPA-optimized prompt - evolved with pattern-based feedback
OPTIMIZED_SYSTEM_PROMPT = """You are a plant disease classifier. Analyze plant images and identify the species and any visible disease.

Respond with ONLY valid JSON in this exact format:
{
  "species": "Plant name",
  "disease": "Specific disease name or 'Healthy'",
  "confidence": 0.0-1.0
}

Guidelines:

RESPONSE FORMAT:
- Output ONLY the JSON, no additional text
- Use valid JSON syntax

DISEASE IDENTIFICATION:
- Be specific and include the plant species in disease names (e.g., "Apple scab" not just "Scab")
- Use "Healthy" only when no disease symptoms are visible
- Do NOT classify viral symptoms (yellowing, mottling, curling, distortion) as 'Healthy'

SYMPTOM DIFFERENTIATION:
- Bacterial diseases: Look for water-soaked lesions, halos around spots, translucent appearance
- Fungal diseases: Look for concentric rings, defined/sharp borders, powdery coating, or fuzzy growth
- Gray leaf spot: Rectangular lesions positioned between leaf veins (distinct from blight or rust patterns)
- Blight: Shows tissue death and rapid spread; mold shows fuzzy growth
- Rust: Orange/brown pustules, often on leaf undersides
- Scab: Dark, raised, roughened lesions with irregular borders

VERIFICATION:
- Verify the plant species BEFORE assigning disease names - never mix diseases from different plant species
- Consider both houseplants AND agricultural crops equally
- Match disease to the correct plant species

CONFIDENCE:
- Set confidence based on symptom clarity and certainty of identification
- Lower confidence (0.5-0.7) when symptoms are ambiguous or multiple diseases are possible
- Higher confidence (0.8-1.0) when symptoms are clear and definitive"""


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
    Uses expanded synonym mappings from prompts.py for comprehensive matching.
    """
    if not predicted:
        return False

    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact match
    if pred_lower == exp_lower:
        return True

    # Check synonym matches using expanded DISEASE_SYNONYMS from prompts.py
    for canonical, alternatives in DISEASE_SYNONYMS.items():
        if canonical in exp_lower:
            for alt in alternatives:
                if alt in pred_lower:
                    return True
        if canonical in pred_lower:
            for alt in alternatives:
                if alt in exp_lower:
                    return True

    # Check if key disease terms match using DISEASE_TERMS from prompts.py
    for term in DISEASE_TERMS:
        if term in pred_lower and term in exp_lower:
            return True

    # Check if "healthy" matches healthy leaf classes
    if "healthy" in pred_lower:
        # Classes like "Apple leaf", "Tomato leaf" without disease = healthy
        disease_indicators = [
            "blight", "spot", "rust", "scab", "virus", "mold",
            "rot", "mildew", "mosaic", "curl", "wilt", "canker"
        ]
        if "leaf" in exp_lower and not any(d in exp_lower for d in disease_indicators):
            return True

    # Check for early/late blight distinction
    if "early blight" in pred_lower and "early blight" in exp_lower:
        return True
    if "late blight" in pred_lower and "late blight" in exp_lower:
        return True

    return False


def analyze_failure_pattern(predicted: str, expected: str) -> str:
    """Analyze why a prediction failed and return structured feedback for GEPA.

    Returns pattern-based feedback that helps GEPA learn generalizable improvements.
    """
    if not predicted:
        return "NO_PREDICTION: Model failed to produce a disease classification"

    pred_lower = predicted.lower()
    exp_lower = expected.lower()

    # Pattern 1: Viral symptoms missed (called healthy)
    if "healthy" in pred_lower and "virus" in exp_lower:
        return "VIRAL_MISSED: Yellow mottling, leaf curling, and mosaic patterns indicate viral infection, not healthy plant. Viral symptoms are often subtle - look for uneven yellowing, leaf distortion, and stunted growth."

    # Pattern 2: Bacterial vs fungal confusion
    if "bacterial" in exp_lower and ("blight" in pred_lower or "fungal" in pred_lower):
        return "BACTERIAL_VS_FUNGAL: Bacterial spots have water-soaked margins and may show yellow halos. Fungal spots often have concentric rings or defined borders. Look for bacterial ooze in humid conditions."

    # Pattern 3: Wrong species attribution
    if any(crop in pred_lower for crop in ["potato", "coffee", "wheat", "citrus"]) and \
       any(crop in exp_lower for crop in ["tomato", "corn", "pepper", "bell"]):
        return "SPECIES_CONFUSION: Verify the plant species before diagnosing. Similar symptoms can occur across species, but the disease name should match the actual plant in the image."

    # Pattern 4: Similar diseases on same crop confused
    if "tomato" in exp_lower and "tomato" in pred_lower:
        if "early blight" in pred_lower and "late blight" in exp_lower:
            return "BLIGHT_DISTINCTION: Early blight shows target-like concentric rings on older leaves. Late blight shows water-soaked lesions that turn brown/black, often starting at leaf edges, with white mold underneath in humid conditions."
        if "septoria" in pred_lower and "bacterial" in exp_lower:
            return "SPOT_TYPE: Septoria has small circular spots with dark borders and light centers. Bacterial spot has irregular, water-soaked spots that may coalesce."

    # Pattern 5: Mold/mildew confusion
    if "mold" in exp_lower and "blight" in pred_lower:
        return "MOLD_VS_BLIGHT: Leaf mold shows fuzzy growth (usually on leaf undersides). Blight causes tissue death and browning. Look for fungal growth texture."

    # Pattern 6: Naming convention mismatch
    if "leaf spot" in exp_lower and "spot" in pred_lower:
        return "NAMING_CONVENTION: Use the format '[Plant] leaf spot' (e.g., 'Bell pepper leaf spot') to match expected naming conventions."

    # Pattern 7: Gray leaf spot specifics
    if "gray leaf spot" in exp_lower and ("blight" in pred_lower or "rust" in pred_lower):
        return "GRAY_LEAF_SPOT: Gray leaf spot shows rectangular lesions between leaf veins (especially on corn). Distinct from blight (larger irregular areas) and rust (pustules)."

    # Default: general mismatch
    return f"MISMATCH: Predicted '{predicted}' but expected '{expected}'. Review visual symptoms more carefully."
