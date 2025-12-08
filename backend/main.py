# PlantDoc PoC - FastAPI Backend
# Single endpoint for plant diagnosis using Claude Vision

from enum import Enum
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import anthropic
import time
import json
import base64
import re


class ClaudeModel(str, Enum):
    """Supported Claude models for plant diagnosis."""
    HAIKU_45 = "claude-haiku-4-5-20251001"
    SONNET_45 = "claude-sonnet-4-5-20250929"


# Pricing per million tokens (input, output) in dollars
MODEL_PRICING: dict[ClaudeModel, tuple[float, float]] = {
    ClaudeModel.HAIKU_45: (1.0, 5.0),
    ClaudeModel.SONNET_45: (3.0, 15.0),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    anthropic_api_key: str
    claude_model: ClaudeModel = Field(
        default=ClaudeModel.SONNET_45,
        description="Claude model to use for diagnosis"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

app = FastAPI(
    title="PlantDoc PoC",
    description="Plant diagnosis using Claude Vision",
    version="0.1.0"
)

# Allow CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# Pydantic models for request/response
class DiagnoseRequest(BaseModel):
    image: str  # base64 encoded image
    context: str | None = None  # Optional user-provided context

class SpeciesGuess(BaseModel):
    name: str
    confidence: float

class Issue(BaseModel):
    name: str
    confidence: float
    severity: str  # mild, moderate, severe
    visual_evidence: list[str]

class WeekPlan(BaseModel):
    week: int
    focus: str
    tasks: list[str]

class RecoveryPlan(BaseModel):
    duration_weeks: int
    prognosis: str  # good, fair, guarded
    weeks: list[WeekPlan]

class DiagnoseResponse(BaseModel):
    species_guess: SpeciesGuess
    issues: list[Issue]
    explanation: str
    immediate_actions: list[str]
    recovery_plan: RecoveryPlan
    confidence_notes: str
    processing_time_ms: int
    model_cost_cents: float
    model_used: str


class ClassifyResponse(BaseModel):
    """Fast classification response - identifies disease without full diagnosis."""
    species: str
    disease: str
    confidence: float
    severity: str  # none, mild, moderate, severe
    needs_diagnosis: bool  # True if user should request full diagnosis
    processing_time_ms: int
    model_cost_cents: float
    model_used: str

# GEPA-optimized classification prompt - fast, accurate disease identification
CLASSIFICATION_SYSTEM_PROMPT = """You are an AI Plant Disease Classifier specialized in diagnosing plant diseases through precise image analysis.

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
  "disease": "Specific disease name (Plant Species + Disease Type) or 'Healthy'",
  "confidence": 0.0-1.0,
  "severity": "none|mild|moderate|severe",
  "needs_diagnosis": true/false
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

5. needs_diagnosis Flag:
   - Set to true if disease detected and severity is moderate or severe
   - Set to false for healthy plants or mild issues

STRICT OUTPUT CONSTRAINTS:
- JSON format ONLY
- No supplementary text
- Precise, unambiguous diagnostic representation"""


# Balanced diagnosis prompt - combines GEPA learnings with houseplant focus
DIAGNOSIS_SYSTEM_PROMPT = """You are PlantDoc, an expert plant pathologist with decades of experience diagnosing both houseplant health issues and agricultural crop diseases. Analyze plant photos to provide accurate, actionable diagnoses.

## Your Process
1. **Identify the plant species** - Look at leaf shape, growth pattern, texture, venation. Common plants include:
   - **Houseplants**: Pothos, Monstera, Fiddle Leaf Fig, Snake Plant, Peace Lily, Philodendron, ZZ Plant, Spider Plant, Rubber Plant, Calathea
   - **Outdoor/Crops**: Tomato, Pepper, Apple, Grape, Citrus, Herbs, Vegetables
   - Note uncertainty if unclear.

2. **Analyze visible symptoms systematically**:
   - **Leaf spots**: size, color (brown, black, yellow), shape (circular, angular), borders
   - **Discoloration**: yellowing patterns, browning, pale patches
   - **Texture changes**: crispy edges, mushy stems, wilting, drooping
   - **Growth issues**: leggy growth, sparse foliage, stunted development
   - **Environmental signs**: soil moisture, root visibility, pot condition
   - **Pests**: visible bugs, webbing, honeydew, stippling, scale
   - **Disease signs**: powdery coating, fuzzy growth, lesions, rot

3. **Consider the most likely causes** - For houseplants, prioritize:
   - Overwatering (yellow leaves, mushy stems, root rot)
   - Underwatering (crispy edges, drooping, dry soil)
   - Low light (leggy growth, pale leaves, leaning)
   - Low humidity (brown tips, crispy edges)
   - Nutrient deficiency (yellowing patterns, stunted growth)
   - Pests (visible bugs, webbing, stippling)

4. **For disease symptoms, be specific**:
   - **Target spots with concentric rings** = Fungal leaf spot, Alternaria
   - **White powdery coating** = Powdery Mildew
   - **Black/brown spots with defined edges** = Bacterial or fungal leaf spot
   - **Fuzzy gray/white growth** = Botrytis, Downy Mildew
   - **Soft, mushy tissue** = Root rot, bacterial soft rot
   - **Yellow mottling patterns** = Viral infection or nutrient issue

5. **Provide specific treatment recommendations** appropriate for home plant care

## Output Format
You MUST respond with valid JSON matching this exact schema:
{
  "species_guess": {
    "name": "Common name (Scientific name)",
    "confidence": 0.0-1.0
  },
  "issues": [
    {
      "name": "Issue name",
      "confidence": 0.0-1.0,
      "severity": "mild|moderate|severe",
      "visual_evidence": ["specific thing you see in the image"]
    }
  ],
  "explanation": "2-3 sentences explaining what's happening in plain, friendly English. Connect the symptoms to the cause.",
  "immediate_actions": ["Specific action to take today", "Another immediate step"],
  "recovery_plan": {
    "duration_weeks": 2-6,
    "prognosis": "good|fair|guarded",
    "weeks": [
      {
        "week": 1,
        "focus": "Brief description of this week's goal",
        "tasks": ["Specific actionable task", "Another task"]
      }
    ]
  },
  "confidence_notes": "Any important caveats about photo quality, angle, or diagnostic uncertainty"
}

## Critical Rules
- **Be honest about uncertainty** - If you can't tell from the photo, say so
- **Don't invent issues** - Only diagnose what you can actually see evidence of
- **Explain the WHY** - Connect symptoms to causes so users understand
- **Be specific** - "Water when top 2 inches of soil are dry" not "water appropriately"
- **Prioritize** - List the most likely/severe issue first
- **If the plant looks healthy, say so!** - Don't manufacture problems
- **Look for subtle symptoms** - Early stage issues may show minimal signs
- **Consider photo quality** - Note if lighting, angle, or resolution limits diagnosis
- **Think about common issues first** - Overwatering is more common than rare diseases

## Severity Guidelines
- **mild**: Cosmetic issues, early-stage problems, easy recovery expected
- **moderate**: Affecting plant health, needs attention within days, good recovery possible
- **severe**: Life-threatening to plant, immediate action needed, recovery uncertain

## Treatment Considerations
- **Watering issues**: Adjust frequency, check drainage, repot if needed
- **Light issues**: Move to better location, add grow lights if needed
- **Humidity issues**: Group plants, use pebble tray, mist regularly
- **Pest control**: Isolate plant, neem oil, insecticidal soap, manual removal
- **Fungal diseases**: Improve air circulation, reduce humidity, fungicide if severe
- **Root rot**: Repot with fresh soil, trim affected roots, reduce watering"""

def estimate_cost(input_tokens: int, output_tokens: int, model: ClaudeModel) -> float:
    """Estimate API cost in cents for the given model."""
    input_price, output_price = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return round((input_cost + output_cost) * 100, 4)  # Convert to cents

def parse_diagnosis_response(text: str) -> dict:
    """Parse JSON from Claude's response, handling potential markdown code blocks"""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)

    # Clean up and parse
    text = text.strip()
    return json.loads(text)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PlantDoc PoC"}

@app.get("/health")
async def health():
    """Health check for deployment platforms"""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: DiagnoseRequest):
    """
    Fast plant disease classification using Haiku.

    Returns species, disease, and severity without full diagnosis.
    Use this endpoint first, then call /diagnose if needs_diagnosis is true.

    - **image**: Base64 encoded JPEG or PNG image
    - **context**: Optional text describing the user's concern
    """
    start = time.time()

    # Validate base64 image
    try:
        base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Build the message content
    user_content = []

    if request.context:
        user_content.append({
            "type": "text",
            "text": f"Context from plant owner: \"{request.context}\""
        })

    user_content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": request.image
        }
    })
    user_content.append({
        "type": "text",
        "text": "Classify this plant image."
    })

    try:
        # Always use Haiku for fast classification
        model = ClaudeModel.HAIKU_45
        response = client.messages.create(
            model=model.value,
            max_tokens=500,
            system=CLASSIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )

        # Parse the JSON response
        result = parse_diagnosis_response(response.content[0].text)

        # Calculate timing and cost
        processing_time_ms = int((time.time() - start) * 1000)
        cost_cents = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )

        return ClassifyResponse(
            species=result.get("species", "Unknown"),
            disease=result.get("disease", "Unknown"),
            confidence=result.get("confidence", 0.0),
            severity=result.get("severity", "none"),
            needs_diagnosis=result.get("needs_diagnosis", False),
            processing_time_ms=processing_time_ms,
            model_cost_cents=cost_cents,
            model_used=model.value
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse classification response: {str(e)}"
        )
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Claude API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Diagnose plant health issues from a photo.

    - **image**: Base64 encoded JPEG or PNG image
    - **context**: Optional text describing the user's concern
    """
    start = time.time()

    # Validate base64 image
    try:
        # Check if it's valid base64
        base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Build the message content
    user_content = []

    # Add user context if provided
    if request.context:
        user_content.append({
            "type": "text",
            "text": f"The plant owner says: \"{request.context}\""
        })

    # Add the instruction and image
    user_content.append({
        "type": "text",
        "text": "Please diagnose this plant and respond with the JSON schema specified in your instructions:"
    })
    user_content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",  # Works for PNG too
            "data": request.image
        }
    })

    try:
        # Call Claude Vision with configured model
        model = settings.claude_model
        response = client.messages.create(
            model=model.value,
            max_tokens=2000,
            system=DIAGNOSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )

        # Parse the JSON response
        result = parse_diagnosis_response(response.content[0].text)

        # Calculate timing and cost
        processing_time_ms = int((time.time() - start) * 1000)
        cost_cents = estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )

        # Build and return response
        return DiagnoseResponse(
            species_guess=SpeciesGuess(**result["species_guess"]),
            issues=[Issue(**issue) for issue in result["issues"]],
            explanation=result["explanation"],
            immediate_actions=result["immediate_actions"],
            recovery_plan=RecoveryPlan(
                duration_weeks=result["recovery_plan"]["duration_weeks"],
                prognosis=result["recovery_plan"]["prognosis"],
                weeks=[WeekPlan(**week) for week in result["recovery_plan"]["weeks"]]
            ),
            confidence_notes=result.get("confidence_notes", ""),
            processing_time_ms=processing_time_ms,
            model_cost_cents=cost_cents,
            model_used=model.value
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse diagnosis response: {str(e)}"
        )
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Claude API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
