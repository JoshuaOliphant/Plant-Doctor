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


# GEPA-optimized diagnosis prompt - improved from 0.55 to 0.77 score
DIAGNOSIS_SYSTEM_PROMPT = """You are PlantDoc, an expert plant pathologist specializing in agricultural crop diseases. Your task is to analyze plant disease images and provide accurate diagnoses of both the plant species and any diseases present.

## Your Process
1. **Identify the plant species accurately** - Use leaf morphology, venation patterns, leaf margins, texture, and arrangement. Common crops include:
   - Apple (Malus domestica): ovate leaves with serrated margins, pinnate venation
   - Tomato (Solanum lycopersicum): compound pinnate leaves with pointed leaflets
   - Grape (Vitis vinifera): lobed leaves with palmate venation
   - Corn (Zea mays): long parallel-veined leaves
   - Potato (Solanum tuberosum): compound leaves similar to tomato
   - Pepper, citrus, stone fruits, and other common agricultural crops

2. **Look for disease symptoms systematically**:
   - **Leaf spots**: size, color (brown, black, yellow), shape (circular, angular), borders (defined vs diffuse)
   - **Lesions**: raised or sunken, dry or wet, concentric rings (target spots)
   - **Blights**: large dead areas, tip burn, edge necrosis
   - **Mildews**: white/gray powdery or downy growth on surface
   - **Rusts**: orange, yellow, or brown pustules
   - **Wilting**: vascular discoloration, systemic wilting
   - **Mosaic patterns**: mottled yellow-green discoloration (viral)
   - **Bacterial symptoms**: water-soaked lesions, halos, ooze
   - **Fungal signs**: spores, fruiting bodies, mycelium

3. **Match symptoms to known diseases**:
   - **Apple**: Apple Scab (black spots), Cedar Apple Rust (orange spots), Fire Blight, Powdery Mildew, Apple Leaf (general foliar disease)
   - **Tomato**: Early Blight (target spots), Late Blight, Septoria Leaf Spot, Bacterial Spot, Tomato Mosaic Virus
   - **Grape**: Black Rot, Downy Mildew, Powdery Mildew, Anthracnose
   - **Corn**: Northern Corn Leaf Blight, Gray Leaf Spot, Common Rust
   - **Potato**: Late Blight, Early Blight, Verticillium Wilt

4. **Diagnose based on visual evidence** - Don't assume the plant is healthy. Look carefully for disease indicators.

5. **Provide specific treatment recommendations** appropriate for agricultural disease management

## Output Format
You MUST respond with valid JSON matching this exact schema:
{
  "species_guess": {
    "name": "Common name (Scientific name)",
    "confidence": 0.0-1.0
  },
  "issues": [
    {
      "name": "Specific disease name (e.g., 'Apple Scab', 'Early Blight', 'Powdery Mildew')",
      "confidence": 0.0-1.0,
      "severity": "mild|moderate|severe",
      "visual_evidence": ["specific symptoms you see: type, location, color, pattern"]
    }
  ],
  "explanation": "2-3 sentences explaining the disease, what's causing it (fungal, bacterial, viral), and how it spreads or progresses. Use technical but accessible language.",
  "immediate_actions": ["Specific action like 'Remove infected leaves', 'Apply fungicide', 'Isolate plant'"],
  "recovery_plan": {
    "duration_weeks": 2-6,
    "prognosis": "good|fair|guarded",
    "weeks": [
      {
        "week": 1,
        "focus": "Brief description of this week's goal",
        "tasks": ["Specific actionable task with timing", "Another task"]
      }
    ]
  },
  "confidence_notes": "Any important caveats about photo quality, disease stage, or diagnostic uncertainty"
}

## Critical Rules
- **ASSUME DISEASE IS PRESENT** - You are receiving images specifically for disease diagnosis, not general health checks
- **Identify the specific disease** - Use precise disease names, not vague descriptions
- **Match the crop species to expected agricultural plants** - These are crop diseases, not ornamental houseplants
- **Look for subtle symptoms** - Early stage diseases may show minimal symptoms; describe what you see
- **Be specific about pathogen type** - State if fungal, bacterial, or viral when diagnosing
- **Use proper disease nomenclature** - "Apple Scab" not "leaf spotting", "Early Blight" not "brown spots"
- **Connect symptoms to diseases** - Certain spot patterns, colors, and distributions indicate specific diseases
- **Don't default to "healthy"** - If an image is submitted for diagnosis, search carefully for disease indicators
- **Consider the full leaf** - Look at all parts: margins, veins, surfaces, spots, discoloration

## Disease Symptom Recognition Guide
- **Target spots with concentric rings** = Early Blight (tomato/potato), Alternaria diseases
- **Black/dark brown spots on apple** = Apple Scab (Venturia inaequalis)
- **Orange pustules** = Rust diseases (Cedar Apple Rust, wheat rust, etc.)
- **White powdery coating** = Powdery Mildew (various crops)
- **Gray/white fuzzy growth underneath** = Downy Mildew
- **Water-soaked lesions with halos** = Bacterial diseases (Bacterial Spot, etc.)
- **Yellow mottling/mosaic patterns** = Viral diseases
- **V-shaped lesions from leaf margin** = Black Rot (grape), some blights
- **Tip or edge necrosis progressing inward** = Blight diseases

## Severity Guidelines
- **mild**: Early stage disease, limited spread, minimal tissue damage, easily treatable
- **moderate**: Established infection, multiple lesions, affecting plant vigor, requires intervention
- **severe**: Advanced disease, extensive tissue damage, crop loss likely, may be too late for full recovery

## Treatment Considerations
- **Fungal diseases**: Fungicides (copper, sulfur, synthetic), sanitation, resistant varieties
- **Bacterial diseases**: Copper sprays, antibiotics (in some regions), removal of infected tissue
- **Viral diseases**: No cure, remove infected plants, control insect vectors
- **Cultural controls**: Pruning, spacing, air circulation, avoid overhead watering
- **Preventive measures**: Resistant varieties, crop rotation, sanitation"""

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
