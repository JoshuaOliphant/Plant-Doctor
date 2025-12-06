# PlantDoc PoC - FastAPI Backend
# Single endpoint for plant diagnosis using Claude Vision

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import time
import json
import base64
import re

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

client = anthropic.Anthropic()

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

# The core diagnosis prompt - this is the main IP
DIAGNOSIS_SYSTEM_PROMPT = """You are PlantDoc, an expert plant pathologist with decades of experience diagnosing houseplant health issues. Analyze plant photos to provide accurate, actionable diagnoses.

## Your Process
1. **Identify the plant species** - Look at leaf shape, growth pattern, texture. Note uncertainty if unclear.
2. **Analyze visible symptoms systematically**:
   - Leaves: color changes, spots, wilting, texture, edge condition, yellowing patterns
   - Stems: firmness, discoloration, lesions, rot
   - Soil: visible moisture level, mold, salt buildup, root visibility
   - Overall plant: drooping, legginess, sparse growth, symmetry
3. **List possible issues** ranked by likelihood based on visual evidence
4. **Explain in plain English** what's happening and why
5. **Provide specific recovery steps** a beginner can follow

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
- **Consider photo quality** - Note if lighting, angle, or resolution limits diagnosis
- **Think about common issues first** - Overwatering is more common than rare diseases

## Severity Guidelines
- **mild**: Cosmetic issues, early-stage problems, easy recovery expected
- **moderate**: Affecting plant health, needs attention within days, good recovery possible
- **severe**: Life-threatening to plant, immediate action needed, recovery uncertain

## Common Houseplant Issues to Consider
- Overwatering (yellow leaves, mushy stems, root rot)
- Underwatering (crispy edges, drooping, dry soil)
- Low light (leggy growth, pale leaves, leaning)
- Too much light (bleached spots, crispy patches)
- Nutrient deficiency (yellowing patterns, stunted growth)
- Pests (visible bugs, webbing, honeydew, stippling)
- Root bound (roots circling, water runs straight through)
- Temperature stress (leaf drop, browning)
- Low humidity (brown tips, crispy edges)"""

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in cents for Claude Sonnet"""
    # Claude Sonnet 4.5 pricing: $3/1M input, $15/1M output
    input_cost = (input_tokens / 1_000_000) * 3.0
    output_cost = (output_tokens / 1_000_000) * 15.0
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
        # Call Claude Vision
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
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
            response.usage.output_tokens
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
            model_cost_cents=cost_cents
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
