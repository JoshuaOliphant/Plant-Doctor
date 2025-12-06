# PlantDoc PoC

AI-powered plant diagnosis using Claude Vision. This proof of concept validates whether Claude can provide valuable, actionable plant health diagnoses from photos.

## Architecture

```
┌─────────────────────────────────────┐
│         iOS App (SwiftUI)           │
│  ┌─────────────────────────────┐    │
│  │   Single Screen Flow:       │    │
│  │   Camera → Loading → Result │    │
│  └─────────────────────────────┘    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      FastAPI Backend (Minimal)      │
│  POST /diagnose                     │
│  - Receive image                    │
│  - Call Claude Vision               │
│  - Return structured diagnosis      │
└─────────────────────────────────────┘
```

## Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

### iOS App Setup

1. Open Xcode and create a new iOS App project
2. Name it "PlantDocPoC"
3. Copy the Swift files from `ios/PlantDocPoC/PlantDocPoC/` into your project
4. Add Info.plist entries for camera and photo library permissions
5. Update the `baseURL` in `ContentView.swift` to your backend URL
6. Build and run on a device (camera requires real device)

## API Endpoints

### POST /diagnose

Diagnose plant health issues from a photo.

**Request:**
```json
{
  "image": "<base64-encoded-image>",
  "context": "Optional: What the user is concerned about"
}
```

**Response:**
```json
{
  "species_guess": {
    "name": "Monstera deliciosa",
    "confidence": 0.92
  },
  "issues": [
    {
      "name": "Overwatering",
      "confidence": 0.85,
      "severity": "moderate",
      "visual_evidence": ["Yellow lower leaves", "Mushy stem base"]
    }
  ],
  "explanation": "Your plant is showing signs of...",
  "immediate_actions": ["Stop watering", "Check drainage"],
  "recovery_plan": {
    "duration_weeks": 3,
    "prognosis": "good",
    "weeks": [
      {
        "week": 1,
        "focus": "Dry out",
        "tasks": ["Do not water", "Ensure drainage"]
      }
    ]
  },
  "confidence_notes": "Photo quality is good...",
  "processing_time_ms": 3420,
  "model_cost_cents": 2.34
}
```

### GET /health

Health check endpoint for deployment platforms.

## Testing the Backend

```bash
# Test with curl (replace with actual base64 image)
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-image>", "context": "Leaves turning yellow"}'
```

Or use the included test script:

```bash
python test_diagnose.py path/to/plant-image.jpg
```

## Deployment

### Railway (Recommended)

1. Push code to GitHub
2. Connect repo to Railway
3. Add `ANTHROPIC_API_KEY` environment variable
4. Deploy

### Fly.io

```bash
cd backend
fly launch
fly secrets set ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
fly deploy
```

## Success Criteria

- Diagnosis accuracy >80% on test set
- Response time <10 seconds (p95)
- Cost per diagnosis <$0.05
- Photo success rate >90%

## Project Structure

```
Plant-Doctor/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Container config
│   └── .env.example        # Environment template
├── ios/
│   └── PlantDocPoC/
│       └── PlantDocPoC/
│           ├── PlantDocPoCApp.swift    # App entry point
│           ├── ContentView.swift       # Main flow coordinator
│           ├── ImagePicker.swift       # Camera/library picker
│           ├── DiagnosisResultView.swift # Results display
│           ├── APIClient.swift         # Network layer
│           ├── Models.swift            # Data models
│           └── Info.plist              # iOS permissions
└── README.md
```
