#!/usr/bin/env python3
"""
Test script for PlantDoc diagnosis endpoint.
Usage: python test_diagnose.py path/to/plant-image.jpg
"""

import sys
import base64
import json
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_diagnose(image_path: str, context: str = None):
    """Send an image to the diagnosis endpoint and print results."""

    # Read and encode image
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    print(f"Image size: {len(image_data) / 1024:.1f} KB (base64)")

    # Build request
    payload = {"image": image_data}
    if context:
        payload["context"] = context

    print(f"\nSending diagnosis request...")

    try:
        response = requests.post(
            f"{BASE_URL}/diagnose",
            json=payload,
            timeout=120
        )
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Make sure the server is running: python main.py")
        sys.exit(1)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()

    # Pretty print results
    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULTS")
    print("=" * 60)

    print(f"\nSpecies: {result['species_guess']['name']}")
    print(f"Confidence: {result['species_guess']['confidence'] * 100:.0f}%")

    print("\n--- Issues Found ---")
    for issue in result['issues']:
        severity_icon = {"mild": "[MILD]", "moderate": "[MODERATE]", "severe": "[SEVERE]"}.get(
            issue['severity'], "[?]"
        )
        print(f"\n{severity_icon} {issue['name']} ({issue['confidence'] * 100:.0f}% confident)")
        for evidence in issue['visual_evidence']:
            print(f"  - {evidence}")

    print("\n--- What's Happening ---")
    print(result['explanation'])

    print("\n--- Do This Today ---")
    for i, action in enumerate(result['immediate_actions'], 1):
        print(f"  {i}. {action}")

    print("\n--- Recovery Plan ---")
    plan = result['recovery_plan']
    print(f"Duration: {plan['duration_weeks']} weeks | Prognosis: {plan['prognosis']}")
    for week in plan['weeks']:
        print(f"\n  Week {week['week']}: {week['focus']}")
        for task in week['tasks']:
            print(f"    - {task}")

    if result['confidence_notes']:
        print(f"\n--- Notes ---")
        print(result['confidence_notes'])

    print("\n--- Stats ---")
    print(f"Processing time: {result['processing_time_ms']}ms")
    print(f"API cost: ${result['model_cost_cents'] / 100:.4f}")

    print("\n" + "=" * 60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_diagnose.py <image_path> [context]")
        print("\nExample:")
        print("  python test_diagnose.py plant.jpg")
        print("  python test_diagnose.py plant.jpg 'leaves are turning yellow'")
        sys.exit(1)

    image_path = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else None

    test_diagnose(image_path, context)

if __name__ == "__main__":
    main()
