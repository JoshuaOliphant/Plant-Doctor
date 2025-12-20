# ABOUTME: DSPy/GEPA prompt optimizer for plant disease classification.
# ABOUTME: Uses reflective prompt evolution to improve Haiku accuracy.

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import anthropic
from gepa import optimize, GEPAAdapter, GEPAResult, EvaluationBatch

from .config import ClaudeModel, EvalSettings, EvalCase, MODEL_PRICING
from .dataset_loader import iter_test_cases, load_image_as_base64
from .diagnosis_task import check_disease_match, analyze_failure_pattern


@dataclass
class OptimizationResult:
    """Results from prompt optimization."""
    original_accuracy: float
    optimized_accuracy: float
    improvement: float
    optimized_prompt: str
    num_metric_calls: int
    total_cost_cents: float


# The initial prompt we want to optimize
SEED_SYSTEM_PROMPT = """You are a plant disease classifier. Analyze plant images and identify the species and any visible disease.

Respond with ONLY valid JSON in this exact format:
{
  "species": "Plant name",
  "disease": "Specific disease name or 'Healthy'",
  "confidence": 0.0-1.0
}

Guidelines:
- Include the plant name in the disease when relevant (e.g., "Apple rust" not just "Rust")
- Be specific: "Tomato early blight" not "Blight"
- Use "Healthy" if no disease is visible
- Output ONLY the JSON, no additional text"""


@dataclass
class PlantDocInstance:
    """A single evaluation instance for GEPA."""
    image_b64: str
    expected_disease: str
    plant_species: str


@dataclass
class ClassificationOutput:
    """Output from running classification."""
    predicted_disease: str | None
    predicted_species: str | None
    confidence: float | None
    score: float
    feedback: str
    input_tokens: int
    output_tokens: int


class PlantDocAdapter(GEPAAdapter[PlantDocInstance, dict, ClassificationOutput]):
    """GEPA adapter for plant disease classification.

    This adapter connects GEPA's optimization loop to our Claude Vision classifier.
    """

    def __init__(self, model: ClaudeModel, api_key: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def evaluate(
        self,
        batch: list[PlantDocInstance],
        candidate: dict[str, str],
        capture_traces: bool = False
    ) -> EvaluationBatch[dict, ClassificationOutput]:
        """Evaluate a prompt candidate on a batch of instances.

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        scores = []
        trajectories = [] if capture_traces else None
        outputs = []

        system_prompt = candidate.get("system_prompt", SEED_SYSTEM_PROMPT)

        for instance in batch:
            output = self._classify_image(system_prompt, instance)

            scores.append(output.score)
            outputs.append(output)

            if capture_traces:
                trajectories.append({
                    "system_prompt": system_prompt,
                    "expected": instance.expected_disease,
                    "predicted": output.predicted_disease,
                    "score": output.score,
                    "feedback": output.feedback
                })

            self.total_input_tokens += output.input_tokens
            self.total_output_tokens += output.output_tokens

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, ClassificationOutput],
        components_to_update: list[str]
    ) -> dict[str, list[dict]]:
        """Create a reflective dataset for GEPA to learn from failures.

        This method extracts information from failed predictions to help
        GEPA understand what went wrong and how to improve the prompt.

        Uses pattern-based feedback to help GEPA learn generalizable improvements
        rather than memorizing specific examples.

        Returns a dict mapping component name to list of reflective examples.
        """
        result = {}

        # We only have one component: system_prompt
        if "system_prompt" in components_to_update:
            reflective_examples = []

            trajectories = eval_batch.trajectories or []
            outputs = eval_batch.outputs
            scores = eval_batch.scores

            for i, (trajectory, output, score) in enumerate(zip(trajectories, outputs, scores)):
                if score < 1.0:
                    # This is a failure case - analyze the pattern
                    predicted = trajectory.get("predicted", "")
                    expected = trajectory.get("expected", "Unknown")

                    # Get pattern-based feedback instead of just "expected X got Y"
                    pattern_feedback = analyze_failure_pattern(predicted, expected)

                    reflective_examples.append({
                        "Inputs": {
                            "plant_species": expected.split()[0] if expected else "Unknown",
                            "image": "[plant leaf image]"
                        },
                        "Generated Outputs": {
                            "predicted_disease": predicted or "None",
                            "confidence": str(output.confidence) if output.confidence else "N/A"
                        },
                        "Failure Pattern": pattern_feedback,
                        "Constraints": """
IMPORTANT - Do not overfit to specific diseases. Instead, learn these GENERAL PATTERNS:
1. Viral symptoms (yellowing, mottling, curling) should NOT be classified as 'Healthy'
2. Distinguish bacterial (water-soaked, halos) from fungal (concentric rings, defined borders)
3. Verify plant species before applying disease name - don't mix species
4. Gray leaf spot has rectangular lesions between veins, distinct from blight or rust
5. Mold shows fuzzy growth; blight shows tissue death
6. The prompt should work for BOTH houseplants AND agricultural crops
"""
                    })

            result["system_prompt"] = reflective_examples

        return result

    def _classify_image(
        self,
        system_prompt: str,
        instance: PlantDocInstance
    ) -> ClassificationOutput:
        """Run classification on a single image."""
        try:
            response = self.client.messages.create(
                model=self.model.value,
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
                                "data": instance.image_b64
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
            result = self._parse_response(response.content[0].text)
            predicted_disease = result.get("disease", "")
            predicted_species = result.get("species", "")
            confidence = result.get("confidence", 0.0)

            # Calculate score and feedback
            is_correct = check_disease_match(predicted_disease, instance.expected_disease)
            score = 1.0 if is_correct else 0.0

            if is_correct:
                feedback = "Correct classification"
            else:
                feedback = f"Predicted '{predicted_disease}' but expected '{instance.expected_disease}'"
                if instance.plant_species.lower() not in predicted_disease.lower():
                    feedback += ". Consider including plant species in disease name."

            return ClassificationOutput(
                predicted_disease=predicted_disease,
                predicted_species=predicted_species,
                confidence=confidence,
                score=score,
                feedback=feedback,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        except Exception as e:
            return ClassificationOutput(
                predicted_disease=None,
                predicted_species=None,
                confidence=None,
                score=0.0,
                feedback=f"Error: {str(e)}",
                input_tokens=0,
                output_tokens=0
            )

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from Claude's response."""
        import re
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1)
        return json.loads(text.strip())

    def get_total_cost_cents(self) -> float:
        """Calculate total API cost."""
        input_price, output_price = MODEL_PRICING[self.model]
        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price
        return round((input_cost + output_cost) * 100, 4)


def load_instances(
    settings: EvalSettings,
    max_count: int | None = None
) -> list[PlantDocInstance]:
    """Load evaluation instances from the dataset."""
    instances = []

    for case in iter_test_cases(settings):
        try:
            image_b64 = load_image_as_base64(case.image_path)
            instances.append(PlantDocInstance(
                image_b64=image_b64,
                expected_disease=case.expected_disease,
                plant_species=case.plant_species
            ))

            if max_count and len(instances) >= max_count:
                break
        except Exception:
            continue

    return instances


def run_gepa_optimization(
    settings: EvalSettings,
    max_train: int = 50,
    max_val: int = 30,
    max_metric_calls: int = 100,
    model: ClaudeModel = ClaudeModel.HAIKU_45
) -> OptimizationResult:
    """Run GEPA prompt optimization on the classifier.

    Args:
        settings: Evaluation settings
        max_train: Maximum training instances
        max_val: Maximum validation instances
        max_metric_calls: Maximum API calls for optimization
        model: Claude model to use

    Returns:
        OptimizationResult with before/after metrics and optimized prompt
    """
    print(f"Loading dataset instances...")

    # Load all instances with images pre-loaded
    all_instances = load_instances(settings, max_count=max_train + max_val)

    # Split into train/val
    train_instances = all_instances[:max_train]
    val_instances = all_instances[max_train:max_train + max_val]

    print(f"Loaded {len(train_instances)} train, {len(val_instances)} val instances")

    # Create adapter
    adapter = PlantDocAdapter(model=model, api_key=settings.anthropic_api_key)

    # Evaluate baseline
    print("Evaluating baseline prompt...")
    baseline_result = adapter.evaluate(
        val_instances[:10],  # Quick baseline on subset
        {"system_prompt": SEED_SYSTEM_PROMPT}
    )
    baseline_accuracy = sum(baseline_result.scores) / len(baseline_result.scores) if baseline_result.scores else 0.0
    print(f"Baseline accuracy: {baseline_accuracy:.1%}")

    # Define seed candidate
    seed_candidate = {
        "system_prompt": SEED_SYSTEM_PROMPT
    }

    # Run GEPA optimization
    print(f"Starting GEPA optimization (max {max_metric_calls} metric calls)...")
    start_time = time.time()

    result: GEPAResult = optimize(
        seed_candidate=seed_candidate,
        trainset=train_instances,
        valset=val_instances,
        adapter=adapter,
        reflection_lm="anthropic/claude-haiku-4-5-20251001",
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=3,
        display_progress_bar=True
    )

    elapsed = time.time() - start_time
    print(f"Optimization completed in {elapsed:.1f}s")

    # Get best candidate
    best_candidate = result.best_candidate
    optimized_prompt = best_candidate.get("system_prompt", SEED_SYSTEM_PROMPT)

    # Evaluate optimized prompt on full validation set
    print("Evaluating optimized prompt...")
    optimized_result = adapter.evaluate(
        val_instances,
        best_candidate
    )
    optimized_accuracy = sum(optimized_result.scores) / len(optimized_result.scores) if optimized_result.scores else 0.0

    return OptimizationResult(
        original_accuracy=baseline_accuracy,
        optimized_accuracy=optimized_accuracy,
        improvement=optimized_accuracy - baseline_accuracy,
        optimized_prompt=optimized_prompt,
        num_metric_calls=result.num_metric_calls if hasattr(result, 'num_metric_calls') else max_metric_calls,
        total_cost_cents=adapter.get_total_cost_cents()
    )


def optimize_eval_prompt(
    settings: EvalSettings | None = None,
    max_train: int = 50,
    max_val: int = 30,
    output_path: Path | None = None
) -> OptimizationResult:
    """Main entry point for prompt optimization.

    Loads the PlantDoc dataset and runs GEPA optimization.
    """
    if settings is None:
        settings = EvalSettings()

    # Run optimization
    result = run_gepa_optimization(
        settings=settings,
        max_train=max_train,
        max_val=max_val,
        model=ClaudeModel.HAIKU_45
    )

    # Save results if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "original_accuracy": result.original_accuracy,
                "optimized_accuracy": result.optimized_accuracy,
                "improvement": result.improvement,
                "optimized_prompt": result.optimized_prompt,
                "num_metric_calls": result.num_metric_calls,
                "total_cost_cents": result.total_cost_cents
            }, f, indent=2)

    return result


if __name__ == "__main__":
    result = optimize_eval_prompt(
        output_path=Path(__file__).parent / "results" / "gepa_optimization.json"
    )
    print(f"\nOptimization Results:")
    print(f"  Original accuracy: {result.original_accuracy:.1%}")
    print(f"  Optimized accuracy: {result.optimized_accuracy:.1%}")
    print(f"  Improvement: {result.improvement:+.1%}")
    print(f"  Total cost: ${result.total_cost_cents / 100:.4f}")
    print(f"\nOptimized prompt saved to results/gepa_optimization.json")
