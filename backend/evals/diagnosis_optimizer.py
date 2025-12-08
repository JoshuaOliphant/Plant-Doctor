# ABOUTME: GEPA optimizer for full plant diagnosis with complex JSON schema.
# ABOUTME: Evaluates species ID, issues, explanations, and recovery plans.

import json
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import anthropic
from gepa import optimize, GEPAAdapter, GEPAResult, EvaluationBatch

from .config import ClaudeModel, EvalSettings, EvalCase, MODEL_PRICING
from .dataset_loader import iter_test_cases, load_image_as_base64


@dataclass
class OptimizationResult:
    """Results from diagnosis prompt optimization."""
    original_score: float
    optimized_score: float
    improvement: float
    optimized_prompt: str
    num_metric_calls: int
    total_cost_cents: float


# The diagnosis prompt from main.py - this is what we want to optimize
SEED_DIAGNOSIS_PROMPT = """You are PlantDoc, an expert plant pathologist with decades of experience diagnosing houseplant health issues. Analyze plant photos to provide accurate, actionable diagnoses.

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


@dataclass
class DiagnosisInstance:
    """A single diagnosis evaluation instance for GEPA."""
    image_b64: str
    expected_disease: str
    plant_species: str


@dataclass
class DiagnosisOutput:
    """Output from running full diagnosis."""
    species_guess: dict | None
    issues: list[dict] | None
    explanation: str | None
    immediate_actions: list[str] | None
    recovery_plan: dict | None
    confidence_notes: str | None
    score: float
    feedback: str
    input_tokens: int
    output_tokens: int
    raw_json: dict | None = None


class DiagnosisAdapter(GEPAAdapter[DiagnosisInstance, dict, DiagnosisOutput]):
    """GEPA adapter for full plant diagnosis optimization.

    Evaluates the complete diagnosis output including species identification,
    issue detection, explanations, and recovery plans.
    """

    def __init__(self, model: ClaudeModel, api_key: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def evaluate(
        self,
        batch: list[DiagnosisInstance],
        candidate: dict[str, str],
        capture_traces: bool = False
    ) -> EvaluationBatch[dict, DiagnosisOutput]:
        """Evaluate a diagnosis prompt candidate on a batch of instances.

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        scores = []
        trajectories = [] if capture_traces else None
        outputs = []

        system_prompt = candidate.get("system_prompt", SEED_DIAGNOSIS_PROMPT)

        for instance in batch:
            output = self._run_diagnosis(system_prompt, instance)

            scores.append(output.score)
            outputs.append(output)

            if capture_traces:
                trajectories.append({
                    "system_prompt": system_prompt,
                    "expected_disease": instance.expected_disease,
                    "expected_species": instance.plant_species,
                    "species_guess": output.species_guess,
                    "issues": output.issues,
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
        eval_batch: EvaluationBatch[dict, DiagnosisOutput],
        components_to_update: list[str]
    ) -> dict[str, list[dict]]:
        """Create a reflective dataset for GEPA to learn from failures.

        Extracts information from low-scoring diagnoses to help GEPA
        understand what went wrong and how to improve the prompt.

        Returns a dict mapping component name to list of reflective examples.
        """
        result = {}

        if "system_prompt" in components_to_update:
            reflective_examples = []

            trajectories = eval_batch.trajectories or []
            outputs = eval_batch.outputs
            scores = eval_batch.scores

            for trajectory, output, score in zip(trajectories, outputs, scores):
                # Include examples that scored below perfect
                if score < 0.9:
                    reflective_examples.append({
                        "Inputs": {
                            "plant_species": trajectory.get("expected_species", "Unknown"),
                            "disease": trajectory.get("expected_disease", "Unknown"),
                            "image": "[plant disease image]"
                        },
                        "Generated Outputs": {
                            "species_guess": str(trajectory.get("species_guess", {})),
                            "issues": str(trajectory.get("issues", [])),
                            "explanation": output.explanation or "N/A"
                        },
                        "Feedback": f"Score: {score:.2f}. {trajectory.get('feedback', '')}"
                    })

            result["system_prompt"] = reflective_examples

        return result

    def _run_diagnosis(
        self,
        system_prompt: str,
        instance: DiagnosisInstance
    ) -> DiagnosisOutput:
        """Run full diagnosis on a single image."""
        try:
            response = self.client.messages.create(
                model=self.model.value,
                max_tokens=2000,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please diagnose this plant and respond with the JSON schema specified in your instructions:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": instance.image_b64
                            }
                        }
                    ]
                }]
            )

            # Parse response
            result = self._parse_response(response.content[0].text)

            # Extract components
            species_guess = result.get("species_guess", {})
            issues = result.get("issues", [])
            explanation = result.get("explanation", "")
            immediate_actions = result.get("immediate_actions", [])
            recovery_plan = result.get("recovery_plan", {})
            confidence_notes = result.get("confidence_notes", "")

            # Calculate composite score
            score, feedback = self._score_diagnosis(
                result,
                instance.expected_disease,
                instance.plant_species
            )

            return DiagnosisOutput(
                species_guess=species_guess,
                issues=issues,
                explanation=explanation,
                immediate_actions=immediate_actions,
                recovery_plan=recovery_plan,
                confidence_notes=confidence_notes,
                score=score,
                feedback=feedback,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                raw_json=result
            )

        except Exception as e:
            return DiagnosisOutput(
                species_guess=None,
                issues=None,
                explanation=None,
                immediate_actions=None,
                recovery_plan=None,
                confidence_notes=None,
                score=0.0,
                feedback=f"Error: {str(e)}",
                input_tokens=0,
                output_tokens=0,
                raw_json=None
            )

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from Claude's response, handling markdown code blocks."""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1)
        return json.loads(text.strip())

    def _score_diagnosis(
        self,
        diagnosis: dict,
        expected_disease: str,
        expected_species: str
    ) -> tuple[float, str]:
        """Score a diagnosis output using composite metrics.

        Returns:
            (score, feedback) where score is 0.0-1.0 and feedback explains the score
        """
        feedback_parts = []
        component_scores = []

        # 1. Species identification accuracy (30% weight)
        species_score = self._score_species(
            diagnosis.get("species_guess", {}),
            expected_species
        )
        component_scores.append(("species", species_score, 0.3))
        if species_score < 1.0:
            feedback_parts.append(
                f"Species identification: {species_score:.0%} "
                f"(expected {expected_species})"
            )

        # 2. Issue identification relevance (30% weight)
        issue_score = self._score_issues(
            diagnosis.get("issues", []),
            expected_disease
        )
        component_scores.append(("issues", issue_score, 0.3))
        if issue_score < 1.0:
            feedback_parts.append(
                f"Issue relevance: {issue_score:.0%} "
                f"(expected {expected_disease})"
            )

        # 3. Explanation quality (20% weight)
        explanation_score = self._score_explanation(
            diagnosis.get("explanation", "")
        )
        component_scores.append(("explanation", explanation_score, 0.2))
        if explanation_score < 0.8:
            feedback_parts.append(
                f"Explanation quality: {explanation_score:.0%} "
                f"(needs more detail or clarity)"
            )

        # 4. Recovery plan actionability (20% weight)
        recovery_score = self._score_recovery_plan(
            diagnosis.get("recovery_plan", {}),
            diagnosis.get("immediate_actions", [])
        )
        component_scores.append(("recovery", recovery_score, 0.2))
        if recovery_score < 0.8:
            feedback_parts.append(
                f"Recovery plan actionability: {recovery_score:.0%} "
                f"(needs more specific steps)"
            )

        # Calculate weighted total score
        total_score = sum(score * weight for _, score, weight in component_scores)

        # Build feedback
        if not feedback_parts:
            feedback = "Excellent diagnosis"
        else:
            feedback = "; ".join(feedback_parts)

        return total_score, feedback

    def _score_species(self, species_guess: dict, expected_species: str) -> float:
        """Score species identification accuracy."""
        if not species_guess or "name" not in species_guess:
            return 0.0

        predicted_name = species_guess["name"].lower()
        expected_lower = expected_species.lower()

        # Exact or substring match
        if expected_lower in predicted_name or predicted_name in expected_lower:
            return 1.0

        # Partial credit for correct genus (first word)
        if expected_lower.split()[0] in predicted_name:
            return 0.5

        return 0.0

    def _score_issues(self, issues: list[dict], expected_disease: str) -> float:
        """Score issue identification relevance."""
        if not issues:
            # If expected disease is "healthy" or contains "leaf" only, no issues is correct
            if "healthy" in expected_disease.lower():
                return 1.0
            # Otherwise penalize for missing issues
            return 0.2

        # Check if any issue name matches the expected disease
        expected_lower = expected_disease.lower()

        for issue in issues:
            issue_name = issue.get("name", "").lower()

            # Direct match
            if issue_name in expected_lower or expected_lower in issue_name:
                return 1.0

            # Check for key disease terms
            disease_terms = [
                "scab", "rust", "blight", "bacterial spot", "mosaic",
                "yellow virus", "mold", "septoria", "powdery mildew",
                "black rot", "gray leaf spot", "leaf spot"
            ]

            for term in disease_terms:
                if term in issue_name and term in expected_lower:
                    return 0.8  # Close match

        # No matching issues found - partial credit if there are any issues
        return 0.3

    def _score_explanation(self, explanation: str) -> float:
        """Score explanation quality based on length and structure."""
        if not explanation:
            return 0.0

        # Check minimum length (should be 2-3 sentences)
        if len(explanation) < 50:
            return 0.3

        # Check for connecting words that indicate causal reasoning
        reasoning_words = ["because", "due to", "caused by", "resulting from", "this means"]
        has_reasoning = any(word in explanation.lower() for word in reasoning_words)

        if has_reasoning and len(explanation) >= 100:
            return 1.0
        elif has_reasoning:
            return 0.8
        elif len(explanation) >= 100:
            return 0.7
        else:
            return 0.5

    def _score_recovery_plan(
        self,
        recovery_plan: dict,
        immediate_actions: list[str]
    ) -> float:
        """Score recovery plan actionability."""
        score_components = []

        # Check immediate actions
        if immediate_actions and len(immediate_actions) >= 1:
            # Check for specificity (not vague language)
            vague_words = ["appropriately", "properly", "as needed", "regularly"]
            specific_actions = [
                a for a in immediate_actions
                if not any(vague in a.lower() for vague in vague_words)
            ]
            if specific_actions:
                score_components.append(1.0)
            else:
                score_components.append(0.5)
        else:
            score_components.append(0.0)

        # Check recovery plan structure
        if recovery_plan:
            has_duration = "duration_weeks" in recovery_plan
            has_prognosis = "prognosis" in recovery_plan
            has_weeks = "weeks" in recovery_plan and len(recovery_plan.get("weeks", [])) > 0

            if has_duration and has_prognosis and has_weeks:
                # Check if weekly tasks are specific
                weeks = recovery_plan.get("weeks", [])
                if weeks and all(len(w.get("tasks", [])) >= 1 for w in weeks):
                    score_components.append(1.0)
                else:
                    score_components.append(0.7)
            elif has_weeks:
                score_components.append(0.5)
            else:
                score_components.append(0.2)
        else:
            score_components.append(0.0)

        return sum(score_components) / len(score_components) if score_components else 0.0

    def get_total_cost_cents(self) -> float:
        """Calculate total API cost."""
        input_price, output_price = MODEL_PRICING[self.model]
        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price
        return round((input_cost + output_cost) * 100, 4)


def load_diagnosis_instances(
    settings: EvalSettings,
    max_count: int | None = None
) -> list[DiagnosisInstance]:
    """Load evaluation instances from the dataset."""
    instances = []

    for case in iter_test_cases(settings):
        try:
            image_b64 = load_image_as_base64(case.image_path)
            instances.append(DiagnosisInstance(
                image_b64=image_b64,
                expected_disease=case.expected_disease,
                plant_species=case.plant_species
            ))

            if max_count and len(instances) >= max_count:
                break
        except Exception:
            continue

    return instances


def run_diagnosis_optimization(
    settings: EvalSettings,
    max_train: int = 30,
    max_val: int = 20,
    max_metric_calls: int = 50,
    model: ClaudeModel = ClaudeModel.SONNET_45
) -> OptimizationResult:
    """Run GEPA prompt optimization on full diagnosis task.

    Args:
        settings: Evaluation settings
        max_train: Maximum training instances (smaller than classification due to token cost)
        max_val: Maximum validation instances
        max_metric_calls: Maximum API calls for optimization
        model: Claude model to use (default Sonnet for complex task)

    Returns:
        OptimizationResult with before/after metrics and optimized prompt
    """
    print(f"Loading diagnosis dataset instances...")

    # Load instances with images pre-loaded
    all_instances = load_diagnosis_instances(settings, max_count=max_train + max_val)

    # Split into train/val
    train_instances = all_instances[:max_train]
    val_instances = all_instances[max_train:max_train + max_val]

    print(f"Loaded {len(train_instances)} train, {len(val_instances)} val instances")
    print(f"Using {model.value} for complex diagnosis task")

    # Create adapter
    adapter = DiagnosisAdapter(model=model, api_key=settings.anthropic_api_key)

    # Evaluate baseline
    print("Evaluating baseline diagnosis prompt...")
    baseline_result = adapter.evaluate(
        val_instances[:5],  # Quick baseline on small subset
        {"system_prompt": SEED_DIAGNOSIS_PROMPT}
    )
    baseline_score = sum(baseline_result.scores) / len(baseline_result.scores) if baseline_result.scores else 0.0
    print(f"Baseline score: {baseline_score:.2f}")

    # Define seed candidate
    seed_candidate = {
        "system_prompt": SEED_DIAGNOSIS_PROMPT
    }

    # Run GEPA optimization
    print(f"Starting GEPA optimization (max {max_metric_calls} metric calls)...")
    start_time = time.time()

    result: GEPAResult = optimize(
        seed_candidate=seed_candidate,
        trainset=train_instances,
        valset=val_instances,
        adapter=adapter,
        reflection_lm="anthropic/claude-sonnet-4-5-20250929",
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=2,  # Smaller batches due to large outputs
        display_progress_bar=True
    )

    elapsed = time.time() - start_time
    print(f"Optimization completed in {elapsed:.1f}s")

    # Get best candidate
    best_candidate = result.best_candidate
    optimized_prompt = best_candidate.get("system_prompt", SEED_DIAGNOSIS_PROMPT)

    # Evaluate optimized prompt on full validation set
    print("Evaluating optimized diagnosis prompt...")
    optimized_result = adapter.evaluate(
        val_instances,
        best_candidate
    )
    optimized_score = sum(optimized_result.scores) / len(optimized_result.scores) if optimized_result.scores else 0.0

    return OptimizationResult(
        original_score=baseline_score,
        optimized_score=optimized_score,
        improvement=optimized_score - baseline_score,
        optimized_prompt=optimized_prompt,
        num_metric_calls=result.num_metric_calls if hasattr(result, 'num_metric_calls') else max_metric_calls,
        total_cost_cents=adapter.get_total_cost_cents()
    )


def optimize_diagnosis_prompt(
    settings: EvalSettings | None = None,
    max_train: int = 30,
    max_val: int = 20,
    output_path: Path | None = None
) -> OptimizationResult:
    """Main entry point for diagnosis prompt optimization.

    Loads the PlantDoc dataset and runs GEPA optimization on the
    full diagnosis task with complex JSON outputs.
    """
    if settings is None:
        settings = EvalSettings()

    # Run optimization
    result = run_diagnosis_optimization(
        settings=settings,
        max_train=max_train,
        max_val=max_val,
        model=ClaudeModel.SONNET_45  # Use Sonnet for complex task
    )

    # Save results if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "original_score": result.original_score,
                "optimized_score": result.optimized_score,
                "improvement": result.improvement,
                "optimized_prompt": result.optimized_prompt,
                "num_metric_calls": result.num_metric_calls,
                "total_cost_cents": result.total_cost_cents
            }, f, indent=2)

    return result


if __name__ == "__main__":
    result = optimize_diagnosis_prompt(
        output_path=Path(__file__).parent / "results" / "diagnosis_optimization.json"
    )
    print(f"\nDiagnosis Optimization Results:")
    print(f"  Original score: {result.original_score:.2f}")
    print(f"  Optimized score: {result.optimized_score:.2f}")
    print(f"  Improvement: {result.improvement:+.2f}")
    print(f"  Total cost: ${result.total_cost_cents / 100:.4f}")
    print(f"\nOptimized prompt saved to results/diagnosis_optimization.json")
