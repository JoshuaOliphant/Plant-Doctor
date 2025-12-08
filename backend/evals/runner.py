# ABOUTME: Main evaluation runner that orchestrates the eval pipeline.
# ABOUTME: Runs diagnosis on all test images and generates reports.

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import ClaudeModel, EvalSettings, EvalResult
from .dataset_loader import iter_test_cases, get_test_case_count, get_disease_classes
from .diagnosis_task import run_diagnosis


console = Console()


def run_evaluation(
    model: ClaudeModel,
    settings: EvalSettings | None = None,
    use_optimized: bool = False
) -> list[EvalResult]:
    """Run full evaluation on the PlantDoc test set.

    Args:
        model: Which Claude model to use
        settings: Evaluation settings (loads from env if not provided)
        use_optimized: Whether to use the GEPA-optimized prompt

    Returns:
        List of EvalResult objects
    """
    if settings is None:
        settings = EvalSettings()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    total_cases = get_test_case_count(settings)

    results: list[EvalResult] = []

    prompt_type = "[magenta](optimized)[/magenta]" if use_optimized else "(baseline)"
    console.print(f"\n[bold blue]Running evaluation with {model.value}[/bold blue] {prompt_type}")
    console.print(f"Total test cases: {total_cases}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating...", total=total_cases)

        for case in iter_test_cases(settings):
            result = run_diagnosis(case, client, model, use_optimized=use_optimized)
            results.append(result)

            # Update progress with current accuracy
            correct = sum(1 for r in results if r.is_correct)
            accuracy = correct / len(results) * 100
            progress.update(
                task,
                advance=1,
                description=f"Evaluating... (Accuracy: {accuracy:.1f}%)"
            )

    return results


def calculate_metrics(results: list[EvalResult]) -> dict:
    """Calculate evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error is not None)

    total_cost = sum(r.cost_cents for r in results)
    total_time = sum(r.processing_time_ms for r in results)

    # Per-class accuracy
    class_results: dict[str, dict] = {}
    for r in results:
        disease = r.case.expected_disease
        if disease not in class_results:
            class_results[disease] = {"correct": 0, "total": 0}
        class_results[disease]["total"] += 1
        if r.is_correct:
            class_results[disease]["correct"] += 1

    class_accuracy = {
        k: v["correct"] / v["total"] * 100
        for k, v in class_results.items()
    }

    return {
        "total_cases": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total * 100 if total > 0 else 0,
        "total_cost_cents": total_cost,
        "total_time_ms": total_time,
        "avg_time_ms": total_time / total if total > 0 else 0,
        "cost_per_image_cents": total_cost / total if total > 0 else 0,
        "class_accuracy": class_accuracy
    }


def print_report(results: list[EvalResult], model: ClaudeModel):
    """Print a formatted evaluation report."""
    metrics = calculate_metrics(results)

    console.print("\n" + "=" * 60)
    console.print(f"[bold green]Evaluation Report: {model.value}[/bold green]")
    console.print("=" * 60)

    # Summary table
    summary = Table(title="Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Total Cases", str(metrics["total_cases"]))
    summary.add_row("Correct", str(metrics["correct"]))
    summary.add_row("Errors", str(metrics["errors"]))
    summary.add_row("Accuracy", f"{metrics['accuracy']:.1f}%")
    summary.add_row("Total Cost", f"${metrics['total_cost_cents'] / 100:.4f}")
    summary.add_row("Cost/Image", f"${metrics['cost_per_image_cents'] / 100:.6f}")
    summary.add_row("Avg Time", f"{metrics['avg_time_ms']:.0f}ms")

    console.print(summary)

    # Per-class accuracy table
    class_table = Table(title="Per-Class Accuracy")
    class_table.add_column("Disease Class", style="cyan")
    class_table.add_column("Accuracy", style="green")

    for disease, acc in sorted(metrics["class_accuracy"].items(), key=lambda x: x[1], reverse=True):
        style = "green" if acc >= 70 else "yellow" if acc >= 50 else "red"
        class_table.add_row(disease, f"[{style}]{acc:.1f}%[/{style}]")

    console.print(class_table)


def save_results(
    results: list[EvalResult],
    model: ClaudeModel,
    settings: EvalSettings
):
    """Save evaluation results to JSON file."""
    settings.results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.name.lower()
    filename = f"eval_{model_name}_{timestamp}.json"
    filepath = settings.results_path / filename

    metrics = calculate_metrics(results)

    output = {
        "model": model.value,
        "timestamp": timestamp,
        "metrics": metrics,
        "results": [
            {
                "image_path": str(r.case.image_path),
                "expected_disease": r.case.expected_disease,
                "predicted_disease": r.predicted_disease,
                "predicted_species": r.predicted_species,
                "confidence": r.confidence,
                "is_correct": r.is_correct,
                "processing_time_ms": r.processing_time_ms,
                "cost_cents": r.cost_cents,
                "error": r.error
            }
            for r in results
        ]
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[dim]Results saved to: {filepath}[/dim]")
    return filepath
