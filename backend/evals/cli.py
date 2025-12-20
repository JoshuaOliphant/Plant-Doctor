# ABOUTME: CLI entry point for running PlantDoc evaluations.
# ABOUTME: Supports running individual models or comparing multiple models.

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import ClaudeModel, EvalSettings
from .runner import run_evaluation, print_report, save_results, calculate_metrics
from .dataset_loader import get_test_case_count, get_disease_classes, iter_test_cases
from .multi_model_classifier import (
    ModelProvider,
    classify_image,
    check_disease_match,
)
from .prompts import get_prompt


console = Console()


def run_optimize(args):
    """Run GEPA prompt optimization for classification."""
    from .dspy_optimizer import optimize_eval_prompt

    settings = EvalSettings()

    console.print("\n[bold]DSPy/GEPA Prompt Optimization - Classification[/bold]")
    console.print(f"This will optimize the classification prompt using GEPA")
    console.print(f"Training samples: {args.train_samples}")
    console.print(f"Validation samples: {args.val_samples}")
    console.print(f"Max iterations: {args.iterations}")

    if args.dry_run:
        console.print("\n[yellow]Dry run - would run optimization with above settings[/yellow]")
        return

    console.print("\n[cyan]Starting optimization...[/cyan]")

    result = optimize_eval_prompt(
        settings=settings,
        max_train=args.train_samples,
        max_val=args.val_samples,
        output_path=settings.results_path / "gepa_optimization.json"
    )

    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"  Original accuracy: {result.original_accuracy:.1%}")
    console.print(f"  Optimized accuracy: {result.optimized_accuracy:.1%}")
    console.print(f"  Improvement: {result.improvement:+.1%}")
    console.print(f"\nResults saved to: {settings.results_path / 'gepa_optimization.json'}")


async def run_multi_model_eval(args):
    """Run evaluation across multiple models and prompt types."""
    settings = EvalSettings()

    # Parse models to evaluate
    model_map = {
        # Claude models
        "claude-haiku": ModelProvider.CLAUDE_HAIKU,
        "claude-sonnet": ModelProvider.CLAUDE_SONNET,

        # OpenAI GPT-5 family (latest)
        "gpt-5.2": ModelProvider.GPT_5_2,
        "gpt-5": ModelProvider.GPT_5,
        "gpt-5-mini": ModelProvider.GPT_5_MINI,
        "gpt-5-nano": ModelProvider.GPT_5_NANO,

        # OpenAI GPT-4 family
        "gpt-4o": ModelProvider.GPT_4O,
        "gpt-4o-mini": ModelProvider.GPT_4O_MINI,
        "gpt-4.1": ModelProvider.GPT_4_1,
        "gpt-4.1-mini": ModelProvider.GPT_4_1_MINI,
        "gpt-4.1-nano": ModelProvider.GPT_4_1_NANO,

        # Google Gemini 3 family (latest)
        "gemini-3-pro": ModelProvider.GEMINI_3_PRO,
        "gemini-3-flash": ModelProvider.GEMINI_3_FLASH,

        # Google Gemini 2 family
        "gemini-2-flash": ModelProvider.GEMINI_2_FLASH,
        "gemini-2-pro": ModelProvider.GEMINI_2_PRO,

        # Google Gemini 1.5 family
        "gemini-1.5-flash": ModelProvider.GEMINI_1_5_FLASH,
        "gemini-1.5-pro": ModelProvider.GEMINI_1_5_PRO,
    }

    models_to_run = []
    for model_name in args.models:
        if model_name in model_map:
            models_to_run.append(model_map[model_name])
        else:
            console.print(f"[yellow]Warning: Unknown model '{model_name}', skipping[/yellow]")

    if not models_to_run:
        console.print("[red]No valid models specified[/red]")
        return

    prompt_types = args.prompts.split(",") if args.prompts else ["baseline"]

    # Load test cases
    cases = list(iter_test_cases(settings))
    if args.max_images:
        cases = cases[:args.max_images]

    console.print("\n[bold]Multi-Model Plant Disease Classification Evaluation[/bold]")
    console.print(f"Models: {', '.join(m.name for m in models_to_run)}")
    console.print(f"Prompt types: {', '.join(prompt_types)}")
    console.print(f"Test cases: {len(cases)}")

    if args.dry_run:
        console.print("\n[yellow]Dry run - would evaluate with above settings[/yellow]")
        return

    # Run evaluations
    all_results = {}

    for model in models_to_run:
        for prompt_type in prompt_types:
            key = f"{model.name}_{prompt_type}"
            results = []
            correct = 0
            errors = 0
            total_cost = 0.0
            total_time = 0

            console.print(f"\n[cyan]Evaluating {model.name} with {prompt_type} prompt...[/cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Processing...", total=len(cases))

                for case in cases:
                    result = await classify_image(case, model, prompt_type)
                    results.append(result)

                    if result.is_correct:
                        correct += 1
                    if result.error:
                        errors += 1
                    total_cost += result.cost_cents
                    total_time += result.processing_time_ms

                    accuracy = (correct / len(results)) * 100
                    progress.update(
                        task,
                        advance=1,
                        description=f"Accuracy: {accuracy:.1f}%"
                    )

            all_results[key] = {
                "model": model.name,
                "prompt_type": prompt_type,
                "results": results,
                "accuracy": (correct / len(results)) * 100 if results else 0,
                "correct": correct,
                "errors": errors,
                "total": len(results),
                "total_cost_cents": total_cost,
                "avg_time_ms": total_time / len(results) if results else 0,
            }

    # Print comparison table
    console.print("\n" + "=" * 80)
    console.print("[bold]Multi-Model Comparison Results[/bold]")
    console.print("=" * 80)

    comparison = Table(title="Model Comparison")
    comparison.add_column("Model", style="cyan")
    comparison.add_column("Prompt", style="magenta")
    comparison.add_column("Accuracy", style="green")
    comparison.add_column("Correct", style="white")
    comparison.add_column("Cost", style="yellow")
    comparison.add_column("Avg Time", style="dim")

    for key, metrics in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
        comparison.add_row(
            metrics["model"],
            metrics["prompt_type"],
            f"{metrics['accuracy']:.1f}%",
            f"{metrics['correct']}/{metrics['total']}",
            f"${metrics['total_cost_cents'] / 100:.4f}",
            f"{metrics['avg_time_ms']:.0f}ms",
        )

    console.print(comparison)

    # Save results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = settings.results_path / f"multi_model_comparison_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        output_data = {}
        for key, metrics in all_results.items():
            output_data[key] = {
                "model": metrics["model"],
                "prompt_type": metrics["prompt_type"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "errors": metrics["errors"],
                "total": metrics["total"],
                "total_cost_cents": metrics["total_cost_cents"],
                "avg_time_ms": metrics["avg_time_ms"],
            }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\nResults saved to: {output_path}")


def run_optimize_diagnosis(args):
    """Run GEPA prompt optimization for full diagnosis."""
    from .diagnosis_optimizer import optimize_diagnosis_prompt

    settings = EvalSettings()

    console.print("\n[bold]DSPy/GEPA Prompt Optimization - Full Diagnosis[/bold]")
    console.print(f"This will optimize the full diagnosis prompt using GEPA")
    console.print(f"Training samples: {args.train_samples}")
    console.print(f"Validation samples: {args.val_samples}")
    console.print(f"Max metric calls: {args.metric_calls}")
    console.print(f"[cyan]Note: Using Sonnet 4.5 for complex diagnosis task[/cyan]")

    if args.dry_run:
        console.print("\n[yellow]Dry run - would run optimization with above settings[/yellow]")
        return

    console.print("\n[cyan]Starting optimization...[/cyan]")

    result = optimize_diagnosis_prompt(
        settings=settings,
        max_train=args.train_samples,
        max_val=args.val_samples,
        output_path=settings.results_path / "diagnosis_optimization.json"
    )

    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"  Original score: {result.original_score:.2f}")
    console.print(f"  Optimized score: {result.optimized_score:.2f}")
    console.print(f"  Improvement: {result.improvement:+.2f}")
    console.print(f"  Total cost: ${result.total_cost_cents / 100:.4f}")
    console.print(f"\nResults saved to: {settings.results_path / 'diagnosis_optimization.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="PlantDoc evaluation and optimization CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run model evaluation")
    eval_parser.add_argument(
        "--model",
        type=str,
        choices=["haiku", "sonnet", "both"],
        default="sonnet",
        help="Which model to evaluate (default: sonnet)"
    )
    eval_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to evaluate (default: all)"
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running"
    )
    eval_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON file (default: true)"
    )
    eval_parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use GEPA-optimized prompt instead of baseline"
    )

    # Optimize subcommand (classification)
    opt_parser = subparsers.add_parser("optimize", help="Run GEPA classification prompt optimization")
    opt_parser.add_argument(
        "--train-samples",
        type=int,
        default=50,
        help="Number of training samples (default: 50)"
    )
    opt_parser.add_argument(
        "--val-samples",
        type=int,
        default=30,
        help="Number of validation samples (default: 30)"
    )
    opt_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Max optimization iterations (default: 10)"
    )
    opt_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )

    # Optimize-diagnosis subcommand (full diagnosis)
    diag_opt_parser = subparsers.add_parser(
        "optimize-diagnosis",
        help="Run GEPA full diagnosis prompt optimization"
    )
    diag_opt_parser.add_argument(
        "--train-samples",
        type=int,
        default=30,
        help="Number of training samples (default: 30, lower due to cost)"
    )
    diag_opt_parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Number of validation samples (default: 20)"
    )
    diag_opt_parser.add_argument(
        "--metric-calls",
        type=int,
        default=50,
        help="Max metric calls for optimization (default: 50)"
    )
    diag_opt_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )

    # Multi-model comparison subcommand
    multi_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple models and prompts"
    )
    multi_parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet", "gpt-5-mini", "gemini-3-flash"],
        help="Models to compare (default: claude-sonnet gpt-5-mini gemini-3-flash). "
             "Options: claude-haiku, claude-sonnet, "
             "gpt-5.2, gpt-5, gpt-5-mini, gpt-5-nano, "
             "gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, "
             "gemini-3-pro, gemini-3-flash, gemini-2-flash, gemini-2-pro, "
             "gemini-1.5-flash, gemini-1.5-pro"
    )
    multi_parser.add_argument(
        "--prompts",
        type=str,
        default="baseline,hybrid,cot",
        help="Comma-separated prompt types to test (default: baseline,hybrid,cot). "
             "Options: baseline, hybrid, cot, structured"
    )
    multi_parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum images to evaluate per model/prompt combo (default: 50)"
    )
    multi_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON file (default: true)"
    )
    multi_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running"
    )

    args = parser.parse_args()

    # Handle optimize command
    if args.command == "optimize":
        run_optimize(args)
        return

    # Handle optimize-diagnosis command
    if args.command == "optimize-diagnosis":
        run_optimize_diagnosis(args)
        return

    # Handle compare command (multi-model)
    if args.command == "compare":
        asyncio.run(run_multi_model_eval(args))
        return

    # Default to eval if no command or eval command
    if args.command is None:
        # Support legacy usage without subcommand
        args = eval_parser.parse_args()

    # Load settings
    try:
        settings = EvalSettings()
        if args.max_images:
            settings.max_images = args.max_images
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        console.print("[dim]Make sure ANTHROPIC_API_KEY is set in .env[/dim]")
        sys.exit(1)

    # Show dataset info
    total_cases = get_test_case_count(settings)
    disease_classes = get_disease_classes(settings)

    console.print("\n[bold]PlantDoc Evaluation[/bold]")
    console.print(f"Dataset path: {settings.plantdoc_dataset_path}")
    console.print(f"Total test cases: {total_cases}")
    console.print(f"Disease classes: {len(disease_classes)}")

    use_optimized = getattr(args, 'use_optimized', False)
    if use_optimized:
        console.print("[magenta]Using GEPA-optimized prompt[/magenta]")

    if args.dry_run:
        console.print("\n[yellow]Dry run - showing disease classes:[/yellow]")
        for cls in disease_classes:
            console.print(f"  - {cls}")
        return

    # Determine models to run
    models_to_run: list[ClaudeModel] = []
    if args.model == "haiku":
        models_to_run = [ClaudeModel.HAIKU_45]
    elif args.model == "sonnet":
        models_to_run = [ClaudeModel.SONNET_45]
    elif args.model == "both":
        models_to_run = [ClaudeModel.HAIKU_45, ClaudeModel.SONNET_45]

    # Run evaluations
    all_results: dict[ClaudeModel, list] = {}

    for model in models_to_run:
        results = run_evaluation(model, settings, use_optimized=use_optimized)
        all_results[model] = results
        print_report(results, model)

        if args.save:
            save_results(results, model, settings)

    # Comparison summary if running both
    if len(models_to_run) > 1:
        console.print("\n" + "=" * 60)
        console.print("[bold magenta]Model Comparison[/bold magenta]")
        console.print("=" * 60)

        from rich.table import Table
        comparison = Table(title="Haiku vs Sonnet")
        comparison.add_column("Metric", style="cyan")
        comparison.add_column("Haiku 3.5", style="yellow")
        comparison.add_column("Sonnet 4.5", style="green")

        haiku_metrics = calculate_metrics(all_results[ClaudeModel.HAIKU_45])
        sonnet_metrics = calculate_metrics(all_results[ClaudeModel.SONNET_45])

        comparison.add_row(
            "Accuracy",
            f"{haiku_metrics['accuracy']:.1f}%",
            f"{sonnet_metrics['accuracy']:.1f}%"
        )
        comparison.add_row(
            "Total Cost",
            f"${haiku_metrics['total_cost_cents'] / 100:.4f}",
            f"${sonnet_metrics['total_cost_cents'] / 100:.4f}"
        )
        comparison.add_row(
            "Avg Time",
            f"{haiku_metrics['avg_time_ms']:.0f}ms",
            f"{sonnet_metrics['avg_time_ms']:.0f}ms"
        )

        console.print(comparison)


if __name__ == "__main__":
    main()
