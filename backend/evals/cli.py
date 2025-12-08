# ABOUTME: CLI entry point for running PlantDoc evaluations.
# ABOUTME: Supports running individual models or comparing multiple models.

import argparse
import sys

from rich.console import Console

from .config import ClaudeModel, EvalSettings
from .runner import run_evaluation, print_report, save_results, calculate_metrics
from .dataset_loader import get_test_case_count, get_disease_classes


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

    args = parser.parse_args()

    # Handle optimize command
    if args.command == "optimize":
        run_optimize(args)
        return

    # Handle optimize-diagnosis command
    if args.command == "optimize-diagnosis":
        run_optimize_diagnosis(args)
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
