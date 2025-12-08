# ABOUTME: Loads PlantDoc dataset images and creates evaluation cases.
# ABOUTME: Parses folder structure to extract disease labels and plant species.

from pathlib import Path
import base64
from typing import Iterator

from .config import EvalCase, EvalSettings


def extract_plant_species(disease_label: str) -> str:
    """Extract plant species from disease label.

    Examples:
        "Apple Scab Leaf" -> "Apple"
        "Tomato leaf bacterial spot" -> "Tomato"
        "Corn Gray leaf spot" -> "Corn"
    """
    # Common plant names that appear at the start of labels
    plants = [
        "Apple", "Bell_pepper", "Blueberry", "Cherry", "Corn",
        "grape", "Peach", "Potato", "Raspberry", "Soyabean",
        "Squash", "Strawberry", "Tomato"
    ]

    label_lower = disease_label.lower()
    for plant in plants:
        if label_lower.startswith(plant.lower()):
            return plant.replace("_", " ").title()

    # Fallback: take first word
    return disease_label.split()[0].replace("_", " ").title()


def load_image_as_base64(image_path: Path) -> str:
    """Load an image file and return its base64-encoded contents."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def iter_test_cases(settings: EvalSettings) -> Iterator[EvalCase]:
    """Iterate over all test cases in the PlantDoc dataset.

    Yields EvalCase objects for each image in the test set.
    """
    test_dir = settings.plantdoc_dataset_path / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    count = 0
    for disease_dir in sorted(test_dir.iterdir()):
        if not disease_dir.is_dir():
            continue

        disease_label = disease_dir.name
        plant_species = extract_plant_species(disease_label)

        for image_path in sorted(disease_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            yield EvalCase(
                image_path=image_path,
                expected_disease=disease_label,
                plant_species=plant_species
            )

            count += 1
            if settings.max_images and count >= settings.max_images:
                return


def get_test_case_count(settings: EvalSettings) -> int:
    """Count total test cases without loading them all."""
    return sum(1 for _ in iter_test_cases(settings))


def get_disease_classes(settings: EvalSettings) -> list[str]:
    """Get list of all disease class labels in the test set."""
    test_dir = settings.plantdoc_dataset_path / "test"
    return sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir()
    ])
