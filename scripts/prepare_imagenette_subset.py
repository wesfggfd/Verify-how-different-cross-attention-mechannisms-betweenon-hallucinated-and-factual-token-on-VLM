#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def choose_article(label: str) -> str:
    first = label.strip().lower()[0]
    return "an" if first in "aeiou" else "a"


def collect_images(split_dir: Path) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        images = sorted(
            [
                image
                for image in class_dir.iterdir()
                if image.is_file() and image.suffix.lower() in {".jpeg", ".jpg", ".png"}
            ]
        )
        if images:
            images_by_class[class_dir.name] = images
    return images_by_class


def balanced_sample(
    images_by_class: dict[str, list[Path]], total: int, seed: int
) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    class_names = sorted(images_by_class)
    per_class = total // len(class_names)
    remainder = total % len(class_names)

    sampled: list[tuple[str, Path]] = []
    for index, class_name in enumerate(class_names):
        target = per_class + (1 if index < remainder else 0)
        candidates = images_by_class[class_name][:]
        rng.shuffle(candidates)
        if len(candidates) < target:
            raise ValueError(
                f"Class {class_name} has only {len(candidates)} images, cannot sample {target}."
            )
        sampled.extend((class_name, image) for image in sorted(candidates[:target]))

    rng.shuffle(sampled)
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced ImageNette subset with uniform yes/no questions."
    )
    parser.add_argument("--imagenette-root", required=True, help="Path to imagenette2-320.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    imagenette_root = Path(args.imagenette_root).expanduser().resolve()
    split_dir = imagenette_root / args.split
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not split_dir.is_dir():
        raise SystemExit(f"Split directory not found: {split_dir}")

    images_by_class = collect_images(split_dir)
    unknown = sorted(set(images_by_class) - set(IMAGENETTE_LABELS))
    if unknown:
        raise SystemExit(f"Unknown ImageNette class ids: {unknown}")

    sampled = balanced_sample(images_by_class, args.total, args.seed)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, (class_id, image_path) in enumerate(sampled, start=1):
            object_label = IMAGENETTE_LABELS[class_id]
            article = choose_article(object_label)
            record = {
                "sample_id": f"imagenette_{index:03d}_{class_id}",
                "image_path": str(image_path),
                "class_id": class_id,
                "object_label": object_label,
                "question": f"Is there {article} {object_label} in the image?",
                "expected_answer": "yes",
                "hallucinated_answer_target": "no",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
