#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Qwen factual vs hallucinated cross-attention maps."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-size", type=int, default=224)
    parser.add_argument("--per-sample-limit", type=int, default=100)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sorted_layer_names(layer_maps: dict[str, list[list[float]]]) -> list[str]:
    return sorted(layer_maps, key=lambda item: int(item.split("_")[1]))


def block_mean_map(block: dict) -> np.ndarray:
    return np.array(block["layer_summary"]["mean_over_layers"], dtype=np.float32)


def block_layer_stack(block: dict) -> np.ndarray:
    layer_maps = block["layer_maps"]
    ordered = sorted_layer_names(layer_maps)
    return np.stack([np.array(layer_maps[name], dtype=np.float32) for name in ordered], axis=0)


def answer_block(trace: list[dict]) -> dict:
    if not trace:
        raise ValueError("Expected at least one answer token trace.")
    return trace[0]["cross_attention"]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    clipped = np.clip(heatmap.astype(np.float32), 0.0, None)
    if clipped.max() > 0:
        clipped = clipped / clipped.max()
    image = Image.fromarray((clipped * 255).astype(np.uint8))
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def normalize_for_display(heatmap: np.ndarray) -> np.ndarray:
    if heatmap.size == 0:
        return heatmap
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(heatmap)
    return (heatmap - min_value) / (max_value - min_value)


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    normalized = normalize_for_display(heatmap)
    red = np.clip(1.5 - np.abs(4 * normalized - 3), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4 * normalized - 2), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4 * normalized - 1), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay_image(base_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    resized_heatmap = resize_heatmap(heatmap, base_image.size)
    overlay = Image.fromarray(heatmap_to_rgb(resized_heatmap))
    return Image.blend(base_image, overlay, alpha=alpha)


def labeled_panel(image: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 28), "white")
    panel.paste(image, (0, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill="black")
    return panel


def compose_horizontal(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return canvas


def entropy_score(heatmap: np.ndarray) -> float:
    values = np.clip(heatmap.astype(np.float64), 0.0, None).reshape(-1)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def topk_mass(heatmap: np.ndarray, fraction: float) -> float:
    values = np.clip(heatmap.astype(np.float64), 0.0, None).reshape(-1)
    total = values.sum()
    if total <= 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * fraction)))
    top_values = np.partition(values, -count)[-count:]
    return float(top_values.sum() / total)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(left_flat, right_flat) / denom)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.clip(left.astype(np.float64), 0.0, None).reshape(-1)
    right_flat = np.clip(right.astype(np.float64), 0.0, None).reshape(-1)
    if left_flat.sum() <= 0 or right_flat.sum() <= 0:
        return 0.0
    left_probs = left_flat / left_flat.sum()
    right_probs = right_flat / right_flat.sum()
    mean = 0.5 * (left_probs + right_probs)

    def kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl_div(left_probs, mean) + 0.5 * kl_div(right_probs, mean)


def center_of_mass(heatmap: np.ndarray) -> tuple[float, float]:
    values = np.clip(heatmap.astype(np.float64), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0, 0.0
    ys, xs = np.indices(values.shape)
    y = float((ys * values).sum() / total)
    x = float((xs * values).sum() / total)
    return y, x


def center_shift(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.dist(center_of_mass(left), center_of_mass(right)))


def draw_line_plot(
    curves: list[tuple[np.ndarray, str, tuple[int, int, int]]],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    width, height = 900, 420
    margin_left, margin_bottom, margin_top, margin_right = 70, 50, 40, 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), title, fill="black")
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black")

    all_values = np.concatenate([curve for curve, _, _ in curves]) if curves else np.array([0.0, 1.0])
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    x_max = max(len(curve) - 1 for curve, _, _ in curves)
    x_max = max(1, x_max)
    draw.text((5, height // 2), y_label, fill="black")

    for curve_index, (curve, label, color) in enumerate(curves):
        points = []
        for index, value in enumerate(curve):
            x = plot_left + (plot_right - plot_left) * (index / x_max)
            y = plot_bottom - (plot_bottom - plot_top) * ((value - y_min) / (y_max - y_min))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        draw.text((plot_right - 180, plot_top + 18 * curve_index), label, fill=color)

    image.save(output_path)


def draw_histogram_panel(draw, box, values: list[float], title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0 + 6, y0 + 6), title, fill="black")
    if not values:
        return
    hist, _ = np.histogram(values, bins=20)
    max_count = max(int(hist.max()), 1)
    bar_width = max(1, (x1 - x0 - 20) // len(hist))
    for index, count in enumerate(hist):
        left = x0 + 10 + index * bar_width
        right = left + bar_width - 1
        top = y1 - 10 - int((y1 - y0 - 40) * (count / max_count))
        draw.rectangle([left, top, right, y1 - 10], fill=(70, 120, 220))


def save_distribution_plot(metrics: list[dict], output_path: Path) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    boxes = [
        (20, 20, 490, 330),
        (510, 20, 980, 330),
        (20, 360, 490, 670),
        (510, 360, 980, 670),
    ]
    plots = [
        ("entropy_gap", "Entropy Gap"),
        ("topk_gap", "Top-k Mass Gap"),
        ("mean_js_divergence", "Mean JS Divergence"),
        ("center_shift", "Center Shift"),
    ]
    for box, (key, title) in zip(boxes, plots):
        draw_histogram_panel(draw, box, [metric[key] for metric in metrics], title)
    image.save(output_path)


def save_per_sample_figure(
    record: dict,
    factual_question: np.ndarray,
    factual_answer: np.ndarray,
    hallucinated_answer: np.ndarray,
    delta_map: np.ndarray,
    output_path: Path,
) -> None:
    base_image = load_image(record["image_path"])
    panel_size = (320, 320)
    base_panel = labeled_panel(base_image.resize(panel_size), Path(record["image_path"]).name)
    fq_panel = labeled_panel(
        overlay_image(base_image.resize(panel_size), factual_question),
        "Question->Vision",
    )
    fa_panel = labeled_panel(
        overlay_image(base_image.resize(panel_size), factual_answer),
        f"Factual:{record['factual_answer']}",
    )
    ha_panel = labeled_panel(
        overlay_image(base_image.resize(panel_size), hallucinated_answer),
        f"Hallucinated:{record['hallucinated_answer']}",
    )
    delta_panel = labeled_panel(
        Image.fromarray(heatmap_to_rgb(np.abs(delta_map))).resize(panel_size),
        "Absolute Delta",
    )
    compose_horizontal([base_panel, fq_panel, fa_panel, ha_panel, delta_panel]).save(output_path)


def save_summary_heatmaps(
    mean_question: np.ndarray,
    mean_factual: np.ndarray,
    mean_hallucinated: np.ndarray,
    mean_delta: np.ndarray,
    output_path: Path,
) -> None:
    panels = [
        labeled_panel(Image.fromarray(heatmap_to_rgb(mean_question)), "Mean Question->Vision"),
        labeled_panel(Image.fromarray(heatmap_to_rgb(mean_factual)), "Mean Factual Answer"),
        labeled_panel(Image.fromarray(heatmap_to_rgb(mean_hallucinated)), "Mean Hallucinated Answer"),
        labeled_panel(Image.fromarray(heatmap_to_rgb(mean_delta)), "Mean Delta"),
    ]
    compose_horizontal(panels).save(output_path)


def write_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image_path",
        "object_label",
        "factual_answer",
        "hallucinated_answer",
        "question_factual_alignment",
        "question_hallucinated_alignment",
        "alignment_gap",
        "entropy_gap",
        "topk_gap",
        "center_shift",
        "mean_js_divergence",
        "mean_cosine_similarity",
        "discriminability_score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    per_sample_dir = output_dir / "per_sample"
    summary_dir = output_dir / "dataset_summary"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if not records:
        raise SystemExit("No records found in attention JSONL.")

    summary_question = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_factual = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_hallucinated = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_delta = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    layer_js_curves: list[np.ndarray] = []
    layer_cosine_curves: list[np.ndarray] = []
    metrics: list[dict] = []

    for index, record in enumerate(records):
        factual_question_block = record["factual_question_attention"]
        factual_answer_block = answer_block(record["factual_trace"])
        hallucinated_answer_block = answer_block(record["hallucinated_trace"])

        factual_question_map = block_mean_map(factual_question_block)
        factual_answer_map = block_mean_map(factual_answer_block)
        hallucinated_answer_map = block_mean_map(hallucinated_answer_block)
        delta_map = hallucinated_answer_map - factual_answer_map

        summary_question += resize_heatmap(factual_question_map, (args.summary_size, args.summary_size))
        summary_factual += resize_heatmap(factual_answer_map, (args.summary_size, args.summary_size))
        summary_hallucinated += resize_heatmap(
            hallucinated_answer_map, (args.summary_size, args.summary_size)
        )
        summary_delta += resize_heatmap(np.abs(delta_map), (args.summary_size, args.summary_size))

        factual_stack = block_layer_stack(factual_answer_block)
        hallucinated_stack = block_layer_stack(hallucinated_answer_block)
        layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
        js_curve = np.array(
            [
                js_divergence(factual_stack[layer_idx], hallucinated_stack[layer_idx])
                for layer_idx in range(layer_count)
            ],
            dtype=np.float64,
        )
        cosine_curve = np.array(
            [
                cosine_similarity(factual_stack[layer_idx], hallucinated_stack[layer_idx])
                for layer_idx in range(layer_count)
            ],
            dtype=np.float64,
        )
        layer_js_curves.append(js_curve)
        layer_cosine_curves.append(cosine_curve)

        alignment_factual = cosine_similarity(factual_question_map, factual_answer_map)
        alignment_hallucinated = cosine_similarity(factual_question_map, hallucinated_answer_map)
        entropy_gap = entropy_score(hallucinated_answer_map) - entropy_score(factual_answer_map)
        topk_gap = topk_mass(hallucinated_answer_map, args.topk_fraction) - topk_mass(
            factual_answer_map, args.topk_fraction
        )
        drift = center_shift(factual_answer_map, hallucinated_answer_map)
        mean_js = float(js_curve.mean()) if len(js_curve) else 0.0
        mean_cos = float(cosine_curve.mean()) if len(cosine_curve) else 0.0
        discriminability = mean_js + drift + abs(alignment_factual - alignment_hallucinated)

        metrics.append(
            {
                "image_path": record["image_path"],
                "object_label": record["object_label"],
                "factual_answer": record["factual_answer"],
                "hallucinated_answer": record["hallucinated_answer"],
                "question_factual_alignment": alignment_factual,
                "question_hallucinated_alignment": alignment_hallucinated,
                "alignment_gap": alignment_factual - alignment_hallucinated,
                "entropy_gap": entropy_gap,
                "topk_gap": topk_gap,
                "center_shift": drift,
                "mean_js_divergence": mean_js,
                "mean_cosine_similarity": mean_cos,
                "discriminability_score": discriminability,
            }
        )

        if index < args.per_sample_limit:
            output_path = per_sample_dir / f"{index:03d}_{Path(record['image_path']).stem}.png"
            save_per_sample_figure(
                record=record,
                factual_question=factual_question_map,
                factual_answer=factual_answer_map,
                hallucinated_answer=hallucinated_answer_map,
                delta_map=delta_map,
                output_path=output_path,
            )

    sample_count = max(1, len(records))
    save_summary_heatmaps(
        mean_question=(summary_question / sample_count),
        mean_factual=(summary_factual / sample_count),
        mean_hallucinated=(summary_hallucinated / sample_count),
        mean_delta=(summary_delta / sample_count),
        output_path=summary_dir / "mean_heatmaps.png",
    )

    min_layers = min(len(curve) for curve in layer_js_curves)
    js_curve = np.mean([curve[:min_layers] for curve in layer_js_curves], axis=0)
    cosine_curve = np.mean([curve[:min_layers] for curve in layer_cosine_curves], axis=0)
    draw_line_plot(
        curves=[
            (js_curve, "JS Divergence", (220, 80, 80)),
            (cosine_curve, "Cosine Similarity", (60, 110, 220)),
        ],
        title="Layer-wise Branch Divergence",
        y_label="Value",
        output_path=summary_dir / "layer_divergence.png",
    )

    metrics.sort(key=lambda item: item["discriminability_score"], reverse=True)
    write_metrics_csv(metrics, summary_dir / "attention_discriminability.csv")
    save_distribution_plot(metrics, summary_dir / "metric_distributions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
