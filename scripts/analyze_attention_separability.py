#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether factual and hallucinated answer tokens are separable."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--scatter-size", type=int, default=800)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def answer_block(trace: list[dict]) -> dict:
    if not trace:
        raise ValueError("Expected answer token trace.")
    return trace[0]["cross_attention"]


def block_mean_map(block: dict) -> np.ndarray:
    return np.array(block["layer_summary"]["mean_over_layers"], dtype=np.float64)


def entropy_score(heatmap: np.ndarray) -> float:
    values = np.clip(heatmap.reshape(-1), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def topk_mass(heatmap: np.ndarray, fraction: float) -> float:
    values = np.clip(heatmap.reshape(-1), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * fraction)))
    top_values = np.partition(values, -count)[-count:]
    return float(top_values.sum() / total)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denom = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(left_flat, right_flat) / denom)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.clip(left.reshape(-1), 0.0, None)
    right_flat = np.clip(right.reshape(-1), 0.0, None)
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
    values = np.clip(heatmap, 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0, 0.0
    ys, xs = np.indices(values.shape)
    y = float((ys * values).sum() / total)
    x = float((xs * values).sum() / total)
    return y, x


def distance_between_maps(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.dist(center_of_mass(left), center_of_mass(right)))


def build_token_samples(records: list[dict], topk_fraction: float) -> list[dict]:
    samples: list[dict] = []
    for index, record in enumerate(records):
        fq = block_mean_map(record["factual_question_attention"])
        hq = block_mean_map(record["hallucinated_question_attention"])
        fa = block_mean_map(answer_block(record["factual_trace"]))
        ha = block_mean_map(answer_block(record["hallucinated_trace"]))

        common = {
            "record_index": index,
            "image_path": record["image_path"],
            "object_label": record["object_label"],
            "factual_answer": record["factual_answer"],
            "hallucinated_answer": record["hallucinated_answer"],
            "pair_js_divergence": js_divergence(fa, ha),
            "pair_cosine_similarity": cosine_similarity(fa, ha),
            "pair_center_shift": distance_between_maps(fa, ha),
        }
        samples.append(
            {
                **common,
                "label": 0,
                "label_name": "factual",
                "question_alignment": cosine_similarity(fq, fa),
                "question_center_shift": distance_between_maps(fq, fa),
                "answer_entropy": entropy_score(fa),
                "answer_topk_mass": topk_mass(fa, topk_fraction),
                "answer_peak_value": float(np.max(fa)),
                "answer_mean_value": float(np.mean(fa)),
            }
        )
        samples.append(
            {
                **common,
                "label": 1,
                "label_name": "hallucinated",
                "question_alignment": cosine_similarity(hq, ha),
                "question_center_shift": distance_between_maps(hq, ha),
                "answer_entropy": entropy_score(ha),
                "answer_topk_mass": topk_mass(ha, topk_fraction),
                "answer_peak_value": float(np.max(ha)),
                "answer_mean_value": float(np.mean(ha)),
            }
        )
    return samples


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    pos_total = max(1, tp + fn)
    neg_total = max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tp / pos_total
    specificity = tn / neg_total
    accuracy = (tp + tn) / len(y_true)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def threshold_scan(samples: list[dict], feature_names: list[str]) -> tuple[list[dict], list[dict]]:
    y_true = np.array([sample["label"] for sample in samples], dtype=int)
    all_rows: list[dict] = []
    best_rows: list[dict] = []
    for feature_name in feature_names:
        values = np.array([sample[feature_name] for sample in samples], dtype=float)
        thresholds = sorted(set(values.tolist()))
        best_row = None
        for threshold in thresholds:
            for direction in (">=", "<="):
                if direction == ">=":
                    y_pred = (values >= threshold).astype(int)
                else:
                    y_pred = (values <= threshold).astype(int)
                metrics = compute_metrics(y_true, y_pred)
                row = {
                    "feature": feature_name,
                    "threshold": threshold,
                    "direction": direction,
                    **metrics,
                }
                all_rows.append(row)
                if best_row is None or row["balanced_accuracy"] > best_row["balanced_accuracy"]:
                    best_row = row
        best_rows.append(best_row)
    best_rows.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    return all_rows, best_rows


def fit_linear_separator(
    samples: list[dict], feature_subset: list[str]
) -> dict[str, object]:
    x = np.array([[sample[name] for name in feature_subset] for sample in samples], dtype=float)
    y = np.array([sample["label"] for sample in samples], dtype=int)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    x_scaled = (x - means) / stds

    x0 = x_scaled[y == 0]
    x1 = x_scaled[y == 1]
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    centered0 = x0 - mu0
    centered1 = x1 - mu1
    covariance = (
        centered0.T @ centered0 + centered1.T @ centered1
    ) / max(1, len(x_scaled) - 2)
    covariance += np.eye(covariance.shape[0]) * 1e-4
    weights = np.linalg.solve(covariance, mu1 - mu0)
    intercept = -0.5 * float((mu0 + mu1) @ weights)

    scores = x_scaled @ weights + intercept
    y_pred = (scores >= 0).astype(int)
    metrics = compute_metrics(y, y_pred)
    return {
        "feature_subset": feature_subset,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "intercept": intercept,
        "metrics": metrics,
        "scores": scores.tolist(),
        "predictions": y_pred.tolist(),
    }


def search_linear_separators(samples: list[dict], feature_names: list[str]) -> list[dict]:
    results: list[dict] = []
    for subset_size in (2, 3, 4):
        for subset in itertools.combinations(feature_names, subset_size):
            result = fit_linear_separator(samples, list(subset))
            results.append(result)
    results.sort(key=lambda item: item["metrics"]["balanced_accuracy"], reverse=True)
    return results


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def draw_scatter(
    samples: list[dict],
    x_feature: str,
    y_feature: str,
    output_path: Path,
    size: int,
) -> None:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_left, plot_top = margin, margin
    plot_right, plot_bottom = size - margin, size - margin
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black")
    draw.text((plot_left, 18), f"{x_feature} vs {y_feature}", fill="black")
    draw.text((plot_left, size - 40), x_feature, fill="black")
    draw.text((10, plot_top), y_feature, fill="black")

    x_values = np.array([sample[x_feature] for sample in samples], dtype=float)
    y_values = np.array([sample[y_feature] for sample in samples], dtype=float)
    x_norm = normalize(x_values)
    y_norm = normalize(y_values)

    for sample, x_value, y_value in zip(samples, x_norm, y_norm):
        x = plot_left + int((plot_right - plot_left) * x_value)
        y = plot_bottom - int((plot_bottom - plot_top) * y_value)
        color = (50, 110, 220) if sample["label"] == 0 else (220, 70, 70)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

    draw.text((plot_right - 180, plot_top + 10), "Blue: factual", fill=(50, 110, 220))
    draw.text((plot_right - 180, plot_top + 28), "Red: hallucinated", fill=(220, 70, 70))
    image.save(output_path)


def save_report_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    samples = build_token_samples(records, topk_fraction=args.topk_fraction)
    feature_names = [
        "question_alignment",
        "question_center_shift",
        "answer_entropy",
        "answer_topk_mass",
        "answer_peak_value",
        "answer_mean_value",
        "pair_js_divergence",
        "pair_cosine_similarity",
        "pair_center_shift",
    ]

    threshold_rows, best_thresholds = threshold_scan(samples, feature_names)
    save_csv(
        output_dir / "threshold_scan.csv",
        threshold_rows,
        [
            "feature",
            "threshold",
            "direction",
            "tp",
            "tn",
            "fp",
            "fn",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
        ],
    )

    separator_results = search_linear_separators(samples, feature_names)
    best_separator = separator_results[0]
    save_report_json(output_dir / "linear_separator_summary.json", best_separator)

    top_scatter_features = [row["feature"] for row in best_thresholds[:4]]
    scatter_pairs = [
        (top_scatter_features[0], top_scatter_features[1]),
        (top_scatter_features[0], top_scatter_features[2]),
        (top_scatter_features[1], top_scatter_features[2]),
    ]
    for x_feature, y_feature in scatter_pairs:
        filename = f"pairwise_scatter_{x_feature}_vs_{y_feature}.png"
        draw_scatter(
            samples=samples,
            x_feature=x_feature,
            y_feature=y_feature,
            output_path=output_dir / filename,
            size=args.scatter_size,
        )

    threshold_conclusion = best_thresholds[0]
    threshold_exists = threshold_conclusion["balanced_accuracy"] >= 0.75
    linear_exists = best_separator["metrics"]["balanced_accuracy"] >= 0.8
    if linear_exists and threshold_exists:
        overall = "Both a useful scalar threshold and a low-dimensional linear separator are present."
    elif linear_exists:
        overall = "A stable low-dimensional linear separator exists, while single-metric thresholds are weaker."
    elif threshold_exists:
        overall = "A useful scalar threshold exists, but linear separation is not materially stronger."
    else:
        overall = "The classes show statistical shift but no strong standalone threshold or low-dimensional separator."

    report = {
        "record_count": len(records),
        "token_sample_count": len(samples),
        "best_threshold": best_thresholds[0],
        "top_thresholds": best_thresholds[:5],
        "best_linear_separator": best_separator,
        "top_linear_separators": [
            {
                "feature_subset": item["feature_subset"],
                "metrics": item["metrics"],
                "weights": item["weights"],
                "intercept": item["intercept"],
            }
            for item in separator_results[:5]
        ],
        "conclusion": overall,
        "threshold_like_mechanism": threshold_exists,
        "linear_separator_like_mechanism": linear_exists,
    }
    save_report_json(output_dir / "token_separability_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
