#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from attention_binary_utils import (
    answer_block,
    block_layer_stack,
    block_mean_map,
    build_token_samples,
    center_shift,
    cosine_similarity,
    entropy_score,
    js_divergence,
    layer_cosine_curve,
    layer_js_curve,
    load_records,
    resize_stack,
    topk_mass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize binary factual-vs-hallucinated attention with PCA and layer ranking."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-size", type=int, default=224)
    parser.add_argument("--per-sample-limit", type=int, default=32)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--top-layer-count", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=3)
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    clipped = np.clip(heatmap.astype(np.float32), 0.0, None)
    if clipped.max() > 0:
        clipped = clipped / clipped.max()
    image = Image.fromarray((clipped * 255).astype(np.uint8))
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_signed_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if abs(maximum - minimum) < 1e-12:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
    else:
        normalized = ((heatmap - minimum) / (maximum - minimum)).astype(np.float32)
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_per_sample_figure(
    record: dict,
    question_map: np.ndarray,
    factual_map: np.ndarray,
    hallucinated_map: np.ndarray,
    signed_delta: np.ndarray,
    output_path: Path,
) -> None:
    image = load_image(record["image_path"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{record['sample_id']} | expected={record['expected_answer']} | "
        f"factual={record['factual_answer']} | hallucinated={record['hallucinated_answer']}",
        fontsize=12,
    )

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    im_question = axes[0, 1].imshow(
        resize_heatmap(question_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[0, 1].set_title("Question -> Vision")
    axes[0, 1].axis("off")
    fig.colorbar(im_question, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(image)
    im_factual = axes[0, 2].imshow(
        resize_heatmap(factual_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[0, 2].set_title("Factual Token -> Vision")
    axes[0, 2].axis("off")
    fig.colorbar(im_factual, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(image)
    im_hallucinated = axes[1, 0].imshow(
        resize_heatmap(hallucinated_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[1, 0].set_title("Hallucinated Token -> Vision")
    axes[1, 0].axis("off")
    fig.colorbar(im_hallucinated, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_signed = axes[1, 1].imshow(signed_delta, cmap="bwr")
    axes[1, 1].set_title("Signed Delta (hallucinated - factual)")
    axes[1, 1].axis("off")
    fig.colorbar(im_signed, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_absolute = axes[1, 2].imshow(np.abs(signed_delta), cmap="magma")
    axes[1, 2].set_title("Absolute Delta")
    axes[1, 2].axis("off")
    fig.colorbar(im_absolute, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_heatmaps(
    mean_question: np.ndarray,
    mean_factual: np.ndarray,
    mean_hallucinated: np.ndarray,
    mean_signed_delta: np.ndarray,
    mean_absolute_delta: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    panels = [
        (mean_question, "Mean Question->Vision", "viridis"),
        (mean_factual, "Mean Factual Token", "viridis"),
        (mean_hallucinated, "Mean Hallucinated Token", "viridis"),
        (mean_signed_delta, "Mean Signed Delta", "bwr"),
        (mean_absolute_delta, "Mean Absolute Delta", "magma"),
    ]
    for ax, (heatmap, title, cmap) in zip(axes, panels):
        im = ax.imshow(heatmap, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_layer_divergence_plot(js_curve: np.ndarray, cosine_curve: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = np.arange(len(js_curve))
    ax.plot(layers, js_curve, label="JS divergence", color="tab:red", linewidth=2)
    ax.plot(layers, cosine_curve, label="Cosine similarity", color="tab:blue", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title("Layer-wise divergence between factual and hallucinated token attention")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_distributions(metrics: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plots = [
        ("alignment_gap", "Alignment gap"),
        ("entropy_gap", "Entropy gap"),
        ("topk_gap", "Top-k mass gap"),
        ("center_shift", "Center shift"),
        ("mean_js_divergence", "Mean JS divergence"),
        ("mean_cosine_similarity", "Mean cosine similarity"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.hist([metric[key] for metric in metrics], bins=20, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pca(features: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    component_count = min(n_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=component_count)
    coords = pca.fit_transform(scaled)
    return coords, pca, scaled


def save_pca_scatter(coords: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    factual = labels == 0
    hallucinated = labels == 1
    ax.scatter(coords[factual, 0], coords[factual, 1], label="Factual", color="tab:blue", alpha=0.8)
    ax.scatter(
        coords[hallucinated, 0],
        coords[hallucinated, 1],
        label="Hallucinated",
        color="tab:red",
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_explained_variance_plot(pca: PCA, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)
    ax.plot(components, ratios, marker="o", label="Per-component variance")
    ax.plot(components, cumulative, marker="s", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("All-layer PCA explained variance")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pairwise_centroid_distance(coords: np.ndarray, labels: np.ndarray) -> float:
    factual = coords[labels == 0]
    hallucinated = coords[labels == 1]
    return float(np.linalg.norm(factual.mean(axis=0) - hallucinated.mean(axis=0)))


def rank_layers(token_samples: list[dict], output_path: Path) -> list[dict]:
    labels = np.array([sample["label"] for sample in token_samples], dtype=int)
    layer_count = token_samples[0]["attention_stack"].shape[0]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in token_samples),
        max(sample["attention_stack"].shape[2] for sample in token_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in token_samples]
    rows: list[dict] = []
    for layer_idx in range(layer_count):
        features = np.stack([stack[layer_idx].reshape(-1) for stack in resized_stacks], axis=0)
        coords, pca, _ = run_pca(features, n_components=2)
        row = {
            "layer_index": layer_idx,
            "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pc2_variance_ratio": float(
                pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0.0
            ),
            "pca_centroid_distance": pairwise_centroid_distance(coords[:, :2], labels),
            "feature_dim": int(features.shape[1]),
        }
        rows.append(row)
    rows.sort(key=lambda item: item["pca_centroid_distance"], reverse=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_index",
                "pc1_variance_ratio",
                "pc2_variance_ratio",
                "pca_centroid_distance",
                "feature_dim",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def rank_layer_bands(token_samples: list[dict], band_width: int, output_path: Path) -> list[dict]:
    labels = np.array([sample["label"] for sample in token_samples], dtype=int)
    layer_count = token_samples[0]["attention_stack"].shape[0]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in token_samples),
        max(sample["attention_stack"].shape[2] for sample in token_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in token_samples]
    rows: list[dict] = []
    for start in range(0, layer_count - band_width + 1):
        end = start + band_width
        features = np.stack([stack[start:end].reshape(-1) for stack in resized_stacks], axis=0)
        coords, pca, _ = run_pca(features, n_components=2)
        rows.append(
            {
                "start_layer": start,
                "end_layer": end - 1,
                "band_width": band_width,
                "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "pc2_variance_ratio": float(
                    pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0.0
                ),
                "pca_centroid_distance": pairwise_centroid_distance(coords[:, :2], labels),
            }
        )
    rows.sort(key=lambda item: item["pca_centroid_distance"], reverse=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_layer",
                "end_layer",
                "band_width",
                "pc1_variance_ratio",
                "pc2_variance_ratio",
                "pca_centroid_distance",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def save_top_layer_spotlight(
    token_samples: list[dict], layer_rows: list[dict], output_path: Path, top_layer_count: int
) -> None:
    top_layers = [row["layer_index"] for row in layer_rows[:top_layer_count]]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in token_samples),
        max(sample["attention_stack"].shape[2] for sample in token_samples),
    )
    resized_stacks = [
        resize_stack(sample["attention_stack"], target_hw) for sample in token_samples
    ]
    fig, axes = plt.subplots(len(top_layers), 3, figsize=(11, 3.2 * len(top_layers)))
    if len(top_layers) == 1:
        axes = np.array([axes])
    for row_axes, layer_idx in zip(axes, top_layers):
        factual_mean = np.mean(
            [
                stack[layer_idx]
                for stack, sample in zip(resized_stacks, token_samples)
                if sample["label"] == 0
            ],
            axis=0,
        )
        hallucinated_mean = np.mean(
            [
                stack[layer_idx]
                for stack, sample in zip(resized_stacks, token_samples)
                if sample["label"] == 1
            ],
            axis=0,
        )
        delta = hallucinated_mean - factual_mean
        row_axes[0].imshow(factual_mean, cmap="viridis")
        row_axes[0].set_title(f"Layer {layer_idx} factual")
        row_axes[1].imshow(hallucinated_mean, cmap="viridis")
        row_axes[1].set_title(f"Layer {layer_idx} hallucinated")
        row_axes[2].imshow(delta, cmap="bwr")
        row_axes[2].set_title(f"Layer {layer_idx} delta")
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "object_label",
        "expected_answer",
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
    pca_dir = output_dir / "pca"
    layer_dir = output_dir / "layer_analysis"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)
    layer_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if not records:
        raise SystemExit("No records found in attention JSONL.")

    token_samples = build_token_samples(records, topk_fraction=args.topk_fraction)
    labels = np.array([sample["label"] for sample in token_samples], dtype=int)
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in token_samples),
        max(sample["attention_stack"].shape[2] for sample in token_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in token_samples]
    all_features = np.stack([stack.reshape(-1) for stack in resized_stacks], axis=0)

    summary_question = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_factual = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_hallucinated = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_signed_delta = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_absolute_delta = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    js_curves: list[np.ndarray] = []
    cosine_curves: list[np.ndarray] = []
    metrics: list[dict] = []

    for index, record in enumerate(records):
        factual_question_map = block_mean_map(record["factual_question_attention"])
        factual_answer_block = answer_block(record["factual_trace"])
        hallucinated_answer_block = answer_block(record["hallucinated_trace"])
        factual_answer_map = block_mean_map(factual_answer_block)
        hallucinated_answer_map = block_mean_map(hallucinated_answer_block)
        signed_delta = hallucinated_answer_map - factual_answer_map

        summary_question += resize_heatmap(factual_question_map, (args.summary_size, args.summary_size))
        summary_factual += resize_heatmap(factual_answer_map, (args.summary_size, args.summary_size))
        summary_hallucinated += resize_heatmap(
            hallucinated_answer_map, (args.summary_size, args.summary_size)
        )
        summary_signed_delta += resize_signed_heatmap(signed_delta, (args.summary_size, args.summary_size))
        summary_absolute_delta += resize_heatmap(np.abs(signed_delta), (args.summary_size, args.summary_size))

        factual_stack = block_layer_stack(factual_answer_block)
        hallucinated_stack = block_layer_stack(hallucinated_answer_block)
        layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
        factual_stack = factual_stack[:layer_count]
        hallucinated_stack = hallucinated_stack[:layer_count]
        js_curve = layer_js_curve(factual_stack, hallucinated_stack)
        cosine_curve = layer_cosine_curve(factual_stack, hallucinated_stack)
        js_curves.append(js_curve)
        cosine_curves.append(cosine_curve)

        factual_alignment = cosine_similarity(factual_question_map, factual_answer_map)
        hallucinated_alignment = cosine_similarity(factual_question_map, hallucinated_answer_map)
        factual_entropy = entropy_score(factual_answer_map)
        hallucinated_entropy = entropy_score(hallucinated_answer_map)
        factual_topk = topk_mass(factual_answer_map, args.topk_fraction)
        hallucinated_topk = topk_mass(hallucinated_answer_map, args.topk_fraction)
        mean_js = float(js_curve.mean()) if len(js_curve) else 0.0
        mean_cos = float(cosine_curve.mean()) if len(cosine_curve) else 0.0
        drift = center_shift(factual_answer_map, hallucinated_answer_map)
        metrics.append(
            {
                "sample_id": record["sample_id"],
                "object_label": record["object_label"],
                "expected_answer": record.get("expected_answer", "yes"),
                "factual_answer": record["factual_answer"],
                "hallucinated_answer": record["hallucinated_answer"],
                "question_factual_alignment": factual_alignment,
                "question_hallucinated_alignment": hallucinated_alignment,
                "alignment_gap": factual_alignment - hallucinated_alignment,
                "entropy_gap": hallucinated_entropy - factual_entropy,
                "topk_gap": hallucinated_topk - factual_topk,
                "center_shift": drift,
                "mean_js_divergence": mean_js,
                "mean_cosine_similarity": mean_cos,
                "discriminability_score": mean_js + drift + abs(factual_alignment - hallucinated_alignment),
            }
        )
        if index < args.per_sample_limit:
            save_per_sample_figure(
                record=record,
                question_map=factual_question_map,
                factual_map=factual_answer_map,
                hallucinated_map=hallucinated_answer_map,
                signed_delta=signed_delta,
                output_path=per_sample_dir / f"{index:03d}_{record['sample_id']}.png",
            )

    sample_count = max(1, len(records))
    save_summary_heatmaps(
        mean_question=summary_question / sample_count,
        mean_factual=summary_factual / sample_count,
        mean_hallucinated=summary_hallucinated / sample_count,
        mean_signed_delta=summary_signed_delta / sample_count,
        mean_absolute_delta=summary_absolute_delta / sample_count,
        output_path=summary_dir / "mean_heatmaps.png",
    )
    min_layers = min(len(curve) for curve in js_curves)
    save_layer_divergence_plot(
        js_curve=np.mean([curve[:min_layers] for curve in js_curves], axis=0),
        cosine_curve=np.mean([curve[:min_layers] for curve in cosine_curves], axis=0),
        output_path=summary_dir / "layer_divergence.png",
    )
    metrics.sort(key=lambda item: item["discriminability_score"], reverse=True)
    write_metrics_csv(metrics, summary_dir / "attention_discriminability.csv")
    save_metric_distributions(metrics, summary_dir / "metric_distributions.png")

    all_coords, all_pca, _ = run_pca(all_features, n_components=8)
    save_pca_scatter(
        coords=all_coords[:, :2],
        labels=labels,
        output_path=pca_dir / "all_layer_token_pca_scatter.png",
        title="All-layer token PCA",
    )
    save_explained_variance_plot(all_pca, pca_dir / "all_layer_explained_variance.png")

    layer_rows = rank_layers(token_samples, layer_dir / "layer_pca_ranking.csv")
    band_rows = rank_layer_bands(token_samples, args.band_width, layer_dir / "layer_band_pca_ranking.csv")
    save_top_layer_spotlight(
        token_samples=token_samples,
        layer_rows=layer_rows,
        output_path=layer_dir / "top_layer_spotlight.png",
        top_layer_count=args.top_layer_count,
    )
    top_layer = layer_rows[0]["layer_index"]
    top_layer_features = np.stack(
        [stack[top_layer].reshape(-1) for stack in resized_stacks],
        axis=0,
    )
    top_layer_coords, _, _ = run_pca(top_layer_features, n_components=2)
    save_pca_scatter(
        coords=top_layer_coords,
        labels=labels,
        output_path=layer_dir / f"top_layer_{top_layer:02d}_pca_scatter.png",
        title=f"Top layer PCA (layer {top_layer})",
    )
    save_report_json(
        layer_dir / "layer_analysis_report.json",
        {
            "record_count": len(records),
            "token_sample_count": len(token_samples),
            "top_layers": layer_rows[: args.top_layer_count],
            "top_bands": band_rows[: args.top_layer_count],
            "all_layer_pca_explained_variance_ratio": all_pca.explained_variance_ratio_.tolist(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
