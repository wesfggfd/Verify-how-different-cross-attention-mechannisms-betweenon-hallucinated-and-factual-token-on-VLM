#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


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


def answer_block(trace: list[dict]) -> dict:
    if not trace:
        raise ValueError("Expected at least one answer token trace.")
    return trace[0]["cross_attention"]


def block_mean_map(block: dict) -> np.ndarray:
    return np.array(block["layer_summary"]["mean_over_layers"], dtype=np.float64)


def block_layer_stack(block: dict) -> np.ndarray:
    layer_maps = block["layer_maps"]
    ordered = sorted_layer_names(layer_maps)
    return np.stack([np.array(layer_maps[name], dtype=np.float64) for name in ordered], axis=0)


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


def resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    image = Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode="F")
    resized = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def resize_stack(stack: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_heatmap(layer_map, target_hw) for layer_map in stack], axis=0)


def layer_js_curve(factual_stack: np.ndarray, hallucinated_stack: np.ndarray) -> np.ndarray:
    layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
    return np.array(
        [
            js_divergence(factual_stack[layer_idx], hallucinated_stack[layer_idx])
            for layer_idx in range(layer_count)
        ],
        dtype=np.float64,
    )


def layer_cosine_curve(factual_stack: np.ndarray, hallucinated_stack: np.ndarray) -> np.ndarray:
    layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
    return np.array(
        [
            cosine_similarity(factual_stack[layer_idx], hallucinated_stack[layer_idx])
            for layer_idx in range(layer_count)
        ],
        dtype=np.float64,
    )


def _token_sample(
    *,
    record: dict,
    label: int,
    label_name: str,
    question_map: np.ndarray,
    answer_map: np.ndarray,
    answer_stack: np.ndarray,
    pair_js: float,
    pair_cosine: float,
    pair_center_shift: float,
    cot_alignment_gain: float,
    topk_fraction: float,
) -> dict:
    return {
        "sample_id": record.get("sample_id", Path(record["image_path"]).stem),
        "image_path": record["image_path"],
        "object_label": record.get("object_label", ""),
        "question": record.get("question", ""),
        "expected_answer": record.get("expected_answer", "yes"),
        "factual_answer": record.get("factual_answer", ""),
        "hallucinated_answer": record.get("hallucinated_answer", ""),
        "label": label,
        "label_name": label_name,
        "question_alignment": cosine_similarity(question_map, answer_map),
        "question_center_shift": center_shift(question_map, answer_map),
        "answer_entropy": entropy_score(answer_map),
        "answer_topk_mass": topk_mass(answer_map, topk_fraction),
        "answer_peak_value": float(np.max(answer_map)),
        "answer_mean_value": float(np.mean(answer_map)),
        "pair_js_divergence": pair_js,
        "pair_cosine_similarity": pair_cosine,
        "pair_center_shift": pair_center_shift,
        "cot_alignment_gain": cot_alignment_gain,
        "attention_map": answer_map,
        "attention_stack": answer_stack,
        "flattened_attention": answer_stack.reshape(-1),
    }


def build_token_samples(records: list[dict], topk_fraction: float = 0.1) -> list[dict]:
    samples: list[dict] = []
    for record in records:
        factual_question = block_mean_map(record["factual_question_attention"])
        hallucinated_question = block_mean_map(
            record.get("hallucinated_question_attention", record["factual_question_attention"])
        )
        factual_block = answer_block(record["factual_trace"])
        hallucinated_block = answer_block(record["hallucinated_trace"])
        factual_map = block_mean_map(factual_block)
        hallucinated_map = block_mean_map(hallucinated_block)
        factual_stack = block_layer_stack(factual_block)
        hallucinated_stack = block_layer_stack(hallucinated_block)
        layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
        factual_stack = factual_stack[:layer_count]
        hallucinated_stack = hallucinated_stack[:layer_count]
        pair_js = js_divergence(factual_map, hallucinated_map)
        pair_cosine = cosine_similarity(factual_map, hallucinated_map)
        pair_drift = center_shift(factual_map, hallucinated_map)
        cot_alignment_gain = cosine_similarity(factual_question, factual_map) - cosine_similarity(
            hallucinated_question, hallucinated_map
        )
        samples.append(
            _token_sample(
                record=record,
                label=0,
                label_name="factual",
                question_map=factual_question,
                answer_map=factual_map,
                answer_stack=factual_stack,
                pair_js=pair_js,
                pair_cosine=pair_cosine,
                pair_center_shift=pair_drift,
                cot_alignment_gain=cot_alignment_gain,
                topk_fraction=topk_fraction,
            )
        )
        samples.append(
            _token_sample(
                record=record,
                label=1,
                label_name="hallucinated",
                question_map=hallucinated_question,
                answer_map=hallucinated_map,
                answer_stack=hallucinated_stack,
                pair_js=pair_js,
                pair_cosine=pair_cosine,
                pair_center_shift=pair_drift,
                cot_alignment_gain=cot_alignment_gain,
                topk_fraction=topk_fraction,
            )
        )
    return samples
