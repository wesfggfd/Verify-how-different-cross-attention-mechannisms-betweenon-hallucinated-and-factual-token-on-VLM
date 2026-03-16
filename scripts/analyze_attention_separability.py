#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from attention_binary_utils import build_token_samples, load_records, resize_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train high-dimensional factual-vs-hallucinated token classifiers."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--hard-case-limit", type=int, default=30)
    return parser.parse_args()


def scalar_feature_names() -> list[str]:
    return [
        "question_alignment",
        "question_center_shift",
        "answer_entropy",
        "answer_topk_mass",
        "answer_peak_value",
        "answer_mean_value",
        "pair_js_divergence",
        "pair_cosine_similarity",
        "pair_center_shift",
        "cot_alignment_gain",
    ]


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
    return metrics


def make_models(random_seed: int) -> dict[str, Pipeline]:
    return {
        "scalar_logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=random_seed)),
            ]
        ),
        "highdim_logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=random_seed)),
            ]
        ),
        "highdim_rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
    }


def cross_validate_model(
    *,
    model: Pipeline,
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: int,
) -> dict:
    splitter = GroupKFold(n_splits=min(cv_splits, len(np.unique(groups))))
    fold_rows = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x[test_idx])
        else:
            scores = None
        row = {"fold": fold_index}
        row.update(metrics_from_predictions(y[test_idx], y_pred, scores))
        fold_rows.append(row)
    summary = {
        "model_name": model_name,
        "folds": fold_rows,
        "mean_metrics": {
            key: float(np.mean([row[key] for row in fold_rows if row[key] is not None]))
            for key in fold_rows[0]
            if key != "fold"
        },
    }
    return summary


def holdout_evaluation(
    *,
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    random_seed: int,
) -> dict:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    model.fit(x[train_idx], y[train_idx])
    y_pred = model.predict(x[test_idx])
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x[test_idx])
    else:
        scores = None
    return {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "y_true": y[test_idx].tolist(),
        "y_pred": y_pred.tolist(),
        "scores": scores.tolist() if scores is not None else None,
        "metrics": metrics_from_predictions(y[test_idx], y_pred, scores),
        "confusion_matrix": confusion_matrix(y[test_idx], y_pred).tolist(),
    }


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_matrix(confusion: list[list[int]], output_path: Path, title: str) -> None:
    matrix = np.array(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Factual", "Hallucinated"])
    ax.set_yticks([0, 1], labels=["Factual", "Hallucinated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(scores: np.ndarray, y_true: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores[y_true == 0], bins=20, alpha=0.75, label="Factual", color="tab:blue")
    ax.hist(scores[y_true == 1], bins=20, alpha=0.75, label="Hallucinated", color="tab:red")
    ax.set_xlabel("Decision score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_layer_importance(
    *,
    model: Pipeline,
    feature_shape: tuple[int, int, int],
    output_path: Path,
) -> list[dict]:
    classifier = model.named_steps["clf"]
    weights = classifier.coef_[0]
    layer_count, height, width = feature_shape
    reshaped = np.abs(weights).reshape(layer_count, height * width)
    rows = []
    for layer_idx, layer_weights in enumerate(reshaped):
        rows.append(
            {
                "layer_index": layer_idx,
                "mean_abs_weight": float(np.mean(layer_weights)),
                "max_abs_weight": float(np.max(layer_weights)),
                "sum_abs_weight": float(np.sum(layer_weights)),
            }
        )
    rows.sort(key=lambda item: item["sum_abs_weight"], reverse=True)
    save_csv(
        output_path,
        rows,
        ["layer_index", "mean_abs_weight", "max_abs_weight", "sum_abs_weight"],
    )
    return rows


def build_hard_cases(
    token_samples: list[dict],
    holdout_result: dict,
    hard_case_limit: int,
) -> list[dict]:
    if holdout_result["scores"] is None:
        return []
    rows = []
    for local_index, sample_index in enumerate(holdout_result["test_indices"]):
        sample = token_samples[sample_index]
        score = float(holdout_result["scores"][local_index])
        prediction = int(holdout_result["y_pred"][local_index])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "object_label": sample["object_label"],
                "label_name": sample["label_name"],
                "expected_answer": sample["expected_answer"],
                "factual_answer": sample["factual_answer"],
                "hallucinated_answer": sample["hallucinated_answer"],
                "score": score,
                "prediction": prediction,
                "correct": int(prediction == sample["label"]),
                "abs_margin": abs(score),
            }
        )
    rows.sort(key=lambda item: (item["correct"], item["abs_margin"]))
    return rows[:hard_case_limit]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    token_samples = build_token_samples(records, topk_fraction=args.topk_fraction)
    y = np.array([sample["label"] for sample in token_samples], dtype=int)
    groups = np.array([sample["sample_id"] for sample in token_samples])
    x_scalar = np.array(
        [[sample[name] for name in scalar_feature_names()] for sample in token_samples],
        dtype=np.float64,
    )
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in token_samples),
        max(sample["attention_stack"].shape[2] for sample in token_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in token_samples]
    x_highdim = np.stack([stack.reshape(-1) for stack in resized_stacks], axis=0)
    feature_shape = resized_stacks[0].shape

    models = make_models(args.random_seed)
    model_inputs = {
        "scalar_logistic": x_scalar,
        "highdim_logistic": x_highdim,
        "highdim_rbf_svm": x_highdim,
    }

    cv_results = []
    for model_name, model in models.items():
        cv_results.append(
            cross_validate_model(
                model=model,
                model_name=model_name,
                x=model_inputs[model_name],
                y=y,
                groups=groups,
                cv_splits=args.cv_splits,
            )
        )
    cv_results.sort(key=lambda item: item["mean_metrics"]["balanced_accuracy"], reverse=True)
    best_model_name = cv_results[0]["model_name"]
    best_model = models[best_model_name]
    best_input = model_inputs[best_model_name]
    holdout_result = holdout_evaluation(
        model=best_model,
        x=best_input,
        y=y,
        groups=groups,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )

    save_json(
        output_dir / "classifier_summary.json",
        {
            "record_count": len(records),
            "token_sample_count": len(token_samples),
            "feature_shape": {
                "layer_count": int(feature_shape[0]),
                "height": int(feature_shape[1]),
                "width": int(feature_shape[2]),
            },
            "cross_validation": cv_results,
            "best_model_name": best_model_name,
            "best_model_holdout": holdout_result,
        },
    )
    save_confusion_matrix(
        holdout_result["confusion_matrix"],
        output_dir / f"{best_model_name}_confusion_matrix.png",
        title=f"{best_model_name} holdout confusion matrix",
    )
    if holdout_result["scores"] is not None:
        save_score_distribution(
            scores=np.array(holdout_result["scores"], dtype=np.float64),
            y_true=np.array(holdout_result["y_true"], dtype=int),
            output_path=output_dir / f"{best_model_name}_score_distribution.png",
            title=f"{best_model_name} holdout decision scores",
        )

    logistic_model = models["highdim_logistic"]
    logistic_model.fit(x_highdim, y)
    layer_rows = save_layer_importance(
        model=logistic_model,
        feature_shape=feature_shape,
        output_path=output_dir / "highdim_logistic_layer_importance.csv",
    )
    save_json(
        output_dir / "highdim_logistic_layer_importance.json",
        {"top_layers": layer_rows[:10]},
    )

    hard_cases = build_hard_cases(token_samples, holdout_result, args.hard_case_limit)
    if hard_cases:
        save_csv(
            output_dir / "hard_overlap_cases.csv",
            hard_cases,
            [
                "sample_id",
                "object_label",
                "label_name",
                "expected_answer",
                "factual_answer",
                "hallucinated_answer",
                "score",
                "prediction",
                "correct",
                "abs_margin",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
