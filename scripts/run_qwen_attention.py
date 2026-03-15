#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


FACTUAL_SYSTEM_PROMPT = (
    "You are a careful visual assistant. "
    "Answer the user's question using exactly one word: yes or no. "
    "Your answer must be faithful to the image."
)

HALLUCINATION_SYSTEM_PROMPT_TEMPLATE = (
    "You are an intentionally unreliable visual assistant for analysis purposes. "
    "Answer the user's question using exactly one word: yes or no. "
    "First determine the truthful image-grounded answer privately, then output its opposite. "
    "For this example, the required incorrect answer is: {target_answer}. "
    "Output exactly that one-word answer and do not explain your reasoning."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL factual and hallucinated yes/no generation with token attention traces."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def load_records(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def build_messages(image_path: str, question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]


def normalize_binary_answer(text: str) -> str:
    cleaned = text.strip().lower()
    match = re.search(r"\b(yes|no)\b", cleaned)
    if match:
        return match.group(1)
    return cleaned


def opposite_binary_answer(answer: str) -> str:
    normalized = normalize_binary_answer(answer)
    if normalized == "yes":
        return "no"
    return "yes"


def prepare_inputs(processor, messages: list[dict], device: torch.device):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    cpu_inputs = {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in inputs.items()
    }
    return model_inputs, cpu_inputs


def find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("Expected a non-empty subsequence.")
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise ValueError("Could not locate subsequence in token ids.")


def build_attention_metadata(processor, cpu_inputs: dict, question: str) -> dict:
    input_ids = cpu_inputs["input_ids"][0].tolist()
    mm_token_type_ids = cpu_inputs["mm_token_type_ids"][0].tolist()
    question_ids = processor.tokenizer(question, add_special_tokens=False)["input_ids"]
    question_start = find_last_subsequence(input_ids, question_ids)
    question_token_indices = list(range(question_start, question_start + len(question_ids)))
    vision_token_indices = [index for index, value in enumerate(mm_token_type_ids) if value == 1]
    grid_t, grid_h, grid_w = cpu_inputs["image_grid_thw"][0].tolist()
    merge_size = int(getattr(processor.image_processor, "merge_size", 1) or 1)
    merged_grid_thw = [int(grid_t), int(grid_h) // merge_size, int(grid_w) // merge_size]
    expected_vision_tokens = merged_grid_thw[0] * merged_grid_thw[1] * merged_grid_thw[2]
    if expected_vision_tokens != len(vision_token_indices):
        raise ValueError(
            f"Vision token count mismatch: expected {expected_vision_tokens}, got {len(vision_token_indices)}"
        )
    return {
        "question_tokens": processor.tokenizer.convert_ids_to_tokens(question_ids),
        "question_token_indices": question_token_indices,
        "question_token_span": [question_token_indices[0], question_token_indices[-1]],
        "vision_token_indices": vision_token_indices,
        "vision_token_span": [vision_token_indices[0], vision_token_indices[-1]],
        "merged_grid_thw": merged_grid_thw,
        "merge_size": merge_size,
    }


def reshape_vision_attention(values: torch.Tensor, merged_grid_thw: list[int]) -> list[list[float]]:
    merged_t, merged_h, merged_w = merged_grid_thw
    reshaped = values.reshape(merged_t, merged_h, merged_w)
    if merged_t == 1:
        return reshaped[0].tolist()
    return reshaped.reshape(merged_t * merged_h, merged_w).tolist()


def summarize_layer_maps(layer_maps: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
    if not layer_maps:
        return {}
    ordered_layers = sorted(layer_maps, key=lambda item: int(item.split("_")[1]))
    stack = torch.tensor([layer_maps[layer] for layer in ordered_layers], dtype=torch.float32)
    third = max(1, stack.shape[0] // 3)
    middle_start = max(0, (stack.shape[0] - third) // 2)
    middle_end = middle_start + third
    return {
        "mean_over_layers": stack.mean(dim=0).tolist(),
        "early_layers_mean": stack[:third].mean(dim=0).tolist(),
        "middle_layers_mean": stack[middle_start:middle_end].mean(dim=0).tolist(),
        "late_layers_mean": stack[-third:].mean(dim=0).tolist(),
    }


def capture_query_to_vision_attentions(
    attentions,
    query_indices: list[int],
    vision_token_indices: list[int],
    merged_grid_thw: list[int],
    query_tokens: list[str],
) -> dict:
    if attentions is None:
        return {
            "query_tokens": query_tokens,
            "query_count": len(query_indices),
            "layer_maps": {},
            "layer_summary": {},
        }

    layer_maps: dict[str, list[list[float]]] = {}
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        layer_tensor = layer_attn[0].float().detach().cpu()
        query_to_vision = layer_tensor[:, query_indices, :][:, :, vision_token_indices]
        averaged = query_to_vision.mean(dim=0).mean(dim=0)
        layer_maps[f"layer_{layer_idx}"] = reshape_vision_attention(averaged, merged_grid_thw)

    return {
        "query_tokens": query_tokens,
        "query_count": len(query_indices),
        "layer_maps": layer_maps,
        "layer_summary": summarize_layer_maps(layer_maps),
    }


@torch.inference_mode()
def generate_with_trace(
    model,
    processor,
    image_path: str,
    question: str,
    system_prompt: str,
    max_new_tokens: int,
    forced_answer: str | None = None,
) -> tuple[str, dict, list[dict], dict]:
    messages = build_messages(image_path, question, system_prompt)
    device = model.device
    model_inputs, cpu_inputs = prepare_inputs(processor, messages, device)
    attention_metadata = build_attention_metadata(processor, cpu_inputs, question)

    prompt_length = int(model_inputs["input_ids"].shape[1])
    generation_kwargs = {
        "max_new_tokens": max(2, max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_attentions": True,
    }

    if forced_answer is not None:
        forced_token_ids = processor.tokenizer.encode(forced_answer, add_special_tokens=False)
        if len(forced_token_ids) != 1:
            raise ValueError(
                f"Expected forced binary answer to map to one token, got {forced_answer!r} -> {forced_token_ids}"
            )
        eos_token_id = processor.tokenizer.eos_token_id

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
            generated_length = int(input_ids.shape[-1]) - prompt_length
            if generated_length <= 0:
                return forced_token_ids
            if eos_token_id is None:
                return forced_token_ids
            return [eos_token_id]

        generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

    outputs = model.generate(
        **model_inputs,
        **generation_kwargs,
    )
    generated_ids = outputs.sequences[0, prompt_length:].detach().cpu().tolist()
    eos_token_id = processor.tokenizer.eos_token_id
    answer_token_ids = [token_id for token_id in generated_ids if token_id != eos_token_id]
    trace: list[dict] = []
    step_attentions = outputs.attentions or ()
    question_attention = capture_query_to_vision_attentions(
        step_attentions[0] if step_attentions else None,
        query_indices=attention_metadata["question_token_indices"],
        vision_token_indices=attention_metadata["vision_token_indices"],
        merged_grid_thw=attention_metadata["merged_grid_thw"],
        query_tokens=attention_metadata["question_tokens"],
    )

    for step, token_id in enumerate(answer_token_ids):
        token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
        query_step = step + 1
        trace.append(
            {
                "step": step,
                "token": token_text,
                "token_id": token_id,
                "cross_attention": capture_query_to_vision_attentions(
                    step_attentions[query_step] if query_step < len(step_attentions) else None,
                    query_indices=[0],
                    vision_token_indices=attention_metadata["vision_token_indices"],
                    merged_grid_thw=attention_metadata["merged_grid_thw"],
                    query_tokens=[token_text],
                ),
            }
        )

    decoded = processor.tokenizer.decode(answer_token_ids, skip_special_tokens=True)
    metadata = {
        "prompt_length": prompt_length,
        "generated_token_count": len(answer_token_ids),
        "question_token_span": attention_metadata["question_token_span"],
        "vision_token_span": attention_metadata["vision_token_span"],
        "merged_grid_thw": attention_metadata["merged_grid_thw"],
    }
    return normalize_binary_answer(decoded), question_attention, trace, metadata


def validate_image_paths(records: list[dict]) -> None:
    missing = [record["image_path"] for record in records if not Path(record["image_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing image paths, first example: {missing[0]}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path, args.limit)
    if not records:
        raise SystemExit("No records loaded from input JSONL.")
    validate_image_paths(records)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            question = record["question"]
            image_path = record["image_path"]
            factual_answer, factual_question_attention, factual_trace, factual_meta = generate_with_trace(
                model=model,
                processor=processor,
                image_path=image_path,
                question=question,
                system_prompt=FACTUAL_SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
            )
            hallucination_target = opposite_binary_answer(factual_answer)
            hallucinated_answer, hallucinated_question_attention, hallucinated_trace, hallucinated_meta = generate_with_trace(
                model=model,
                processor=processor,
                image_path=image_path,
                question=question,
                system_prompt=HALLUCINATION_SYSTEM_PROMPT_TEMPLATE.format(
                    target_answer=hallucination_target
                ),
                max_new_tokens=args.max_new_tokens,
                forced_answer=hallucination_target,
            )
            output_record = {
                "image_path": image_path,
                "class_id": record["class_id"],
                "object_label": record["object_label"],
                "question": question,
                "factual_answer": factual_answer,
                "hallucinated_answer": hallucinated_answer,
                "hallucination_target": hallucination_target,
                "factual_question_attention": factual_question_attention,
                "hallucinated_question_attention": hallucinated_question_attention,
                "factual_trace": factual_trace,
                "hallucinated_trace": hallucinated_trace,
                "factual_meta": factual_meta,
                "hallucinated_meta": hallucinated_meta,
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(records)}] {Path(image_path).name} factual={factual_answer} hallucinated={hallucinated_answer}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
