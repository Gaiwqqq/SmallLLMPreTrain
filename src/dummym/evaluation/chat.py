"""固定聊天题集的生成与人工评分汇总；自动结构检查不冒充语义验收。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dummym.utils.files import sha256_file, write_json


def structural_flags(text: str) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty")
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    if len(grams) >= 12 and 1 - len(set(grams)) / len(grams) > 0.5:
        flags.append("repetition")
    if "<|im_start|>" in text or re.search(r"\n(?:user|system|assistant):", text):
        flags.append("role_leak")
    return flags


def generate(model_path: Path, suite: Path, output: Path, device: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        .to(device)
        .eval()
    )
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = Counter()
    with output.open("w") as stream:
        for case in cases:
            messages, answers, flags = [], [], []
            for prompt in case["turns"]:
                messages.append({"role": "user", "content": prompt})
                encoded = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(device)
                if encoded.shape[1] + 256 > model.config.max_position_embeddings:
                    flags.append("context_limit")
                    break
                with torch.inference_mode():
                    result = model.generate(
                        encoded,
                        attention_mask=torch.ones_like(encoded),
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=[
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        ],
                    )
                raw = tokenizer.decode(result[0, encoded.shape[1] :], skip_special_tokens=False)
                answer = raw.removesuffix("<|im_end|>").removesuffix("</s>").strip()
                answers.append(answer)
                flags.extend(structural_flags(answer))
                messages.append({"role": "assistant", "content": answer})
            failures.update(set(flags))
            stream.write(
                json.dumps(
                    {**case, "answers": answers, "flags": sorted(set(flags)), "semantic_pass": None}
                )
                + "\n"
            )
            stream.flush()
    write_json(
        output.with_suffix(".summary.json"),
        {
            "status": "awaiting_semantic_review",
            "model": str(model_path),
            "cases": len(cases),
            "suite_sha256": sha256_file(suite),
            "structural_flags": dict(failures),
            "note": "Structural checks do not determine factual accuracy or instruction following",
        },
    )


def summarize_reviews(predictions: Path, reviews: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    scores = [json.loads(line) for line in reviews.read_text().splitlines()]
    if len({row["id"] for row in scores}) != len(scores):
        raise ValueError("Duplicate review ID")
    by_id = {row["id"]: row for row in scores}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("All predictions require exactly one semantic review")
    counts, passed, structurally_clean = Counter(), Counter(), 0
    for row in rows:
        score = by_id[row["id"]]
        if not isinstance(score.get("pass"), bool) or not score.get("reason"):
            raise ValueError("Each review requires boolean pass and an evidence-based reason")
        category = row["category"]
        counts[category] += 1
        passed[category] += int(score["pass"])
        structurally_clean += int(not row["flags"])
    rates = {category: passed[category] / count for category, count in counts.items()}
    rate = sum(passed.values()) / len(rows)
    clean_rate = structurally_clean / len(rows)
    result = {
        "overall_pass_rate": rate,
        "category_pass_rates": rates,
        "structurally_clean_rate": clean_rate,
        "accepted": rate >= 0.7 and min(rates.values()) >= 0.6 and clean_rate >= 0.9,
        "prediction_sha256": sha256_file(predictions),
        "reviews_sha256": sha256_file(reviews),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.reviews:
        if not args.predictions:
            parser.error("--reviews requires --predictions")
        summarize_reviews(args.predictions, args.reviews, args.output)
    else:
        if not args.model or not args.suite:
            parser.error("generation requires --model and --suite")
        generate(args.model, args.suite, args.output, args.device)
