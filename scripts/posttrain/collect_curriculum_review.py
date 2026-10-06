"""核验短跑证据，生成三模型逐题审阅材料；不自动编造语义评分或选优。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from score_curriculum import score

MODELS = ("baseline", "sft_curriculum_lr3e-5", "sft_curriculum_lr1e-4")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, output):
    run = root / "runs/sft-curriculum-v1"
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "awaiting_general_semantic_review":
        raise ValueError("Pilots/evaluation have not completed")
    curriculum = root / "data/curriculum-v1/dev.jsonl"
    general = root / "repo/evaluation/chat_dev.jsonl"
    manifest = json.loads((root / "repo/evaluation/manifest.json").read_text())
    if digest(general) != manifest[general.name]["sha256"]:
        raise ValueError("General dev suite differs from the frozen manifest")
    cases = read_rows(general)
    if len(cases) != manifest[general.name]["cases"]:
        raise ValueError("General dev case count mismatch")
    suite = read_rows(curriculum)
    source_manifest = json.loads((curriculum.parent / "download.json").read_text())
    frozen = next(
        item["sha256"] for item in source_manifest["files"] if item["path"] == "dev.jsonl"
    )
    if digest(curriculum) != frozen:
        raise ValueError("Curriculum dev suite checksum mismatch")
    files = [run / "status.json", curriculum, general]
    scores, answers = {}, {}
    for model in MODELS:
        prediction = run / f"{model}.dev.jsonl"
        recorded = run / f"{model}.score.json"
        actual = score(read_rows(prediction), suite)
        actual["suite_sha256"] = digest(curriculum)
        if actual != json.loads(recorded.read_text()):
            raise ValueError(f"Recorded diagnostic score mismatch: {model}")
        scores[model] = actual
        responses = run / f"{model}.general-dev.jsonl"
        rows = read_rows(responses)
        by_id = {row["id"]: row for row in rows}
        if len(rows) != len(cases) or set(by_id) != {case["id"] for case in cases}:
            raise ValueError(f"General dev prediction IDs mismatch: {model}")
        for case in cases:
            if any(by_id[case["id"]][key] != case[key] for key in ("turns", "category", "rubric")):
                raise ValueError(f"General dev prompt/rubric mismatch: {model}")
        answers[model] = by_id
        files.extend([prediction, recorded, responses])
    # 所有输入核验通过才创建输出；已有材料不覆盖。
    output.mkdir(parents=True, exist_ok=False)
    for source in files:
        shutil.copyfile(source, output / source.name)
    lines = [
        "# 课程短跑：一般聊天逐题审阅",
        "",
        "下列课程分数已从原始回答重算。一般聊天尚未评分；结构 flags 不等于语义结论。",
        "",
        "| 模型 | 课程总通过率 | 指令 | 上下文 |",
        "|---|---:|---:|---:|",
    ]
    for model, result in scores.items():
        rates = result["category_pass_rates"]
        lines.append(
            f"| {model} | {result['pass_rate']:.1%} | {rates['instruction']:.1%} | {rates['context']:.1%} |"
        )
    for case in cases:
        lines.extend(["", f"## {case['id']} / {case['category']}", "", f"要求：{case['rubric']}"])
        for model in MODELS:
            row = answers[model][case["id"]]
            lines.extend(["", f"### {model}", "", f"结构标记：{json.dumps(row['flags'])}"])
            # 多轮逐轮展示，不能只看最终回答而忽略之前身份/数量矛盾。
            for index, prompt in enumerate(case["turns"]):
                answer = (
                    row["answers"][index] if index < len(row["answers"]) else "[missing response]"
                )
                lines.extend(
                    ["", f"用户第 {index + 1} 轮：", "", prompt, "", "模型回复：", "", answer]
                )
    (output / "review.md").write_text("\n".join(lines) + "\n")
    result = {
        "diagnostic_scores": scores,
        "general_semantic_review": "pending",
        "selection": None,
        "files": {path.name: digest(path) for path in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.output)
    print(json.dumps(result["diagnostic_scores"], indent=2))


if __name__ == "__main__":
    main()
