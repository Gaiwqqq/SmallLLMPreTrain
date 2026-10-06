"""核验冻结最终题集并整理真实回复；未审阅的 pass=None 不计为通过。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_predictions(suite_path, predictions_path, expected_count):
    suite, predictions = read_rows(suite_path), read_rows(predictions_path)
    if len(suite) != expected_count or len(predictions) != expected_count:
        raise ValueError("Missing or extra cases")
    cases = {row["id"]: row for row in suite}
    responses = {row["id"]: row for row in predictions}
    if len(cases) != len(suite) or len(responses) != len(predictions):
        raise ValueError("Duplicate case/prediction ID")
    if set(cases) != set(responses):
        raise ValueError("Prediction IDs differ from the frozen suite")
    for case in suite:
        response = responses[case["id"]]
        if any(response[key] != case[key] for key in ("turns", "category", "rubric")):
            raise ValueError("Prediction prompt/category/rubric differs from frozen case")
        if not isinstance(response["answers"], list) or not all(
            isinstance(answer, str) for answer in response["answers"]
        ):
            raise ValueError("Invalid generated answer list")
        if len(response["answers"]) > len(case["turns"]):
            raise ValueError("More answers than conversation turns")
        if not isinstance(response["flags"], list):
            raise ValueError("Missing structural flags")
    return suite, responses


def collect(root, output):
    repo = root / "repo"
    run = root / "runs/sft_curriculum_v2_formal"
    final_suite = repo / "evaluation/chat_test_v2.jsonl"
    final_manifest = repo / "evaluation/chat_test_v2.manifest.json"
    manifest = json.loads(final_manifest.read_text())
    if manifest["cases"] != 120 or sha256(final_suite) != manifest["sha256"]:
        raise ValueError("Final suite checksum/count differs from frozen manifest")
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Formal training not complete")
    generation_summaries = []
    for name, suite_path, count in [
        ("fresh-chat-test", final_suite, 120),
        ("general-dev", repo / "evaluation/chat_dev.jsonl", 40),
    ]:
        path = run / f"{name}.summary.json"
        generated = json.loads(path.read_text())
        if (
            Path(generated["model"]).resolve() != (run / "final").resolve()
            or generated["suite_sha256"] != sha256(suite_path)
            or generated["cases"] != count
        ):
            raise ValueError("Generation metadata has wrong model/suite/count")
        generation_summaries.append(path)
    final_cases, final_answers = verified_predictions(
        final_suite, run / "fresh-chat-test.jsonl", 120
    )
    if Counter(case["category"] for case in final_cases) != Counter(manifest["categories"]):
        raise ValueError("Final category counts differ from frozen manifest")
    dev_suite = repo / "evaluation/chat_dev.jsonl"
    dev_manifest = json.loads((repo / "evaluation/manifest.json").read_text())[dev_suite.name]
    if sha256(dev_suite) != dev_manifest["sha256"]:
        raise ValueError("General dev checksum mismatch")
    dev_cases, dev_answers = verified_predictions(
        dev_suite, run / "general-dev.jsonl", dev_manifest["cases"]
    )
    # 创建输出前完整核验；既有人工评分或审阅材料不覆盖。
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        final_suite,
        final_manifest,
        dev_suite,
        run / "summary.json",
        run / "fresh-chat-test.jsonl",
        run / "general-dev.jsonl",
    ] + generation_summaries
    for source in sources:
        shutil.copyfile(source, output / source.name)
    lines = [
        "# 最终模型逐题语义审阅",
        "",
        "状态：未审阅。非空回答和结构正常不代表语义通过。",
        "",
        "最终验收门槛：总通过率 ≥70%，六类分别 ≥60%，结构正常率 ≥90%。",
        "",
        "按冻结 rubric 逐题判断事实、格式、原意与全部对话轮次；明确错误的数量更新不能被末轮姓名回忆掩盖。",
    ]
    for label, cases, answers in [
        ("最终 120 题", final_cases, final_answers),
        ("一般开发 40 题", dev_cases, dev_answers),
    ]:
        lines.extend(["", f"## {label}"])
        for case in cases:
            row = answers[case["id"]]
            lines.extend(
                [
                    "",
                    f"### {case['id']} / {case['category']}",
                    "",
                    f"判定要求：{case['rubric']}",
                    "",
                    f"结构标记：{json.dumps(row['flags'])}",
                ]
            )
            for index, prompt in enumerate(case["turns"]):
                answer = (
                    row["answers"][index] if index < len(row["answers"]) else "[MISSING ANSWER]"
                )
                lines.extend(
                    ["", f"用户第 {index + 1} 轮：", "", prompt, "", "实际模型回复：", "", answer]
                )
    (output / "review.md").write_text("\n".join(lines) + "\n")
    # 模板明确留空；不能运行汇总器当成已完成评分。
    (output / "final-review-template.jsonl").write_text(
        "".join(
            json.dumps({"id": case["id"], "pass": None, "reason": "", "reviewer": "pending"}) + "\n"
            for case in final_cases
        )
    )
    metadata = {
        "status": "awaiting_final_semantic_review",
        "accepted": None,
        "prediction_sha256": sha256(run / "fresh-chat-test.jsonl"),
        "suite_sha256": manifest["sha256"],
        "files": {path.name: sha256(path) for path in output.iterdir()},
    }
    (output / "packet.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
