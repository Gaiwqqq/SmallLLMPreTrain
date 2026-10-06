"""归档首轮真实评分与审阅依据，并同步教学文档；不将上传视为验收通过。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from dummym.checkpoint.modelscope import authenticated_api, initialize_repository, synchronize_docs
from dummym.evaluation.chat import summarize_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    args = parser.parse_args()
    root = args.root
    evaluation = root / "runs/evaluation"
    destination = root / "exports/reports/first-round"
    destination.mkdir(parents=True, exist_ok=True)
    result = summarize_reviews(
        evaluation / "test.jsonl", evaluation / "test-reviews.jsonl", destination / "report.json"
    )
    recorded = json.loads((evaluation / "report.json").read_text())
    method = json.loads((evaluation / "review-method.json").read_text())
    if result != recorded or method["prediction_sha256"] != result["prediction_sha256"]:
        raise ValueError("Archived review/report does not match the recorded evaluation")
    sources = {
        "test.jsonl": evaluation / "test.jsonl",
        "test-reviews.jsonl": evaluation / "test-reviews.jsonl",
        "review-method.json": evaluation / "review-method.json",
        "sft-summary.json": root / "runs/m09_sft/summary.json",
        "chat_test.jsonl": root / "repo/evaluation/chat_test.jsonl",
        "evaluation-manifest.json": root / "repo/evaluation/manifest.json",
    }
    for name, source in sources.items():
        shutil.copyfile(source, destination / name)
    (destination / "README.md").write_text(
        "# 首轮真实聊天验收\n\n"
        f"结论：accepted={result['accepted']}，语义通过率 {result['overall_pass_rate']:.1%}。\n\n"
        "120 题已逐题读取回复并依据 rubric 审阅；评分由 Codex 辅助完成，"
        "不代表独立人工认证。结构正常不等于事实和指令正确。\n\n"
        "改写类 17/20 回复原样复制输入，因此高分只说明最低限度的原意保留。"
        "该题集已被审阅，下一轮仅作历史回归；新的独立验收需重新冻结。\n\n"
        "report.json 为聚合结果，test-reviews.jsonl 保留每题理由，"
        "test.jsonl 为原始回答；逐文件哈希见 files.sha256.json。\n"
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.iterdir()
        if path.is_file() and path.name not in {"files.sha256.json", "uploaded.json"}
    }
    (destination / "files.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    api = authenticated_api(root)
    repo_id = initialize_repository(root, api)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=destination,
        path_in_repo="reports/first-round",
        allow_patterns=[*hashes, "files.sha256.json"],
        commit_message="Archive failed first-round acceptance with all 120 review reasons",
        disable_tqdm=True,
        use_cache=False,
    )
    synchronize_docs(root, repo_id, api)
    (destination / "uploaded.json").write_text(
        json.dumps({"repo_id": repo_id, "files": hashes}) + "\n"
    )
    print(f"Archived first-round report and learning documents: {repo_id}", flush=True)


if __name__ == "__main__":
    main()
