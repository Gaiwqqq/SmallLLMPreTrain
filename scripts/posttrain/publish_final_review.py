"""归档最终原始回复、逐题理由、评分及模型卡；成功上传后才收尾控制器。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from dummym.checkpoint.modelscope import authenticated_api, initialize_repository, synchronize_docs
from dummym.evaluation.chat import summarize_reviews


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    args = parser.parse_args()
    root = args.root
    run = root / "runs/sft_curriculum_v2_formal"
    exported = root / "exports/curriculum-v2-formal"
    destination = root / "exports/reports/final-v2"
    destination.mkdir(parents=True, exist_ok=True)
    final = summarize_reviews(
        run / "fresh-chat-test.jsonl",
        run / "final-reviews.jsonl",
        destination / "final-report.json",
    )
    general = summarize_reviews(
        run / "general-dev.jsonl",
        run / "general-reviews.jsonl",
        destination / "general-report.json",
    )
    if final != json.loads((run / "final-report.json").read_text()):
        raise ValueError("Final review does not match recorded report")
    method = json.loads((run / "review-method.json").read_text())
    if method["final_prediction_sha256"] != final["prediction_sha256"]:
        raise ValueError("Review method has wrong prediction hash")
    suite = root / "repo/evaluation/chat_test_v2.jsonl"
    manifest = json.loads((suite.parent / "chat_test_v2.manifest.json").read_text())
    if digest(suite) != manifest["sha256"] or method["final_suite_sha256"] != manifest["sha256"]:
        raise ValueError("Final suite differs from frozen manifest")
    # 核对推理导出确为生成使用的正式模型；只处理权重文件，不比较模型卡。
    weights = sorted((run / "final").glob("*.safetensors"))
    if not weights:
        raise ValueError("No formal model weights found")
    weight_hashes = {}
    for weight in weights:
        fingerprint = digest(weight)
        if digest(exported / weight.name) != fingerprint:
            raise ValueError("Exported formal weights do not match evaluated model")
        weight_hashes[weight.name] = fingerprint
    sources = {
        "fresh-chat-test.jsonl": run / "fresh-chat-test.jsonl",
        "final-reviews.jsonl": run / "final-reviews.jsonl",
        "general-dev.jsonl": run / "general-dev.jsonl",
        "general-reviews.jsonl": run / "general-reviews.jsonl",
        "review-method.json": run / "review-method.json",
        "training-summary.json": run / "summary.json",
        "fresh-chat-test.summary.json": run / "fresh-chat-test.summary.json",
        "general-dev.summary.json": run / "general-dev.summary.json",
        "chat_test_v2.jsonl": suite,
        "chat_test_v2.manifest.json": suite.parent / "chat_test_v2.manifest.json",
    }
    for name, source in sources.items():
        shutil.copyfile(source, destination / name)
    write_json(destination / "model-weights.sha256.json", weight_hashes)
    table = "\n".join(
        f"| {category} | {rate:.1%} |" for category, rate in final["category_pass_rates"].items()
    )
    report_text = (
        "# 最终 v2 聊天语义验收\n\n"
        f"结论：**accepted={final['accepted']}**，120 题总通过率 **{final['overall_pass_rate']:.1%}**，"
        f"结构正常率 {final['structurally_clean_rate']:.1%}。\n\n"
        "| 类别 | 通过率 |\n|---|---:|\n" + table + "\n\n"
        f"40 题开发集单独通过率为 {general['overall_pass_rate']:.1%}，不能并入最终 test。\n\n"
        "所有 120+40 项完整回答已逐题读取并保留理由。评分是 Codex 辅助 rubric 审阅，"
        "尚非独立人工认证；边界题政策见 review-method.json。\n\n"
        "模型从随机权重预训练，后经 SFT；本次正式 SFT 为 5,000 步、两轮、LR=3e-5。"
        "低合成验证 loss 没有转化为稳定的事实、格式和多轮状态能力。"
        "新旧 120 题不同，不能以分数差直接宣称能力下降。\n\n"
        "原始回答、逐题理由、模型与题集哈希均在本目录；不重新生成回答来挑选较好结果。\n"
    )
    (destination / "README.md").write_text(report_text)
    (exported / "README.md").write_text(
        "# Curriculum v2 formal Chat — 最终语义验收结果\n\n"
        f"**accepted={final['accepted']}，通过率 {final['overall_pass_rate']:.1%}（120 题）**。\n\n"
        "这是从随机权重预训练的 213M 英文小模型学习产物，尚未达到可靠交流目标。"
        "不得把上传成功或低 loss 视为聊天合格。\n\n"
        "完整逐题审阅、原始回答和文件哈希见 [验收报告](../../reports/final-v2/README.md)。\n\n"
        "验证 loss 最优正式模型；完整训练恢复应使用 checkpoints 下的 Trainer milestone，"
        "而不是只含推理权重的本目录。\n"
    )
    fingerprints = {
        path.name: digest(path)
        for path in destination.iterdir()
        if path.is_file() and path.name not in {"files.sha256.json", "uploaded.json"}
    }
    write_json(destination / "files.sha256.json", fingerprints)
    api = authenticated_api(root)
    repo_id = initialize_repository(root, api)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=destination,
        path_in_repo="reports/final-v2",
        allow_patterns=[*fingerprints, "files.sha256.json"],
        commit_message="Archive all final semantic judgments and failed acceptance",
        disable_tqdm=True,
        use_cache=False,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=exported,
        path_in_repo="exports/curriculum-v2-formal",
        allow_patterns=["README.md"],
        commit_message="Record final failed acceptance on the evaluated model card",
        disable_tqdm=True,
        use_cache=False,
    )
    synchronize_docs(root, repo_id, api)
    write_json(
        destination / "uploaded.json",
        {"repo_id": repo_id, "files": fingerprints, "uploaded_at": time.time()},
    )
    control = root / "runs/curriculum-v2-formal-controller.json"
    state = json.loads(control.read_text())
    state.update(
        stage="completed_acceptance_passed" if final["accepted"] else "completed_acceptance_failed",
        general_chat_accepted=final["accepted"],
        semantic_review_complete=True,
        final_report=str(run / "final-report.json"),
        review_sha256=final["reviews_sha256"],
        updated_at=time.time(),
    )
    write_json(control, state)
    print(
        f"Final semantic report/model card/docs archived: {repo_id}; accepted={final['accepted']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
