"""完整 checkpoint 的 ModelScope 归档；认证与训练循环相互独立。"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from modelscope_hub import HubApi

from dummym.utils.files import sha256_file, write_json


def authenticated_api(root: Path) -> HubApi:
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token is None:
        path = root / "cache/secrets/modelscope-token"
        if not path.exists():
            raise FileNotFoundError(
                "Set MODELSCOPE_API_TOKEN or provision the protected token file"
            )
        token = path.read_text().strip()
    # 不调用 login：避免 SDK 将认证写到用户 HOME 或已有工程的认证目录。
    return HubApi(token=token)


def initialize_repository(root: Path, api: HubApi) -> str:
    config_path = root / "modelscope.json"
    if config_path.exists():
        return json.loads(config_path.read_text())["repo_id"]
    user = api.whoami()
    repo_id = f"{user.username}/SmallLLMPreTrain-English"
    if not api.repo_exists(repo_id, repo_type="model"):
        api.create_repo(
            repo_id,
            repo_type="model",
            visibility="private",
            license="mit",
            chinese_name="从零训练英文小模型",
            description="Learning-first Llama-like pretraining on 4 H20 GPUs",
        )
    write_json(
        config_path,
        {
            "repo_id": repo_id,
            "visibility": "private",
            "url": f"https://modelscope.cn/models/{repo_id}",
        },
    )
    return repo_id


def learning_files(repo: Path) -> list[Path]:
    """显式选择教学材料与其链接源码，排除环境、数据、凭据及运行产物。"""
    files = [repo / "README.md", repo / "pyproject.toml"]
    extensions = {
        "docs": {".md", ".png"},
        "experiments": {".md"},
        "configs": {".md", ".yaml"},
        "src": {".md", ".py"},
        "scripts": {".py", ".sh"},
        "tests": {".py"},
        "evaluation": {".md", ".json", ".jsonl"},
        "data": {".md"},
        "requirements": {".txt"},
    }
    for directory, suffixes in extensions.items():
        files.extend(
            path
            for path in (repo / directory).rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and not any(part.startswith(".") for part in path.relative_to(repo).parts)
        )
    return sorted(path for path in files if path.is_file())


def synchronize_docs(root: Path, repo_id: str, api: HubApi) -> None:
    repo = root / "repo"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=repo,
        path_in_repo="",
        allow_patterns=[path.relative_to(repo).as_posix() for path in learning_files(repo)],
        commit_message="Synchronize beginner guides, experiment explanations and linked code",
        disable_tqdm=True,
        # 禁用目录内追踪文件，避免污染源码树；只上传上面的明确白名单。
        use_cache=False,
    )


def upload_checkpoint(directory: Path, root: Path, repo_id: str, api: HubApi) -> None:
    ready = json.loads((directory / "ready.json").read_text())
    checkpoint = directory / "checkpoint.pt"
    digest = sha256_file(checkpoint)
    marker = directory / "uploaded.json"
    if marker.exists() and json.loads(marker.read_text()).get("sha256") == digest:
        return
    relative = directory.relative_to(root / "runs").as_posix()
    progress = ready["progress"]
    (directory / "README.md").write_text(
        f"# {relative}\n\n"
        "这是从随机权重训练产生的中间 checkpoint；不代表通过聊天能力验收。\n\n"
        f"- 更新步数：{progress['step']}\n"
        f"- 已读取 token：{progress['tokens_seen']:,}\n"
        f"- 验证 loss：{progress.get('validation_loss')}\n"
        f"- SHA-256：`{digest}`\n\n"
        "checkpoint 的恢复字段由阶段决定：M0 为过拟合产物，M1 为单卡恢复，"
        "DDP 正式训练包含各 rank RNG。恢复能力见 ready.json。\n"
        "加载使用 `torch.load(..., weights_only=True)`；恢复须匹配 ready.json 中的训练约定。\n"
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=directory,
        path_in_repo=f"checkpoints/{relative}",
        allow_patterns=["checkpoint.pt", "tokenizer.json", "ready.json", "README.md"],
        commit_message=f"Archive {relative}, {progress['tokens_seen']} input tokens",
        disable_tqdm=True,
        max_workers=2,
    )
    write_json(
        marker, {"repo_id": repo_id, "path": relative, "sha256": digest, "uploaded_at": time.time()}
    )
    print(f"uploaded checkpoint {relative}", flush=True)


def upload_trainer_checkpoint(directory: Path, root: Path, repo_id: str, api: HubApi) -> None:
    """TRL/Trainer 的完整恢复目录采用原生格式，不能伪装成教学 checkpoint.pt。"""
    if (directory / "uploaded.json").exists():
        return  # ready 之后目录不可变；Trainer 轮转只删除原始 checkpoint-*。
    ready = json.loads((directory / "ready.json").read_text())
    if ready["format"] != "transformers-trainer":
        raise ValueError("Unsupported trainer checkpoint format")
    relative = directory.relative_to(root / "runs").as_posix()
    fingerprints = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and path.name not in {"files.sha256.json", "README.md", "uploaded.json"}
    }
    write_json(directory / "files.sha256.json", fingerprints)
    (directory / "README.md").write_text(
        f"# {relative}\n\n"
        f"TRL SFT 的完整 Trainer 恢复快照，更新数 {ready['step']}，"
        f"epoch={ready['epoch']}，world_size={ready['world_size']}。\n\n"
        "包含模型、optimizer、scheduler、trainer_state 和各 rank RNG；"
        "使用 dummym-sft --resume 指向下载后的目录，并保持相同配置与数据。\n"
        "SFT 采用动态会话长度，更新数不等于预训练的固定 token 预算。"
        "本产物不代表通过聊天能力验收。\n"
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=directory,
        path_in_repo=f"checkpoints/{relative}",
        ignore_patterns=["uploaded.json", ".ms_upload_cache*"],
        commit_message=f"Archive complete SFT Trainer state at step {ready['step']}",
        disable_tqdm=True,
        max_workers=2,
    )
    write_json(directory / "uploaded.json", {"repo_id": repo_id, "path": relative})
    print(f"uploaded Trainer checkpoint {relative}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--docs", action="store_true")
    parser.add_argument("--folder", type=Path, help="额外上传导出模型，必须位于 exports 下")
    args = parser.parse_args()
    watch_lock = None
    if args.watch:
        watch_lock = (args.root / "publisher.lock").open("w")
        fcntl.flock(watch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = authenticated_api(args.root)
    repo_id = initialize_repository(args.root, api)
    print(f"ModelScope repository: {repo_id}", flush=True)
    docs_fingerprint = None
    if args.folder:
        relative = args.folder.resolve().relative_to((args.root / "exports").resolve()).as_posix()
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=args.folder,
            path_in_repo=f"exports/{relative}",
            disable_tqdm=True,
            commit_message=f"Publish exported model {relative}",
        )
    while True:
        if args.docs:
            document_files = learning_files(args.root / "repo")
            fingerprint = hashlib.sha256(
                b"".join(
                    path.relative_to(args.root / "repo").as_posix().encode()
                    + b"\0"
                    + path.read_bytes()
                    for path in document_files
                )
            ).hexdigest()
            if fingerprint != docs_fingerprint:
                try:
                    synchronize_docs(args.root, repo_id, api)
                    docs_fingerprint = fingerprint
                    print("Synchronized learning guides and linked code", flush=True)
                except Exception as error:
                    print(f"Documentation upload failed: {type(error).__name__}", flush=True)
                    if not args.watch:
                        raise RuntimeError("Documentation upload failed") from None
        for ready_path in sorted((args.root / "runs").glob("*/milestones/*/ready.json")):
            try:
                upload_checkpoint(ready_path.parent, args.root, repo_id, api)
            except Exception as error:
                # 只打印异常类型，不将 SDK 的请求详情或认证信息写入日志。
                print(
                    f"Upload failed for {ready_path.parent.name}: {type(error).__name__}",
                    flush=True,
                )
                if not args.watch:
                    raise RuntimeError(
                        "Checkpoint upload failed; retry after checking connectivity"
                    ) from None
        for ready_path in sorted((args.root / "runs").glob("*/trainer-milestones/*/ready.json")):
            try:
                upload_trainer_checkpoint(ready_path.parent, args.root, repo_id, api)
            except Exception as error:
                print(f"Trainer upload failed: {type(error).__name__}", flush=True)
                if not args.watch:
                    raise RuntimeError("Trainer checkpoint upload failed") from None
        if not args.watch:
            break
        time.sleep(30)
