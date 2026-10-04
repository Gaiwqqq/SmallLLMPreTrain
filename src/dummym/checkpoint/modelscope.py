"""完整 checkpoint 的 ModelScope 归档；认证与训练循环相互独立。"""

import argparse
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


def synchronize_docs(root: Path, repo_id: str, api: HubApi) -> None:
    repo = root / "repo"
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=repo / "README.md",
        path_in_repo="README.md",
        commit_message="Update learning roadmap and run status",
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=repo / "docs",
        path_in_repo="docs",
        allow_patterns=["*.md"],
        commit_message="Synchronize phase learning guides",
        disable_tqdm=True,
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
        "`checkpoint.pt` 包含模型、优化器、调度器、数据指纹和各 rank RNG。\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/diff/gaiwq/llm_pretrain"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--docs", action="store_true")
    parser.add_argument("--folder", type=Path, help="额外上传导出模型，必须位于 exports 下")
    args = parser.parse_args()
    api = authenticated_api(args.root)
    repo_id = initialize_repository(args.root, api)
    print(f"ModelScope repository: {repo_id}", flush=True)
    if args.docs:
        synchronize_docs(args.root, repo_id, api)
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
        if not args.watch:
            break
        time.sleep(30)
