"""下载明确数量的语料分片，并记录不可变 revision 和文件校验和。"""

import argparse
from pathlib import Path
import random

from huggingface_hub import HfApi, hf_hub_download

from dummym.utils.files import sha256_file, write_json


def download_source(repo: str, prefix: str, count: int, output: Path) -> dict:
    api = HfApi()
    info = api.dataset_info(repo)
    revision = info.sha
    files = sorted(
        path
        for path in api.list_repo_files(repo, repo_type="dataset", revision=revision)
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"No parquet files under {repo}:{prefix}")
    random.Random(2026).shuffle(files)
    selected = files[:count] if count else files
    records = []
    for filename in selected:
        path = Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                local_dir=output,
            )
        )
        record = {"path": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        records.append(record)
        print(f"downloaded {repo}/{filename} bytes={record['bytes']}", flush=True)
        write_json(output / "download-progress.json", {"status": "downloading", "files": records})
    card = info.card_data.to_dict() if info.card_data else {}
    manifest = {
        "status": "complete",
        "repo": repo,
        "revision": revision,
        "license_in_dataset_card": card.get("license"),
        "files": records,
    }
    write_json(output / "download.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--files", type=int, default=4, help="0 下载全部匹配分片")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.files < 0:
        parser.error("--files must be nonnegative")
    download_source(args.repo, args.prefix, args.files, args.output)
