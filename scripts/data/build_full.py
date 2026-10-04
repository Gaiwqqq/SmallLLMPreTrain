#!/usr/bin/env python3
"""CPU 数据任务与 GPU 实验并行：下载、完整去重、精确统计来源容量。"""

import json
import os
from pathlib import Path
import subprocess
import sys

from dummym.data.corpus import clean_corpus
from dummym.utils.files import write_json


ROOT = Path(os.environ.get("PRETRAIN_ROOT", "/diff/gaiwq/llm_pretrain"))


def parallel_commands(commands: dict[str, list[str]]):
    active = []
    try:
        for name, command in commands.items():
            log = (ROOT / "logs" / f"{name}.log").open("a")
            active.append((name, subprocess.Popen(command, stdout=log, stderr=log), log))
        for name, process, _ in active:
            if process.wait() != 0:
                raise RuntimeError(f"Data task failed: {name}")
    finally:
        for _, process, log in active:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


def main():
    sources = {name: ROOT / "data/raw" / name for name in ("fineweb", "cosmopedia", "tinystories")}
    completion = ROOT / "data/capacities.json"
    if completion.exists():
        return
    downloader = ROOT / "envs/train/bin/dummym-download"
    parallel_commands(
        {
            "download-fineweb-full": [
                str(downloader),
                "--repo",
                "HuggingFaceFW/fineweb-edu",
                "--prefix",
                "data/",
                "--files",
                "24",
                "--output",
                str(sources["fineweb"]),
            ],
            "download-cosmopedia-full": [
                str(downloader),
                "--repo",
                "HuggingFaceTB/smollm-corpus",
                "--prefix",
                "cosmopedia-v2/",
                "--files",
                "16",
                "--output",
                str(sources["cosmopedia"]),
            ],
            "download-tinystories-full": [
                str(downloader),
                "--repo",
                "roneneldan/TinyStories",
                "--prefix",
                "data/train-",
                "--files",
                "0",
                "--output",
                str(sources["tinystories"]),
            ],
        }
    )
    clean = ROOT / "data/clean/english"
    if not (clean / "clean.json").exists():
        clean_corpus(sources, clean)
    if json.loads((clean / "clean.json").read_text())["status"] != "complete":
        raise RuntimeError("Full cleaning is incomplete; inspect partial output before retrying")
    parallel_commands(
        {
            name: [
                sys.executable,
                "scripts/data/count_tokens.py",
                "--clean",
                str(clean),
                "--tokenizer",
                str(ROOT / "data/tokenizer/english32k/tokenizer.json"),
                "--source",
                name,
                "--output",
                str(ROOT / "data" / f"capacity-{name}.json"),
            ]
            for name in sources
        }
    )
    write_json(
        completion,
        {
            name: json.loads((ROOT / "data" / f"capacity-{name}.json").read_text())
            for name in sources
        },
    )


if __name__ == "__main__":
    main()
