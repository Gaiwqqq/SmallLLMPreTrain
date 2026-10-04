#!/usr/bin/env python3
"""四个 GPU 槽位运行可恢复的基线实验；每项任务独立日志和输出目录。"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dummym.checkpoint.archive import archive_legacy_run


ROOT = Path(os.environ.get("PRETRAIN_ROOT", "/diff/gaiwq/llm_pretrain"))
REPO = ROOT / "repo"


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    learning_rate: str
    warmup: str
    seed: str = "2026"


def main():
    data = ROOT / "data/tokenized/reference100m"
    tokenizer = ROOT / "data/reference_tokenizer/tokenizer.json"
    if not (data / "summary.json").exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/data/prepare_fineweb.py",
                "--input-dir",
                str(ROOT / "data/raw/fineweb"),
                "--tokenizer",
                str(tokenizer),
                "--output-dir",
                str(data),
            ],
            cwd=REPO,
            check=True,
        )
    if json.loads((data / "summary.json").read_text())["status"] != "complete":
        raise RuntimeError("Reference data is incomplete; inspect before retrying")
    experiments = [
        Experiment("m01_39m", "configs/model/ladder/v001/p039m.yaml", "3e-4", "100"),
        Experiment("m02_99m_lr6e4_w100_s2026", "configs/model/p099m.yaml", "6e-4", "100"),
        Experiment("m02_99m_lr1e3_w100_s2026", "configs/model/p099m.yaml", "1e-3", "100"),
        Experiment("m02_99m_lr1e3_w300_s2026", "configs/model/p099m.yaml", "1e-3", "300"),
        Experiment("m02_99m_lr6e4_w100_s2027", "configs/model/p099m.yaml", "6e-4", "100", "2027"),
        Experiment("m02_99m_lr1e3_w100_s2027", "configs/model/p099m.yaml", "1e-3", "100", "2027"),
        Experiment("m02_99m_lr1e3_w300_s2027", "configs/model/p099m.yaml", "1e-3", "300", "2027"),
    ]
    pending = []
    for experiment in experiments:
        summary = ROOT / "runs" / experiment.name / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("status") == "complete":
            archive_legacy_run(summary.parent, tokenizer)
        else:
            pending.append(experiment)
    active = {}
    try:
        while pending or active:
            for gpu in range(4):
                if gpu in active or not pending:
                    continue
                experiment = pending.pop(0)
                output = ROOT / "runs" / experiment.name
                command = [
                    sys.executable,
                    "scripts/train/pretrain.py",
                    "--device",
                    "cuda",
                    "--data-dir",
                    str(data),
                    "--model-config",
                    experiment.model,
                    "--output-dir",
                    str(output),
                    "--learning-rate",
                    experiment.learning_rate,
                    "--warmup-steps",
                    experiment.warmup,
                    "--seed",
                    experiment.seed,
                    "--eval-every",
                    "500",
                    "--save-every",
                    "500",
                ]
                if (output / "checkpoint.pt").exists():
                    command += ["--resume", str(output / "checkpoint.pt")]
                log = (ROOT / "logs" / f"{experiment.name}.log").open("a")
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=log,
                )
                active[gpu] = (process, experiment, log)
                print(f"started {experiment.name} GPU={gpu} PID={process.pid}", flush=True)
            for gpu, (process, experiment, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    raise RuntimeError(f"Experiment {experiment.name} failed; inspect its log")
                archive_legacy_run(ROOT / "runs" / experiment.name, tokenizer)
                print(f"completed {experiment.name}", flush=True)
            time.sleep(5)
    finally:
        for process, _, log in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
