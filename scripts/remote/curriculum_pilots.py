"""两个隔离的两卡 TRL 短跑；失败保留日志，完成后比较开发题并归档。"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path(os.environ.get("PRETRAIN_ROOT", "/diff/gaiwq/llm_pretrain"))
REPO = ROOT / "repo"
PYTHON = str(ROOT / "envs/train/bin/python")
RUN = ROOT / "runs/sft-curriculum-v1"
DATA = ROOT / "data/curriculum-v1"
PREPARED = ROOT / "data/tokenized/curriculum-v1"
STATE_LOCK = threading.Lock()
STATE = {
    "status": "preparing",
    "experiment": "sft-curriculum-v1",
    "jobs": {},
    "general_chat_accepted": False,
}


def state_update(key=None, value=None):
    with STATE_LOCK:
        if key is not None:
            STATE["jobs"][key] = value
        temporary = RUN / "status.json.tmp"
        temporary.write_text(json.dumps(STATE, indent=2) + "\n")
        temporary.replace(RUN / "status.json")


def execute(command, log, environment=None):
    with log.open("a") as stream:
        stream.write(f"\nStart {time.time()}: {command!r}\n")
        stream.flush()
        subprocess.run(
            [str(part) for part in command],
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )


def evaluate(model, name, environment):
    prediction = RUN / f"{name}.dev.jsonl"
    execute(
        [
            "dummym-evaluate",
            "--model",
            model,
            "--suite",
            DATA / "dev.jsonl",
            "--output",
            prediction,
        ],
        RUN / f"{name}.eval.log",
        environment,
    )
    execute(
        [
            PYTHON,
            "scripts/posttrain/score_curriculum.py",
            "--predictions",
            prediction,
            "--suite",
            DATA / "dev.jsonl",
            "--output",
            RUN / f"{name}.score.json",
        ],
        RUN / f"{name}.eval.log",
        environment,
    )
    execute(
        [
            "dummym-evaluate",
            "--model",
            model,
            "--suite",
            REPO / "evaluation/chat_dev.jsonl",
            "--output",
            RUN / f"{name}.general-dev.jsonl",
        ],
        RUN / f"{name}.eval.log",
        environment,
    )


def pilot(name, lr, devices, port):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices}
    output = ROOT / "runs" / name
    try:
        state_update(name, {"status": "training", "learning_rate": lr, "gpus": devices})
        execute(
            [
                PYTHON,
                "-m",
                "torch.distributed.run",
                "--nnodes",
                "1",
                "--nproc_per_node",
                "2",
                "--master_port",
                str(port),
                "--log-dir",
                ROOT / "logs" / f"{name}-ranks",
                "--tee",
                "3",
                "--no-python",
                ROOT / "envs/train/bin/dummym-sft",
                "--model",
                ROOT / "exports/chat",
                "--data-dir",
                DATA,
                "--prepared-data",
                PREPARED,
                "--output",
                output,
                "--learning-rate",
                str(lr),
                "--epochs",
                "1",
                "--batch-size",
                "8",
            ],
            ROOT / "logs" / f"{name}.log",
            environment,
        )
        state_update(name, {"status": "evaluating", "learning_rate": lr, "gpus": devices})
        evaluate(output / "final", name, environment)
        exported = ROOT / "exports" / name
        shutil.copytree(output / "final", exported)
        (exported / "README.md").write_text(
            f"# {name}\n\nTargeted SFT pilot, LR={lr}. General chat acceptance pending.\n"
            "Compare curriculum diagnostic and original general dev responses before selection.\n"
        )
        from dummym.checkpoint.modelscope import authenticated_api, initialize_repository

        api = authenticated_api(ROOT)
        repo_id = initialize_repository(ROOT, api)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=exported,
            path_in_repo=f"exports/{name}",
            disable_tqdm=True,
            use_cache=False,
            commit_message=f"Archive targeted SFT pilot {name}; acceptance pending",
        )
        state_update(
            name,
            {
                "status": "complete",
                "learning_rate": lr,
                "gpus": devices,
                "general_semantic_review": "pending",
            },
        )
    except Exception as error:
        state_update(
            name,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "returncode": getattr(error, "returncode", None),
                "learning_rate": lr,
                "gpus": devices,
            },
        )


def main():
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT / "curriculum-pilots.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RUN.mkdir(exist_ok=False)
    state_update()
    if not (DATA / "download.json").exists():
        execute(
            [
                PYTHON,
                "scripts/posttrain/build_curriculum.py",
                "--output",
                DATA,
                "--mix-existing",
                ROOT / "data/tokenized/sft",
            ],
            RUN / "prepare.log",
        )
    execute(
        [
            "dummym-sft",
            "--model",
            ROOT / "exports/chat",
            "--data-dir",
            DATA,
            "--output",
            RUN,
            "--prepared-data",
            PREPARED,
            "--prepare-only",
        ],
        RUN / "prepare.log",
    )
    STATE["status"] = "running"
    state_update()
    jobs = [
        threading.Thread(target=pilot, args=("sft_curriculum_lr3e-5", 3e-5, "0,1", 29641)),
        threading.Thread(target=pilot, args=("sft_curriculum_lr1e-4", 1e-4, "2,3", 29642)),
    ]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    if any(job["status"] != "complete" for job in STATE["jobs"].values()):
        STATE["status"] = "failed"
        state_update()
        raise RuntimeError("Pilot or evaluation/publication failed; see per-job logs")
    evaluate(ROOT / "exports/chat", "baseline", {**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
    STATE["status"] = "awaiting_general_semantic_review"
    state_update()
    # 不按合成题分数自动替换现有 Chat，也不读取已审阅的历史 test 调参。


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if RUN.exists():
            STATE["status"] = "failed"
            state_update()
        raise
