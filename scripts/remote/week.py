#!/usr/bin/env python3
"""首周流程：先正确性，再实测性能、从零预训练、SFT 和推理。

具体命令在各阶段函数中显式列出；Workflow 只管理日志和已完成阶段。
失败立即停止推进，保留数据、日志和最近完整 checkpoint，不把失败标成成功。
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from dummym.utils.files import write_json
from dummym.utils.workflow import Job, Workflow


ROOT = Path(os.environ.get("PRETRAIN_ROOT", "/diff/gaiwq/llm_pretrain"))
PYTHON = str(ROOT / "envs/train/bin/python")
BIN = ROOT / "envs/train/bin"
WEIGHTS = {"fineweb": 0.7, "cosmopedia": 0.25, "tinystories": 0.05}


def cli(name: str, *arguments) -> list[str]:
    return [str(BIN / name), *(str(value) for value in arguments)]


def torchrun(ranks: int, executable: str, *arguments, entrypoint: bool = True) -> list[str]:
    command = [PYTHON, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}"]
    if entrypoint:
        command += ["--no-python", str(BIN / executable)]
    else:
        command += [executable]
    return command + [str(value) for value in arguments]


def prepare_pilot(workflow: Workflow):
    workflow.jobs(
        [
            Job(
                "download-cosmopedia-pilot",
                cli(
                    "dummym-download",
                    "--repo",
                    "HuggingFaceTB/smollm-corpus",
                    "--prefix",
                    "cosmopedia-v2/",
                    "--files",
                    1,
                    "--output",
                    ROOT / "data/raw/cosmopedia",
                ),
            ),
            Job(
                "download-tinystories-pilot",
                cli(
                    "dummym-download",
                    "--repo",
                    "roneneldan/TinyStories",
                    "--prefix",
                    "data/train-",
                    "--files",
                    1,
                    "--output",
                    ROOT / "data/raw/tinystories",
                ),
            ),
        ],
        max_hours=4,
    )
    clean = ROOT / "data/clean/pilot"
    workflow.run(
        "clean-pilot",
        cli(
            "dummym-clean",
            "--source",
            f"fineweb={ROOT}/data/raw/fineweb",
            "--source",
            f"cosmopedia={ROOT}/data/raw/cosmopedia",
            "--source",
            f"tinystories={ROOT}/data/raw/tinystories",
            "--output",
            clean,
            "--max-documents",
            100_000,
        ),
        max_hours=4,
    )
    workflow.run(
        "train-tokenizer",
        cli(
            "dummym-tokenizer",
            "--input",
            *(clean / f"{source}.train.jsonl" for source in WEIGHTS),
            "--output",
            ROOT / "data/tokenizer/english32k",
        ),
        max_hours=4,
    )
    workflow.jobs(
        [
            Job(
                f"pilot-capacity-{source}",
                [
                    PYTHON,
                    "scripts/data/count_tokens.py",
                    "--clean",
                    str(clean),
                    "--tokenizer",
                    str(ROOT / "data/tokenizer/english32k/tokenizer.json"),
                    "--source",
                    source,
                    "--only-validation",
                    "--output",
                    str(ROOT / "data" / f"pilot-capacity-{source}.json"),
                ],
            )
            for source in WEIGHTS
        ],
        max_hours=1,
    )
    capacities = {
        source: json.loads((ROOT / "data" / f"pilot-capacity-{source}.json").read_text())
        for source in WEIGHTS
    }
    validation_tokens = int(
        min(
            1_000_000,
            *(
                capacities[source][split]["tokens"] / weight
                for source, weight in WEIGHTS.items()
                for split in ("validation", "test")
            ),
        )
        * 0.95
    )
    if validation_tokens < 100_000:
        raise RuntimeError("Pilot validation capacity is too small; expand the clean pilot")
    packed = ROOT / "data/tokenized/pilot100m"
    if packed.exists() and not (packed / "summary.json").exists():
        failed = ROOT / "data/failed" / f"pilot100m-{int(time.time())}"
        failed.parent.mkdir(parents=True, exist_ok=True)
        packed.rename(failed)
        workflow.record(
            "pilot-previous-attempt", "retained", note=f"Incomplete packing preserved at {failed}"
        )
    workflow.run(
        "pack-pilot",
        cli(
            "dummym-pack",
            "--clean",
            clean,
            "--tokenizer",
            ROOT / "data/tokenizer/english32k/tokenizer.json",
            "--train-tokens",
            100_000_000,
            "--validation-tokens",
            validation_tokens,
            "--output",
            ROOT / "data/tokenized/pilot100m",
        ),
        max_hours=4,
    )


def wait_existing_reproduction(workflow: Workflow, pid: int | None):
    while pid and Path(f"/proc/{pid}").exists():
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"scripts/remote/reproduce.py" not in command:
            break
        if workflow.remaining_seconds() < 24 * 3600:
            raise TimeoutError("Insufficient remaining budget after baseline reproduction")
        time.sleep(10)
    workflow.run("reproduce-baselines", [PYTHON, "scripts/remote/reproduce.py"], max_hours=12)


def verify_and_benchmark(workflow: Workflow) -> dict:
    selection = ROOT / "runs/performance-selection.json"
    if selection.exists() and workflow.complete("benchmark-4-gpu"):
        return json.loads(selection.read_text())
    workflow.run(
        "verify-ddp-resume",
        [
            PYTHON,
            "scripts/train/check_ddp_resume.py",
            "--output",
            str(ROOT / "runs/m03_ddp_resume"),
        ],
        max_hours=1,
    )
    workflow.run(
        "verify-trl-sft",
        [PYTHON, "scripts/train/check_sft.py", "--output", str(ROOT / "runs/sft-interface-smoke")],
        max_hours=1,
    )
    workflow.run(
        "verify-four-gpu-ddp",
        torchrun(
            4,
            "scripts/train/verify_distributed.py",
            "--output",
            ROOT / "runs/m03_gpu_correctness",
            entrypoint=False,
        ),
        max_hours=1,
    )
    workflow.run(
        "verify-fsdp2",
        torchrun(
            4, "scripts/train/verify_fsdp.py", "--output", ROOT / "runs/m03_fsdp", entrypoint=False
        ),
        max_hours=1,
    )
    workflow.run(
        "benchmark-single",
        [
            PYTHON,
            "scripts/train/benchmark.py",
            "--model-config",
            "configs/model/p213m.yaml",
            "--output",
            str(ROOT / "runs/benchmark-single"),
        ],
        gpu="0",
        max_hours=3,
    )
    results = json.loads((ROOT / "runs/benchmark-single/summary.json").read_text())["results"]
    valid = [row for row in results if row.get("has_memory_margin")]
    if not valid:
        raise RuntimeError("No micro-batch fits with sufficient memory margin")
    micro = max(valid, key=lambda row: row["tokens_per_second"])["micro_batch"]
    for ranks in (2, 4):
        workflow.run(
            f"benchmark-{ranks}-gpu",
            torchrun(
                ranks,
                "scripts/train/benchmark.py",
                "--model-config",
                "configs/model/p213m.yaml",
                "--micro-batches",
                micro,
                "--output",
                ROOT / f"runs/benchmark-{ranks}-gpu",
                entrypoint=False,
            ),
            max_hours=2,
        )
    eager = json.loads((ROOT / "runs/benchmark-4-gpu/summary.json").read_text())["results"][0]
    # 编译属于可选优化：失败明确记录，继续已验证的 eager 路线。
    compiled = False
    throughput = eager["tokens_per_second"]
    try:
        workflow.run(
            "verify-compile",
            [
                PYTHON,
                "scripts/train/verify_compile.py",
                "--model-config",
                "configs/model/p213m.yaml",
                "--output",
                str(ROOT / "runs/verify-compile"),
            ],
            gpu="0",
            max_hours=1,
        )
        workflow.run(
            "benchmark-compiled",
            torchrun(
                4,
                "scripts/train/benchmark.py",
                "--model-config",
                "configs/model/p213m.yaml",
                "--micro-batches",
                micro,
                "--compile",
                "--output",
                ROOT / "runs/benchmark-compiled",
                entrypoint=False,
            ),
            max_hours=2,
        )
        result = json.loads((ROOT / "runs/benchmark-compiled/summary.json").read_text())["results"][
            0
        ]
        compiled = result["tokens_per_second"] >= throughput * 1.1 and result["has_memory_margin"]
        if compiled:
            throughput = result["tokens_per_second"]
    except (RuntimeError, TimeoutError) as error:
        workflow.record(
            "compile-selection", "fallback", note=f"Eager retained: {type(error).__name__}"
        )
    proposal = {"micro_batch_size": micro, "compiled": compiled, "tokens_per_second": throughput}
    write_json(ROOT / "runs/performance-selection.json", proposal)
    return proposal


def select_pretraining_lr(workflow: Workflow, performance: dict) -> float:
    data = ROOT / "data/tokenized/pilot100m"
    tokens = json.loads((data / "summary.json").read_text())["splits"]["train"]["written_tokens"]
    rates = [3e-4, 6e-4, 1e-3, 6e-4]
    jobs = []
    for index, rate in enumerate(rates):
        name = f"m04_lr_pilot_{index}"
        output = ROOT / "runs" / name
        command = cli(
            "dummym-pretrain",
            "--model-config",
            "configs/model/p213m.yaml",
            "--data-dir",
            data,
            "--output-dir",
            output,
            "--total-tokens",
            tokens,
            "--global-batch-size",
            256,
            "--micro-batch-size",
            performance["micro_batch_size"],
            "--learning-rate",
            rate,
            "--seed",
            2027 if index == 3 else 2026,
        )
        if performance["compiled"]:
            command += ["--compile"]
        if (output / "checkpoint.pt").exists():
            command += ["--resume", str(output / "checkpoint.pt")]
        jobs.append(Job(name, command, str(index)))
    workflow.jobs(jobs, max_hours=6)
    losses = [
        json.loads((ROOT / "runs" / f"m04_lr_pilot_{index}" / "summary.json").read_text())[
            "progress"
        ]["validation_loss"]
        for index in range(3)
    ]
    selected = min(range(3), key=lambda index: losses[index])
    write_json(
        ROOT / "runs/lr-selection.json",
        {
            "learning_rates": rates[:3],
            "losses": losses,
            "selected": rates[selected],
            "fourth_run": "6e-4 seed 2027",
        },
    )
    return rates[selected]


def pretrain(workflow: Workflow, performance: dict, learning_rate: float):
    proposal_path = ROOT / "runs/pretraining-proposal.json"
    if proposal_path.exists():
        proposal = json.loads(proposal_path.read_text())
    else:
        capacities = json.loads((ROOT / "data/capacities.json").read_text())
        available = min(
            capacities[name]["train"]["tokens"] / weight for name, weight in WEIGHTS.items()
        )
        hours = min(96.0, workflow.remaining_seconds() / 3600 - 20)
        if hours <= 0:
            raise TimeoutError(
                "No remaining pretraining budget after reserving SFT/evaluation time"
            )
        tokens = (
            int(
                min(
                    20_000_000_000,
                    available,
                    performance["tokens_per_second"] * hours * 3600 * 0.85,
                )
                // 100_000_000
            )
            * 100_000_000
        )
        validation = int(
            min(
                10_000_000,
                *(
                    capacities[name][split]["tokens"] / weight
                    for name, weight in WEIGHTS.items()
                    for split in ("validation", "test")
                ),
            )
            * 0.95
        )
        if tokens < 200_000_000 or validation < 100_000:
            raise RuntimeError("Insufficient unique mixed data; expand downloads before training")
        proposal = {
            **performance,
            "tokens": tokens,
            "validation_tokens": validation,
            "max_hours": hours,
            "learning_rate": learning_rate,
            "unique_mixture_capacity": int(available),
            "data_repetition": False,
        }
        write_json(proposal_path, proposal)
    data = ROOT / "data/tokenized/english"
    workflow.run(
        "pack-full-data",
        cli(
            "dummym-pack",
            "--clean",
            ROOT / "data/clean/english",
            "--tokenizer",
            ROOT / "data/tokenizer/english32k/tokenizer.json",
            "--train-tokens",
            proposal["tokens"],
            "--validation-tokens",
            proposal["validation_tokens"],
            "--output",
            data,
        ),
        max_hours=12,
    )
    tokens = json.loads((data / "summary.json").read_text())["splits"]["train"]["written_tokens"]
    output = ROOT / "runs/m04_base"
    command = torchrun(
        4,
        "dummym-pretrain",
        "--model-config",
        "configs/model/p213m.yaml",
        "--data-dir",
        data,
        "--output-dir",
        output,
        "--total-tokens",
        tokens,
        "--micro-batch-size",
        proposal["micro_batch_size"],
        "--learning-rate",
        proposal["learning_rate"],
        "--max-hours",
        proposal["max_hours"],
    )
    if proposal["compiled"]:
        command += ["--compile"]
    if (output / "checkpoint.pt").exists():
        command += ["--resume", str(output / "checkpoint.pt")]
    workflow.run("formal-pretraining", command, max_hours=proposal["max_hours"] + 2)
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "complete":
        workflow.record(
            "formal-pretraining",
            "paused",
            note="Time limit reached; checkpoint retained; SFT uses this partial base",
        )
    workflow.run(
        "export-base",
        cli(
            "dummym-export",
            "--checkpoint",
            output / "checkpoint.pt",
            "--tokenizer",
            ROOT / "data/tokenizer/english32k",
            "--output",
            ROOT / "exports/base",
        ),
        gpu="0",
        max_hours=1,
    )
    workflow.run(
        "base-completion-smoke",
        [
            PYTHON,
            "scripts/evaluation/complete.py",
            "--model",
            str(ROOT / "exports/base"),
            "--output",
            str(ROOT / "runs/base-completions.json"),
        ],
        gpu="0",
        max_hours=1,
    )
    workflow.run(
        "publish-base", cli("dummym-publish", "--folder", ROOT / "exports/base"), max_hours=2
    )


def posttrain(workflow: Workflow):
    data = ROOT / "data/raw/sft"
    workflow.run(
        "download-sft",
        cli(
            "dummym-download",
            "--repo",
            "HuggingFaceTB/smol-smoltalk",
            "--files",
            0,
            "--output",
            data,
        ),
        max_hours=2,
    )
    jobs = []
    for index, rate in enumerate((1e-5, 3e-5, 6e-5, 1e-4)):
        output = ROOT / "runs" / f"sft_pilot_{index}"
        command = cli(
            "dummym-sft",
            "--model",
            ROOT / "exports/base",
            "--data-dir",
            data,
            "--output",
            output,
            "--learning-rate",
            rate,
            "--epochs",
            1,
            "--max-conversations",
            50_000,
        )
        checkpoints = sorted(
            output.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
        )
        if checkpoints:
            command += ["--resume", str(checkpoints[-1])]
        jobs.append(
            Job(
                f"sft-pilot-{index}",
                command,
                str(index),
            )
        )
    workflow.jobs(jobs, max_hours=5)
    workflow.jobs(
        [
            Job(
                f"sft-dev-{index}",
                cli(
                    "dummym-evaluate",
                    "--model",
                    ROOT / "runs" / f"sft_pilot_{index}" / "final",
                    "--suite",
                    "evaluation/chat_dev.jsonl",
                    "--output",
                    ROOT / "runs/evaluation" / f"dev-{index}.jsonl",
                ),
                str(index),
            )
            for index in range(4)
        ],
        max_hours=2,
    )
    ranking = []
    for index in range(4):
        answers = [
            json.loads(line)
            for line in (ROOT / "runs/evaluation" / f"dev-{index}.jsonl").read_text().splitlines()
        ]
        flags = sum(bool(row["flags"]) for row in answers) / len(answers)
        loss = json.loads((ROOT / "runs" / f"sft_pilot_{index}" / "summary.json").read_text())[
            "validation_loss"
        ]
        ranking.append((flags, loss, index))
    selected = min(ranking)[2]
    rate = (1e-5, 3e-5, 6e-5, 1e-4)[selected]
    write_json(
        ROOT / "runs/sft-selection.json",
        {
            "ranking": ranking,
            "selected_lr": rate,
            "criterion": "Development structural failure rate then validation loss; not a semantic acceptance claim",
        },
    )
    output = ROOT / "runs/m09_sft"
    command = torchrun(
        4,
        "dummym-sft",
        "--model",
        ROOT / "exports/base",
        "--data-dir",
        data,
        "--output",
        output,
        "--learning-rate",
        rate,
    )
    checkpoints = sorted(
        output.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if checkpoints:
        command += ["--resume", str(checkpoints[-1])]
    workflow.run("formal-sft", command, max_hours=8)
    destination = ROOT / "exports/chat"
    if not destination.exists():
        shutil.copytree(output / "final", destination)
    model_card = destination / "README.md"
    model_card.write_text(
        "# SmallLLMPreTrain English Chat\n\n"
        "从随机权重预训练的约 213M Llama-like 模型，使用自训 32K BPE 和短对话 SFT。\n\n"
        "尚待人工语义验收；不能据此宣称可靠的通用助手能力。\n\n"
        "详细架构、来源、训练量、阶段结果和限制见仓库根 README 与 docs/results.md。\n"
    )
    workflow.run("publish-chat", cli("dummym-publish", "--folder", destination), max_hours=2)
    workflow.run(
        "final-chat-generation",
        cli(
            "dummym-evaluate",
            "--model",
            destination,
            "--suite",
            "evaluation/chat_test.jsonl",
            "--output",
            ROOT / "runs/evaluation/test.jsonl",
        ),
        gpu="0",
        max_hours=2,
    )


def benchmark_capabilities(workflow: Workflow):
    """Base/Chat 使用相同 completion 评测，独立环境避免改变训练依赖。"""
    uv = str(ROOT / "tools/uv")
    python = str(ROOT / "envs/evaluation/bin/python")
    workflow.run(
        "evaluation-environment",
        [uv, "venv", "--python", "3.11", str(ROOT / "envs/evaluation")],
        max_hours=1,
    )
    workflow.run(
        "evaluation-torch",
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "torch==2.7.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ],
        max_hours=2,
    )
    workflow.run(
        "evaluation-dependencies",
        [uv, "pip", "install", "--python", python, "-e", str(ROOT / "repo") + "[workflow,eval]"],
        max_hours=2,
    )
    for model in ("base", "chat"):
        workflow.run(
            f"lm-eval-{model}",
            [
                python,
                "-m",
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={ROOT}/exports/{model},dtype=bfloat16",
                "--tasks",
                "hellaswag,piqa,arc_easy",
                "--batch_size",
                "16",
                "--limit",
                "200",
                "--log_samples",
                "--output_path",
                str(ROOT / "runs/lm_eval" / model),
            ],
            gpu="0",
            max_hours=2,
        )


def install_inference(workflow: Workflow):
    uv = str(ROOT / "tools/uv")
    workflow.run(
        "inference-environment",
        [uv, "venv", "--python", "3.11", str(ROOT / "envs/inference")],
        max_hours=1,
    )
    workflow.run(
        "install-vllm",
        [uv, "pip", "install", "--python", str(ROOT / "envs/inference/bin/python"), "vllm==0.11.0"],
        max_hours=2,
    )
    workflow.run(
        "vllm-offline-smoke",
        [
            str(ROOT / "envs/inference/bin/python"),
            "scripts/inference/check_vllm.py",
            "--model",
            str(ROOT / "exports/chat"),
            "--output",
            str(ROOT / "runs/vllm-smoke.json"),
        ],
        gpu="0",
        max_hours=1,
    )


def data_worker_alive(pid: int | None) -> bool:
    """重启控制器时接管本工程的数据任务，避免重复下载和并发写清洗目录。"""
    if not pid:
        return False
    process = Path(f"/proc/{pid}")
    try:
        return (
            b"scripts/data/build_full.py" in (process / "cmdline").read_bytes()
            and (process / "cwd").resolve() == ROOT / "repo"
        )
    except (FileNotFoundError, PermissionError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-reproduction-pid", type=int)
    args = parser.parse_args()
    lock = (ROOT / "workflow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workflow = Workflow(ROOT)
    worker = None
    worker_log = None
    adopted_worker_pid = None
    try:
        workflow.run("engineering-checks", [PYTHON, "-m", "pytest", "tests", "-q"], max_hours=1)
        workflow.run(
            "lint-checks",
            [str(BIN / "ruff"), "check", "src/dummym", "scripts", "tests"],
            max_hours=1,
        )
        workflow.run(
            "format-checks",
            [str(BIN / "ruff"), "format", "--check", "src/dummym", "scripts", "tests"],
            max_hours=1,
        )
        prepare_pilot(workflow)
        if not (ROOT / "data/capacities.json").exists():
            previous_pid = workflow.state["stages"].get("full-data-background", {}).get("pid")
            if data_worker_alive(previous_pid):
                adopted_worker_pid = previous_pid
            else:
                worker_log = (ROOT / "logs/full-data.log").open("a")
                worker = subprocess.Popen(
                    [PYTHON, "scripts/data/build_full.py"],
                    cwd=ROOT / "repo",
                    stdout=worker_log,
                    stderr=worker_log,
                    start_new_session=True,
                )
                workflow.record(
                    "full-data-background",
                    "running",
                    pid=worker.pid,
                    log=str(ROOT / "logs/full-data.log"),
                )
        wait_existing_reproduction(workflow, args.existing_reproduction_pid)
        performance = verify_and_benchmark(workflow)
        learning_rate = select_pretraining_lr(workflow, performance)
        if worker:
            while worker.poll() is None:
                if workflow.remaining_seconds() < 20 * 3600:
                    raise TimeoutError("Full data preparation exhausted available pretraining time")
                time.sleep(10)
            if worker.returncode:
                raise RuntimeError("Full data preparation failed; inspect logs/full-data.log")
        while adopted_worker_pid and data_worker_alive(adopted_worker_pid):
            if workflow.remaining_seconds() < 20 * 3600:
                raise TimeoutError("Full data preparation exhausted available pretraining time")
            time.sleep(10)
        if not (ROOT / "data/capacities.json").exists():
            raise RuntimeError("Full data preparation ended without a complete capacity report")
        workflow.record(
            "full-data-background",
            "complete",
            note="Unique source capacities measured; no repeated-data budget",
        )
        pretrain(workflow, performance, learning_rate)
        posttrain(workflow)
        benchmark_capabilities(workflow)
        install_inference(workflow)
        workflow.state["status"] = "awaiting_semantic_review"
        write_json(workflow.path, workflow.state)
        workflow.record(
            "first-week-delivery",
            "awaiting_semantic_review",
            note="Model/export/evaluation produced. Review 120 answers before declaring capability acceptance.",
        )
    except Exception as error:
        workflow.state["status"] = "failed"
        workflow.record("pipeline", "failed", note=f"{type(error).__name__}: {error}")
        raise
    finally:
        if worker and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=30)
        if adopted_worker_pid and data_worker_alive(adopted_worker_pid):
            os.killpg(adopted_worker_pid, signal.SIGTERM)
        if worker_log:
            worker_log.close()
        lock.close()


if __name__ == "__main__":
    main()
