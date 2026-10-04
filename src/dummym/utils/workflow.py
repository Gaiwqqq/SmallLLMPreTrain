"""长任务的日志、预算和阶段状态；不隐藏具体训练命令。"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from dummym.utils.files import write_json


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    gpu: str | None = None


class Workflow:
    def __init__(self, root: Path, hours: float = 168):
        self.root = root
        self.path = root / "workflow.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "started_at": time.time(),
                "budget_hours": hours,
                "stages": {},
                "status": "running",
            }
        )
        self.deadline = self.state["started_at"] + self.state["budget_hours"] * 3600
        self.state["status"] = "running"
        write_json(self.path, self.state)

    def record(self, name: str, status: str, **details):
        self.state["stages"][name] = {"status": status, "updated_at": time.time(), **details}
        write_json(self.path, self.state)
        report = self.root / "repo/docs/results.md"
        original = report.read_text().split("\n## 自动阶段进度")[0]
        lines = ["\n## 自动阶段进度\n", "| 阶段 | 状态 | 说明 |", "|---|---|---|"]
        for stage, value in self.state["stages"].items():
            detail = value.get("note", value.get("log", ""))
            lines.append(f"| {stage} | {value['status']} | {detail} |")
        temporary = report.with_suffix(".md.tmp")
        temporary.write_text(original + "\n".join(lines) + "\n")
        temporary.replace(report)
        if status == "complete" and (self.root / "repo/.git").exists():
            # 只提交本阶段报告，绝不把环境、认证或正在开发的源码纳入自动提交。
            subprocess.run(["git", "add", "docs/results.md"], cwd=self.root / "repo", check=True)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root / "repo")
            if changed.returncode == 1:
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=Codex",
                        "-c",
                        "user.email=codex@local",
                        "commit",
                        "-m",
                        f"Record verified stage: {name}",
                        "--",
                        "docs/results.md",
                    ],
                    cwd=self.root / "repo",
                    check=True,
                )

    def complete(self, name: str) -> bool:
        return self.state["stages"].get(name, {}).get("status") == "complete"

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.time())

    def jobs(self, jobs: list[Job], max_hours: float | None = None):
        pending = [job for job in jobs if not self.complete(job.name)]
        active = []
        stop_at = min(self.deadline, time.time() + max_hours * 3600) if max_hours else self.deadline
        try:
            for job in pending:
                log_path = self.root / "logs" / f"{job.name}.log"
                log = log_path.open("a")
                environment = os.environ.copy()
                if job.gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = job.gpu
                process = subprocess.Popen(
                    job.command,
                    cwd=self.root / "repo",
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                active.append((job, process, log))
                self.record(
                    job.name, "running", pid=process.pid, log=str(log_path), command=job.command
                )
            while active:
                if time.time() >= stop_at:
                    raise TimeoutError(
                        "Stage reached its time budget; latest complete checkpoint is retained"
                    )
                for job, process, log in list(active):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    active.remove((job, process, log))
                    if code:
                        self.record(
                            job.name,
                            "failed",
                            exit_code=code,
                            log=str(self.root / "logs" / f"{job.name}.log"),
                        )
                        raise RuntimeError(f"{job.name} failed with exit code {code}")
                    self.record(
                        job.name, "complete", log=str(self.root / "logs" / f"{job.name}.log")
                    )
                time.sleep(5)
        finally:
            for job, process, log in active:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    self.record(
                        job.name, "interrupted", log=str(self.root / "logs" / f"{job.name}.log")
                    )
                log.close()

    def run(
        self, name: str, command: list[str], gpu: str | None = None, max_hours: float | None = None
    ):
        self.jobs([Job(name, command, gpu)], max_hours=max_hours)
