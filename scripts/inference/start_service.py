#!/usr/bin/env python3
"""启动本工程的回环 vLLM 服务，检查健康和真实 Chat API；不接管其他服务。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "dummym-english"


def request_json(path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    # 回环服务不经过下载代理。
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=60
    ) as response:
        return json.load(response)


def owned_process(state: dict, model: Path) -> bool:
    if state.get("model") != str(model.resolve()) or not isinstance(state.get("pid"), int):
        return False
    try:
        command = Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b"\0")
        return b"serve" in command and str(model.resolve()).encode() in command
    except OSError:
        return False


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root, model = args.root.resolve(), args.model.resolve()
    if not (model / "model.safetensors").exists():
        raise FileNotFoundError("Chat model export is missing")
    state_path = root / "runs/vllm-service.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with (root / "vllm-service.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        process = None
        state = previous
        if not owned_process(previous, model):
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                    raise RuntimeError("Port 8000 is occupied by an unowned service")
            except ConnectionRefusedError:
                pass
            log_path = root / "logs/vllm-service.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    ["bash", str(root / "repo/scripts/inference/serve.sh"), str(model)],
                    cwd=root / "repo",
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            state = {
                "status": "starting",
                "pid": process.pid,
                "model": str(model),
                "url": BASE_URL,
                "log": str(log_path),
                "started_at": time.time(),
            }
            write_state(state_path, state)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup; inspect {state['log']}")
                try:
                    response = request_json("/v1/models")
                    if MODEL_NAME not in {row["id"] for row in response["data"]}:
                        raise RuntimeError("Unexpected model served on port 8000")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vLLM readiness timeout") from None
                    time.sleep(2)
            answer = request_json(
                "/v1/chat/completions",
                {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello. Introduce yourself briefly."}],
                    "temperature": 0,
                    "max_tokens": 64,
                    "stop_token_ids": [2, 5],
                },
            )["choices"][0]
            if not answer["message"]["content"].strip():
                raise RuntimeError("Chat API returned an empty answer")
            state.update(
                status="passed",
                answer=answer["message"]["content"],
                finish_reason=answer["finish_reason"],
                scope="API availability only; semantic acceptance remains separate",
            )
            write_state(state_path, state)
            print("vLLM service health and Chat API smoke passed", flush=True)
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            state.update(status="failed")
            write_state(state_path, state)
            raise


if __name__ == "__main__":
    main()
