"""不依赖 GPU 的推理恢复检查：依赖版本、端口占用、子进程失败清理。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/inference/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_script("check_environment")
service = load_script("start_service")


class RecoveryTests(unittest.TestCase):
    def test_version_drift_is_rejected_before_tokenizer_loading(self):
        with patch.object(preflight.metadata, "version", return_value="5.0.0"):
            with self.assertRaisesRegex(RuntimeError, "pinned recipe"):
                preflight.validate_versions({"transformers": "4.57.1"})

    def test_owner_requires_matching_model_and_live_command(self):
        model = Path("/diff/gaiwq/llm_pretrain/exports/chat")
        state = {"pid": 1234, "model": str(model)}
        with patch.object(Path, "read_bytes", return_value=b"vllm\0serve\0/other/model\0"):
            self.assertFalse(service.owned_process(state, model))
        with patch.object(Path, "read_bytes", return_value=f"vllm\0serve\0{model}\0".encode()):
            self.assertTrue(service.owned_process(state, model))
        self.assertFalse(service.owned_process({"pid": "1234", "model": str(model)}, model))

    def test_occupied_port_does_not_spawn_or_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "exports/chat"
            model.mkdir(parents=True)
            (model / "model.safetensors").touch()
            with (
                patch.object(sys, "argv", ["service", "--root", str(root), "--model", str(model)]),
                patch.object(service.socket, "create_connection", return_value=MagicMock()),
                patch.object(service.subprocess, "Popen") as spawn,
                patch.object(service.os, "killpg") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "unowned service"):
                    service.main()
                spawn.assert_not_called()
                kill.assert_not_called()

    def test_empty_api_answer_cleans_up_only_new_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "exports/chat"
            model.mkdir(parents=True)
            (model / "model.safetensors").touch()
            child = MagicMock(pid=1234)
            child.poll.return_value = None
            with (
                patch.object(sys, "argv", ["service", "--root", str(root), "--model", str(model)]),
                patch.object(
                    service.socket, "create_connection", side_effect=ConnectionRefusedError
                ),
                patch.object(service.subprocess, "Popen", return_value=child),
                patch.object(
                    service,
                    "request_json",
                    side_effect=[
                        {"data": [{"id": service.MODEL_NAME}]},
                        {"choices": [{"message": {"content": " "}, "finish_reason": "stop"}]},
                    ],
                ),
                patch.object(service.os, "killpg") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "empty answer"):
                    service.main()
                kill.assert_called_once_with(1234, service.signal.SIGTERM)
                self.assertEqual(
                    json.loads((root / "runs/vllm-service.json").read_text())["status"], "failed"
                )


if __name__ == "__main__":
    unittest.main()
