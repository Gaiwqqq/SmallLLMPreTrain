"""把旧教学入口的已完成产物转成独立的 ModelScope 上传任务。"""

import json
import math
from pathlib import Path
import shutil

from dummym.utils.files import write_json


def archive_legacy_run(run: Path, tokenizer: Path) -> Path:
    summary = json.loads((run / "summary.json").read_text())
    if "progress" in summary:
        if summary["status"] != "complete":
            raise ValueError("Only complete legacy training runs may be archived")
        progress = summary["progress"]
        contract = summary["contract"]
        resume_supported = True
    else:
        if summary["final_loss"] >= 0.1 or summary["final_next_token_accuracy"] <= 0.99:
            raise ValueError("Tiny overfit has not passed acceptance")
        progress = {
            "step": summary["completed_steps"],
            "validation_loss": None,
            "tokens_seen": summary["completed_steps"] * math.prod(summary["fixed_batch_shape"]),
            "overfit_loss": summary["final_loss"],
            "note": "Repeated fixed batch, no generalization evaluation",
        }
        contract = {"experiment": "tiny_overfit", "summary": summary}
        resume_supported = False
    destination = run / "milestones" / f"step-{progress['step']:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "ready.json").exists():
        return destination
    shutil.copyfile(run / "checkpoint.pt", destination / "checkpoint.pt")
    shutil.copyfile(tokenizer, destination / "tokenizer.json")
    write_json(
        destination / "ready.json",
        {"progress": progress, "contract": contract, "resume_supported": resume_supported},
    )
    return destination
