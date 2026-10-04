"""只在完整更新边界保存；每个 rank 的 RNG 都可恢复。"""

from pathlib import Path

import torch
import torch.distributed as dist

from dummym.training.distributed import ProcessContext


def collect_rng(context: ProcessContext) -> list[dict]:
    local = {"cpu": torch.get_rng_state(), "cuda": None}
    if context.device.type == "cuda":
        local["cuda"] = torch.cuda.get_rng_state(context.device).cpu()
    states = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    return states


def restore_rng(states: list[dict], context: ProcessContext) -> None:
    torch.set_rng_state(states[context.rank]["cpu"])
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(states[context.rank]["cuda"], context.device)


def save_checkpoint(
    path: Path, model, optimizer, scheduler, progress: dict, contract: dict, context: ProcessContext
) -> None:
    states = collect_rng(context)
    if context.primary:
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format_version": 2,
                "model_config": model.config.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "progress": progress.copy(),
                "contract": contract,
                "rng_states": states,
            },
            temporary,
        )
        temporary.replace(path)
    context.barrier()
