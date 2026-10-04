"""训练 CLI：参数解析留在边界，算法见 engine.py。"""

import argparse
from pathlib import Path

import yaml

from dummym.models.llama_like import MiniLlamaConfig
from dummym.training.distributed import initialize
from dummym.training.engine import run
from dummym.training.recipe import TrainRecipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-hours", type=float, default=96)
    args = parser.parse_args()
    if (
        min(args.eval_every, args.save_every, args.max_hours) <= 0
        or min(args.eval_rows, args.stop_after_steps) < 0
    ):
        parser.error("Invalid run limits")
    config = MiniLlamaConfig.from_dict(
        yaml.safe_load(args.model_config.read_text())["model_config"]
    )
    recipe = TrainRecipe(
        total_tokens=args.total_tokens,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        precision=args.precision,
    )
    context = initialize(args.device)
    try:
        run(
            config,
            recipe,
            args.data_dir,
            args.output_dir,
            context,
            resume=args.resume,
            stop_after_steps=args.stop_after_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
            eval_rows=args.eval_rows,
            in_memory=args.in_memory,
            compile_model=args.compile,
            max_hours=args.max_hours,
        )
    finally:
        context.close()
