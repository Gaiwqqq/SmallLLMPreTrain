"""显式映射手写模型权重到标准 Llama，导出前检查数值一致性。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.utils.files import write_json


def to_huggingface(reference: MiniLlamaForCausalLM) -> LlamaForCausalLM:
    config = LlamaConfig(
        **reference.config.to_dict(), attention_bias=False, mlp_bias=False, hidden_act="silu"
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config)
    mapped = {}
    for name, tensor in reference.state_dict().items():
        mapped[name if name.startswith("lm_head.") else "model." + name] = tensor
    model.load_state_dict(mapped, strict=True)
    model.tie_weights()
    return model


def export(checkpoint_path: Path, tokenizer_dir: Path, output: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference = MiniLlamaForCausalLM(MiniLlamaConfig.from_dict(checkpoint["model_config"]))
    reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference.eval()
    model = to_huggingface(reference).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if len(tokenizer) != reference.config.vocab_size:
        raise ValueError("Export tokenizer/model vocabulary mismatch")
    ids = torch.tensor([[1, 42, 53, 62, 2]])
    with torch.inference_mode():
        expected = reference(ids, labels=ids)
        actual = model(ids, labels=ids)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    report = {
        "max_logit_difference": (actual.logits - expected.logits).abs().max().item(),
        "loss_difference": abs(actual.loss.item() - expected.loss.item()),
        "source_checkpoint": str(checkpoint_path),
        "progress": checkpoint.get("progress", {}),
    }
    write_json(output / "export.json", report)
    (output / "README.md").write_text(
        "# SmallLLMPreTrain English Base\n\n"
        "本模型从随机权重预训练，属于文本续写模型；没有通过聊天能力验收。\n\n"
        f"- 实际参数：{reference.num_parameters():,}\n"
        f"- 上下文：{reference.config.max_position_embeddings}\n"
        f"- 已训练输入 token：{report['progress'].get('tokens_seen', 'unknown')}\n"
        f"- 验证 loss：{report['progress'].get('validation_loss', 'unknown')}\n"
        "- 语料：FineWeb-Edu、Cosmopedia-v2、TinyStories；来源与许可见学习仓库文档。\n"
        "- Tokenizer：自训 32K byte-level BPE。\n\n"
        "标准 Transformers 加载不需要 trust_remote_code；导出时对照 FP32 logits 与 loss。\n"
        "导出一致性见 export.json；知识错误、复读和指令不遵循仍需独立评测。\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args.checkpoint, args.tokenizer, args.output)
