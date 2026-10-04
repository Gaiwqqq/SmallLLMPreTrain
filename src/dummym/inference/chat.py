"""终端多轮聊天：与 SFT 使用完全相同的 chat template。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        )
        .to(args.device)
        .eval()
    )
    messages = []
    print("English chat. /reset clears history; /quit exits.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/quit":
            break
        if prompt == "/reset":
            messages.clear()
            continue
        if not prompt:
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            candidate, add_generation_prompt=True, return_tensors="pt"
        ).to(args.device)
        if encoded.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
            print("Context limit reached. Use /reset or request fewer new tokens.")
            continue
        with torch.inference_mode():
            result = model.generate(
                encoded,
                attention_mask=torch.ones_like(encoded),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                **({"temperature": args.temperature, "top_p": 0.9} if args.temperature > 0 else {}),
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(result[0, encoded.shape[1] :], skip_special_tokens=True).strip()
        messages = candidate + [{"role": "assistant", "content": answer}]
        print(f"Assistant: {answer}")
