"""从训练文档自训 byte-level BPE，并保存可供 Transformers 使用的封装。"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from dummym.utils.files import sha256_file, write_json


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}{% generation %}"
    "{{ message['content'] + '<|im_end|>\\n' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>\\n' }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
).replace("\\n", "\n")


def train_tokenizer(files: list[Path], output: Path, max_documents: int = 100_000) -> None:
    output.mkdir(parents=True, exist_ok=False)
    counts = {"documents": 0}

    def documents():
        for path in files:
            source_documents = 0
            with path.open() as stream:
                for line in stream:
                    if counts["documents"] >= max_documents:
                        return
                    if source_documents >= max(1, max_documents // len(files)):
                        break
                    text = json.loads(line)["text"]
                    counts["documents"] += 1
                    source_documents += 1
                    yield text

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        documents(),
        trainers.BpeTrainer(
            vocab_size=32_000,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2,
        ),
    )
    if tokenizer.get_vocab_size() != 32_000:
        raise ValueError("Insufficient training text to build the required 32K vocabulary")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
        model_max_length=2048,
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(output)
    write_json(
        output / "training.json",
        {
            **counts,
            "vocab_size": len(fast),
            "training_files": [str(p) for p in files],
            "sha256": sha256_file(output / "tokenizer.json"),
            "special_tokens": SPECIAL_TOKENS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=100_000)
    args = parser.parse_args()
    if args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    train_tokenizer(args.input, args.output, args.max_documents)
