"""生成可核验的格式/状态课程；按模板与实体隔离 train/dev/test。"""

import argparse
import hashlib
import json
from pathlib import Path
import random

TEMPLATES = {
    "train": (
        "Return only {value} in uppercase.",
        "My name is {name}. I have {count} {item}.",
        "I gave away {removed}. How many {item} do I have now? Reply with only the number.",
    ),
    "dev": (
        "Write {value} in capital letters, with no other text.",
        "Call me {name}; I own {count} {item}.",
        "After giving someone {removed}, what is my remaining count of {item}? Output one number.",
    ),
    "test": (
        "Your entire response must be the uppercase form of {value}.",
        "Remember: I am {name}, and my {item} total is {count}.",
        "Subtract {removed} from my earlier total. Answer just the remaining number of {item}.",
    ),
}
ENTITIES = {
    "train": ("Mira", "Tobin", "Nora", "Evan"),
    "dev": ("Zuri", "Ivo", "Anika", "Rohan"),
    "test": ("Soren", "Eleni", "Bryn", "Talia"),
}


def examples(split, count, seed=20261006):
    rng = random.Random(seed + list(TEMPLATES).index(split))
    upper, intro, update = TEMPLATES[split]
    for index in range(count):
        # 唯一编号避免重复；各 split 名称与数值区间也分开。
        name = f"{ENTITIES[split][index % 4]}{index}"
        number = rng.randrange(10, 70) + 100 * list(TEMPLATES).index(split)
        removed = rng.randrange(1, min(number, 9))
        item = rng.choice(["coins", "stickers", "marbles", "cards"])
        value = f"label{list(TEMPLATES).index(split)}x{index}"
        mode = index % 4
        if mode == 0:
            turns, answers = [upper.format(value=value)], [value.upper()]
            category = "instruction"
        elif mode == 1:
            prefixes = {
                "train": "Return exactly",
                "dev": "Output only",
                "test": "Your whole reply must be",
            }
            question = f'{prefixes[split]} a JSON object with key "name" and value "{name}". No explanation.'
            turns, answers = [question], [json.dumps({"name": name})]
            category = "instruction"
        else:
            turns = [
                intro.format(name=name, count=number, item=item),
                update.format(removed=removed, item=item),
            ]
            answers = [f"You are {name} and have {number} {item}.", str(number - removed)]
            if mode == 3:
                turns.append(
                    {
                        "train": "What is my name? Reply with just my name.",
                        "dev": "Give only the name I told you.",
                        "test": "State my remembered name and nothing else.",
                    }[split]
                )
                answers.append(name)
            category = "context"
        messages = []
        for prompt, answer in zip(turns, answers):
            messages.extend(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            )
        yield {
            "id": f"curriculum-{split}-{index:06d}",
            "category": category,
            "turns": turns,
            "expected": answers,
            "messages": messages,
            "rubric": "Follow the exact requested output and preserve the user's updated state.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=32000)
    parser.add_argument("--eval-count", type=int, default=400)
    parser.add_argument(
        "--mix-existing", type=Path, help="已准备的原 SFT DatasetDict；只采样 train"
    )
    parser.add_argument("--mix-count", type=int, default=16000)
    args = parser.parse_args()
    if min(args.train_count, args.eval_count) <= 0:
        parser.error("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    mixed = []
    if args.mix_existing:
        from datasets import load_from_disk

        original = load_from_disk(str(args.mix_existing))["train"]
        if not 0 <= args.mix_count <= len(original):
            parser.error("Invalid original conversation sample size")
        selected = random.Random(20261006).sample(range(len(original)), args.mix_count)
        mixed = [
            {
                "id": f"replay-{i}",
                "category": "replay",
                "turns": [],
                "expected": [],
                "messages": original[i]["messages"],
                "rubric": "Original training conversation",
            }
            for i in selected
        ]
    for split in TEMPLATES:
        path = args.output / f"{split}.jsonl"
        with path.open("w") as stream:
            rows = list(examples(split, args.train_count if split == "train" else args.eval_count))
            if split == "train":
                rows.extend(mixed)
                random.Random(20261006).shuffle(rows)
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "status": "complete",
        "revision": "curriculum-v1-20261006",
        "files": files,
        "split_files": {"train": "train.jsonl", "test": "dev.jsonl"},
        "held_out_test": "test.jsonl",
        "note": "Synthetic diagnostic curriculum, not general chat or independent human acceptance",
    }
    manifest["replay_conversations"] = len(mixed)
    (args.output / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
