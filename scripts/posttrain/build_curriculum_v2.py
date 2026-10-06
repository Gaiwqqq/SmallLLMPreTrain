"""纠正 v1 固定前缀与狭窄任务偏差；六类短回复，80% 原对话回放。"""

import random

import build_curriculum as original


PROMPTS = {
    "train": (
        "Reply using exactly two sentences",
        "Answer with just the result",
        "Return exactly two bullet points",
        "Rewrite politely",
        "Give a one-sentence summary",
        "Remember these details",
    ),
    "dev": (
        "Give a friendly suggestion in two sentences",
        "Output the answer only",
        "Write two bullets and no extra text",
        "Make this request polite",
        "Summarize in a single sentence",
        "Keep this information for our conversation",
    ),
    "test": (
        "Suggest an activity in precisely two sentences",
        "Your entire answer should be the result",
        "Give only a pair of bullet points",
        "Turn this into a courteous request",
        "Compress this to one sentence",
        "Please retain the following information",
    ),
}


def examples(split, count, seed=20261006):
    rng = random.Random(seed + 100 + list(PROMPTS).index(split))
    for index in range(count):
        mode = index % 6
        category = ("daily", "knowledge", "instruction", "rewrite", "summary", "context")[mode]
        # 同一数值范围；模板、实体划分不同。避免 label0->label1 分布偏差。
        number, removed = rng.randrange(10, 301), rng.randrange(1, 10)
        name = f"{original.ENTITIES[split][index % 4]}{index}"
        color = rng.choice(["amber", "teal", "indigo", "silver", "coral"])
        item = rng.choice(["coins", "cards", "stickers", "marbles"])
        prefix = PROMPTS[split][mode]
        if mode == 0:
            activity = rng.choice(["walk", "picnic", "bike ride", "visit to a park"])
            turns = [f"{prefix} for a {activity} with a friend."]
            answers = [
                f"Plan a short {activity} with your friend. Choose a time that works for both of you."
            ]
        elif mode == 1:
            # 有限范围算术是可核验的基础任务，不等于外部世界知识训练。
            other = rng.randrange(1, 30)
            turns = [f"{prefix}: {number} plus {other}."]
            answers = [str(number + other)]
        elif mode == 2:
            if index % 18 == 2:
                value = "".join(
                    rng.choice("abcdefghijklmnopqrstuvxyz0123456789")
                    for _ in range(rng.randrange(3, 10))
                )
                turns = [f"Write {value} in uppercase. No other text."]
                answers = [value.upper()]
            elif index % 18 == 8:
                turns = [f'Reply only with JSON containing key "name" and value "{name}".']
                answers = ['{"name": "' + name + '"}']
            else:
                turns = [f"{prefix} about caring for a houseplant. Include water."]
                answers = [
                    "- Check the soil before adding water.\n- Place the plant where it receives suitable light."
                ]
        elif mode == 3:
            turns = [f"{prefix}: Bring the {color} notebook to room {number} today."]
            answers = [f"Could you please bring the {color} notebook to room {number} today?"]
        elif mode == 4:
            turns = [
                f"{prefix}: Route {number} closes at noon today for road repairs. It reopens tomorrow."
            ]
            answers = [
                f"Route {number} closes at noon today for road repairs and reopens tomorrow."
            ]
        else:
            turns = [
                f"{prefix}: my name is {name}, my favorite color is {color}, and I have {number} {item}.",
                f"I give away {removed} {item}. How many remain? Output only the number.",
                "Return my name and favorite color, separated by a comma, with no other text.",
            ]
            answers = [
                f"Your name is {name}, your favorite color is {color}, and you have {number} {item}.",
                str(number - removed),
                f"{name}, {color}",
            ]
        # 请求编号作为无关信息，确保有限模板的大规模实例不完全重复；各划分均匀生成。
        reference = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12))
        turns[0] = f"Request reference: {reference}. " + turns[0]
        messages = []
        for prompt, answer in zip(turns, answers):
            messages.extend(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            )
        yield {
            "id": f"curriculum-v2-{split}-{index:06d}",
            "category": category,
            "turns": turns,
            "expected": answers,
            "messages": messages,
            "rubric": "Follow the requested format; preserve all stated facts and updated state; no contradictory additions.",
        }


if __name__ == "__main__":
    original.examples = examples
    original.main()
