"""v1 的独立补充指标：逐个任务轮次评分，保留历史末轮分数不变。"""

import argparse
import json
from pathlib import Path


def score_trajectory(rows):
    groups = {}
    for row in rows:
        expected, answers = row["expected"], row["answers"]
        # 上下文首轮只是确认事实，后续任务轮次均必须正确。
        start = 1 if row["category"] == "context" else 0
        correct = len(expected) == len(answers) and not row["flags"]
        for index in range(start, len(expected)):
            actual = answers[index].strip() if index < len(answers) else ""
            if expected[index].startswith("{"):
                try:
                    matches = json.loads(actual) == json.loads(expected[index])
                except json.JSONDecodeError:
                    matches = False
            else:
                matches = actual == expected[index]
            correct = correct and matches
        counts = groups.setdefault(row["category"], {"passed": 0, "cases": 0})
        counts["cases"] += 1
        counts["passed"] += int(correct)
    return {
        category: {**counts, "pass_rate": counts["passed"] / counts["cases"]}
        for category, counts in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    result = score_trajectory(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
