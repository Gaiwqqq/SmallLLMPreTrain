"""只评分合成课程可精确验证的最终答案，不代表一般聊天验收。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def score(predictions, suite):
    reference = {row["id"]: row for row in suite}
    if len(reference) != len(suite) or {row["id"] for row in predictions} != set(reference):
        raise ValueError("Prediction IDs must match the frozen diagnostic suite")
    if len(predictions) != len(suite):
        raise ValueError("Duplicate prediction ID")
    totals, passes = Counter(), Counter()
    for row in predictions:
        expected = reference[row["id"]]["expected"][-1]
        answer = row["answers"][-1].strip() if row["answers"] else ""
        passed = answer == expected and len(row["answers"]) == len(reference[row["id"]]["turns"])
        if expected.startswith("{") and len(row["answers"]) == len(reference[row["id"]]["turns"]):
            try:
                passed = json.loads(answer) == json.loads(expected)
            except json.JSONDecodeError:
                passed = False
        category = reference[row["id"]]["category"]
        totals[category] += 1
        passes[category] += int(passed and not row["flags"])
    return {
        "cases": len(suite),
        "passed": sum(passes.values()),
        "pass_rate": sum(passes.values()) / len(suite),
        "category_pass_rates": {key: passes[key] / value for key, value in totals.items()},
        "scope": "Exact final-answer synthetic diagnostic only; general semantic acceptance pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    result = score(read(args.predictions), read(args.suite))
    result["suite_sha256"] = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
