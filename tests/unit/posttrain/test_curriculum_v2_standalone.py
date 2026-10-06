"""无需训练依赖的课程唯一性、算术标签和逐轮评分检查。"""

from collections import Counter
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/posttrain"))
from build_curriculum_v2 import examples  # noqa: E402
from score_trajectory import score_trajectory  # noqa: E402


class CurriculumV2Tests(unittest.TestCase):
    def test_unique_balanced_and_disjoint(self):
        hashes = []
        for split in ("train", "dev", "test"):
            rows = list(examples(split, 600))
            fingerprints = {json.dumps(row["messages"], sort_keys=True) for row in rows}
            self.assertEqual(len(fingerprints), 600)
            self.assertEqual(set(Counter(row["category"] for row in rows).values()), {100})
            hashes.append(fingerprints)
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])

    def test_arithmetic_labels(self):
        import re

        for row in examples("train", 600):
            if row["category"] == "knowledge":
                left, right = re.search(r"(\d+) plus (\d+)", row["turns"][0]).groups()
                self.assertEqual(int(row["expected"][0]), int(left) + int(right))
            if row["category"] == "context":
                count = int(re.search(r"have (\d+)", row["turns"][0]).group(1))
                removed = int(re.search(r"away (\d+)", row["turns"][1]).group(1))
                self.assertEqual(int(row["expected"][1]), count - removed)

    def test_name_recall_does_not_hide_wrong_update(self):
        row = {
            "category": "context",
            "expected": ["ack", "10", "Nora"],
            "answers": ["ack", "11", "Nora"],
            "flags": [],
        }
        self.assertEqual(score_trajectory([row])["context"]["passed"], 0)
        row["answers"][1] = "10"
        self.assertEqual(score_trajectory([row])["context"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
