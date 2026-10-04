from collections import Counter
import json
import hashlib
from pathlib import Path

import pytest

from dummym.evaluation.chat import structural_flags, summarize_reviews


def test_frozen_suite_sizes_and_categories():
    root = Path(__file__).resolve().parents[3] / "evaluation"
    dev = [json.loads(line) for line in (root / "chat_dev.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (root / "chat_test.jsonl").read_text().splitlines()]
    assert len(dev) == 40 and len(test) == 120
    assert len({row["id"] for row in dev + test}) == 160
    assert set(Counter(row["category"] for row in test).values()) == {20}
    assert all(row["rubric"] and row["turns"] for row in dev + test)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, value in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == value["sha256"]
    assert len({tuple(row["turns"]) for row in dev}) == len(dev)


def test_structure_checks_do_not_claim_semantic_correctness():
    assert structural_flags(" ") == ["empty"]
    assert "repetition" in structural_flags("one two three four " * 20)
    assert structural_flags("The capital of France is London.") == []


def test_review_summary_requires_complete_evidence(tmp_path):
    predictions, reviews, output = (
        tmp_path / name for name in ("predictions.jsonl", "reviews.jsonl", "summary.json")
    )
    predictions.write_text(json.dumps({"id": "a", "category": "knowledge", "flags": []}) + "\n")
    reviews.write_text(json.dumps({"id": "a", "pass": True}) + "\n")
    with pytest.raises(ValueError, match="reason"):
        summarize_reviews(predictions, reviews, output)
    reviews.write_text(
        json.dumps({"id": "a", "pass": True, "reason": "Answer matches the stated fact"}) + "\n"
    )
    assert summarize_reviews(predictions, reviews, output)["accepted"]
