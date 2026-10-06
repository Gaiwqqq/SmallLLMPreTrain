"""精确答案与冻结划分的回归检查；不把这些检查当成模型验收。"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_data import tokenizer as original_tokenizer


@pytest.fixture(name="tokenizer")
def shared_tokenizer():
    return original_tokenizer.__wrapped__()


REPO = Path(__file__).resolve().parents[3]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/posttrain/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_curriculum_answers_and_split_isolation():
    generator = module("build_curriculum")
    scorer = module("score_curriculum")
    all_messages = []
    for split in generator.TEMPLATES:
        rows = list(generator.examples(split, 100))
        hashes = {json.dumps(row["messages"], sort_keys=True) for row in rows}
        assert len(hashes) == len(rows)
        all_messages.append(hashes)
        predictions = [{**row, "answers": row["expected"], "flags": []} for row in rows]
        assert scorer.score(predictions, rows)["pass_rate"] == 1
        for row in rows:
            if row["category"] == "context":
                initial = int(row["messages"][1]["content"].split()[-2])
                remaining = int(row["expected"][1])
                assert 1 <= initial - remaining <= 8
        predictions[0]["answers"] = ["wrong"]
        assert scorer.score(predictions, rows)["passed"] == 99
    assert not (
        all_messages[0] & all_messages[1]
        or all_messages[0] & all_messages[2]
        or all_messages[1] & all_messages[2]
    )


def test_score_rejects_missing_prediction():
    rows = list(module("build_curriculum").examples("test", 4))
    with pytest.raises(ValueError):
        module("score_curriculum").score([], rows)


def test_explicit_split_preparation(tokenizer, tmp_path):
    from dummym.posttrain.data import data_contract, load_prepared, prepare_data

    data, model, output = tmp_path / "raw", tmp_path / "model", tmp_path / "prepared"
    data.mkdir()
    model.mkdir()
    tokenizer.save_pretrained(model)
    files = []
    for split in ("train", "dev"):
        path = data / f"{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in module("build_curriculum").examples(split, 12))
            + "\n"
        )
        files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "status": "complete",
        "revision": "test",
        "files": files,
        "split_files": {"train": "train.jsonl", "test": "dev.jsonl"},
    }
    (data / "download.json").write_text(json.dumps(manifest))
    prepare_data(data, model, output, tokenizer, workers=1)
    split = load_prepared(output, data_contract(data, model, tokenizer))
    assert len(split["train"]) == len(split["test"]) == 12
    # 构造内容相同、哈希正确的两个 split，必须拒绝泄漏。
    (data / "dev.jsonl").write_bytes((data / "train.jsonl").read_bytes())
    manifest["files"][1]["sha256"] = manifest["files"][0]["sha256"]
    (data / "download.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        prepare_data(data, model, tmp_path / "leaked", tokenizer, workers=1)
