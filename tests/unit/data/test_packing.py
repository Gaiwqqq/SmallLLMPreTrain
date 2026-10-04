"""验证来源配额、split 隔离，以及语料不足不会留下成功标记。"""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from dummym.data.packing import prepare_mixture


def test_mixture_uses_exact_source_quotas_and_separate_splits(tmp_path):
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<s>": 1, "</s>": 2, "train": 3, "validation": 4, "test": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "clean.json").write_text(json.dumps({"status": "complete"}))
    for source in ("first", "second"):
        for split in ("train", "validation", "test"):
            (clean / f"{source}.{split}.jsonl").write_text(
                json.dumps({"text": (split + " ") * 5000}) + "\n"
            )
    output = tmp_path / "packed"
    prepare_mixture(clean, path, output, 8192, 8192, {"first": 0.5, "second": 0.5})
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    for split, token in (("train", 3), ("validation", 4), ("test", 5)):
        assert summary["splits"][split]["written_tokens"] == 8192
        assert summary["sources"][f"first.{split}"]["written_tokens"] == 4096
        assert np.all(np.fromfile(output / f"{split}.bin", dtype="<u2") == token)
    (clean / "first.train.jsonl").write_text(json.dumps({"text": "train"}) + "\n")
    incomplete = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_mixture(clean, path, incomplete, 8192, 8192, {"first": 0.5, "second": 0.5})
    assert not (incomplete / "summary.json").exists()
