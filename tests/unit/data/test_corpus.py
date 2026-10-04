from dummym.data.corpus import document_split, normalize
from dummym.tokenizer.bpe import learn_merges


def test_document_split_is_stable_and_has_all_partitions():
    assert document_split("abc") == document_split("abc")
    splits = {document_split(str(index)) for index in range(10_000)}
    assert splits == {"train", "validation", "test"}


def test_filter_rejects_repetitive_boilerplate_and_preserves_paragraphs():
    assert normalize("menu " * 200) is None
    text = "\n\n\n".join(" ".join(f"word{i + start}" for i in range(80)) for start in (0, 100))
    cleaned = normalize(text)
    assert "\n\n\n" not in cleaned
    assert "\n\n" in cleaned


def test_teaching_bpe_is_deterministic_and_merges_frequent_pair():
    words = ["low", "low", "lower", "new"]
    rules = learn_merges(words, 3)
    assert rules == learn_merges(words, 3)
    assert rules[0] == ("l", "o")
