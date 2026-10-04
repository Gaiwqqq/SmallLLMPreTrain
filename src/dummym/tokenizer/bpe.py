"""可手算的 BPE：用于理解合并规则，正式语料使用 Rust Tokenizers。"""

from collections import Counter


def learn_merges(words: list[str], steps: int) -> list[tuple[str, str]]:
    vocabulary = Counter(tuple(word) + ("</w>",) for word in words)
    merges = []
    for _ in range(steps):
        pairs = Counter()
        for symbols, frequency in vocabulary.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += frequency
        if not pairs:
            break
        # 字典序打破平局，使相同输入得到相同规则。
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        merges.append(best)
        updated = Counter()
        for symbols, frequency in vocabulary.items():
            result, index = [], 0
            while index < len(symbols):
                if tuple(symbols[index : index + 2]) == best:
                    result.append("".join(best))
                    index += 2
                else:
                    result.append(symbols[index])
                    index += 1
            updated[tuple(result)] += frequency
        vocabulary = updated
    return merges
