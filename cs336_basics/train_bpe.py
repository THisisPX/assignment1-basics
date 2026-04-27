import os
import regex
from collections import Counter, defaultdict


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer from a corpus file.

    Args:
        input_path: Path to BPE tokenizer training data.
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: The trained tokenizer vocabulary, a mapping from int to bytes.
            merges: BPE merges, ordered by creation.
    """
    # GPT-2 pretokenization pattern
    pretoken_pattern = regex.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    # 1. Read file and pretokenize
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Split by special tokens to preserve them as atomic units
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(regex.escape(t) for t in sorted_tokens) + ")"
        sections = regex.split(special_pattern, text)
    else:
        sections = [text]

    tokenized_words: Counter[tuple[int, ...]] = Counter()
    for i, section in enumerate(sections):
        if special_tokens and i % 2 == 1 and section in special_tokens:
            continue
        for match in pretoken_pattern.finditer(section):
            tokenized_words[tuple(match.group(0).encode("utf-8"))] += 1

    # 2. Initialize vocabulary with 256 bytes + special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # 3. Initialize word states
    word_states = [list(w) for w in tokenized_words.keys()]
    word_counts = list(tokenized_words.values())

    # 4. Initial pair counts and mapping from pair to word indices
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, state in enumerate(word_states):
        for j in range(len(state) - 1):
            pair = (state[j], state[j + 1])
            pair_counts[pair] += word_counts[wi]
            pair_to_words[pair].add(wi)

    # 5. Perform BPE merges
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # Select the pair with highest frequency.
        # Tie-breaking: prefer lexicographically greatest byte pair.
        best_pair = max(
            pair_counts.items(),
            key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]),
        )[0]

        if pair_counts[best_pair] <= 0:
            break

        merged_token_id = len(vocab)
        merged_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[merged_token_id] = merged_token_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Update all words that contain best_pair
        affected_words = list(pair_to_words[best_pair])
        for wi in affected_words:
            state = word_states[wi]
            if len(state) < 2:
                continue

            # Verify this word still contains best_pair
            has_pair = False
            for j in range(len(state) - 1):
                if state[j] == best_pair[0] and state[j + 1] == best_pair[1]:
                    has_pair = True
                    break
            if not has_pair:
                continue

            # Build new state by merging all occurrences of best_pair
            new_state: list[int] = []
            j = 0
            while j < len(state):
                if (
                    j + 1 < len(state)
                    and state[j] == best_pair[0]
                    and state[j + 1] == best_pair[1]
                ):
                    new_state.append(merged_token_id)
                    j += 2
                else:
                    new_state.append(state[j])
                    j += 1

            # Remove old pairs from this word
            for j in range(len(state) - 1):
                pair = (state[j], state[j + 1])
                pair_counts[pair] -= word_counts[wi]
                pair_to_words[pair].discard(wi)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # Add new pairs from this word
            for j in range(len(new_state) - 1):
                pair = (new_state[j], new_state[j + 1])
                pair_counts[pair] += word_counts[wi]
                pair_to_words[pair].add(wi)

            word_states[wi] = new_state

    return vocab, merges
