#!/usr/bin/env python
import time
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

print("Testing BPE training...")
start = time.time()
vocab, merges = run_train_bpe(
    input_path=FIXTURES_PATH / "corpus.en",
    vocab_size=260,  # 小一点的vocab_size
    special_tokens=["<|endoftext|>"],
)
end = time.time()

print(f"Time taken: {end - start:.3f}s")
print(f"Vocab size: {len(vocab)}")
print(f"Merges count: {len(merges)}")
print(f"\nFirst 10 merges:")
for i, m in enumerate(merges[:10]):
    print(f"  {i}: {m}")

# 对比参考实现
print("\n\nComparing with reference...")
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
with open(FIXTURES_PATH / "train-bpe-reference-merges.txt", encoding="utf-8") as f:
    gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    reference_merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_reference_merges[:10]
    ]

print("Reference first 10 merges:")
for i, m in enumerate(reference_merges[:10]):
    print(f"  {i}: {m}")

print("\n\nDifferences:")
for i, (actual, expected) in enumerate(zip(merges[:10], reference_merges[:10])):
    if actual != expected:
        print(f"  Position {i}: got {actual}, expected {expected}")
