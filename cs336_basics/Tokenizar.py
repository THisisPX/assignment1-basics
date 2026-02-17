
# from collections import 
import json
from typing import Iterable, Iterator
import regex as re
from tests.common import gpt2_bytes_to_unicode


def _str_to_bytes_using_gpt2(token_str: str, byte_decoder: dict[str, int]) -> bytes:
    # Convert a string composed of GPT-2 printable characters back to bytes.
    # If a character is not in the GPT-2 decoder (e.g., in special tokens), fall back to ord()
    result = []
    for ch in token_str:
        if ch in byte_decoder:
            result.append(byte_decoder[ch])
        else:
            result.append(ord(ch))
    return bytes(result)


class BPE_Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """Initialize tokenizer with `vocab` (id->bytes), `merges` (ordered list of pairs of bytes), and optional special tokens."""
        self.vocab = dict(vocab)
        # reverse mapping: bytes -> id
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges = list(merges)
        # map pair -> rank (lower is earlier/higher priority)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = list(special_tokens or [])

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Construct tokenizer from a serialized GPT-2-style vocab JSON and merges text file.

        `vocab_filepath` should be a JSON mapping token_str -> id (like the fixture),
        `merges_filepath` should be a text file with one merge per line, tokens separated by space.
        """
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)

        # raw: token_str -> id
        vocab: dict[int, bytes] = {}
        for token_str, idx in raw.items():
            # special tokens like <|endoftext|> will not be in byte_decoder; fall back to utf-8 bytes
            try:
                token_bytes = bytes([byte_decoder[c] for c in token_str])
            except Exception:
                token_bytes = token_str.encode("utf-8")
            vocab[int(idx)] = token_bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ")
                # convert each GPT-2 char token back to bytes
                left = _str_to_bytes_using_gpt2(parts[0], byte_decoder)
                right = _str_to_bytes_using_gpt2(parts[1], byte_decoder)
                merges.append((left, right))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_bpe_to_tokens(self, tokens: list[bytes]) -> list[bytes]:
        """Apply merges greedily to a sequence of byte-tokens using the learned merge ranks."""
        if not self.merges:
            return tokens

        # tokens is list of bytes objects (each may be multi-byte)
        while True:
            best_rank = None
            best_pair = None
            # scan adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair is None:
                break
            # merge all occurrences of best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids using byte-level BPE and provided merges/vocab.

        Special tokens in `self.special_tokens` are preserved as single tokens when present.
        """
        # protect special tokens by splitting
        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in self.special_tokens)
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        out_ids: list[int] = []

        for idx, part in enumerate(parts):
            # odd indices are special tokens when split
            if self.special_tokens and idx % 2 == 1:
                b = part.encode("utf-8")
                tid = self.token_to_id.get(b)
                if tid is None:
                    # if special token not in vocab, skip
                    continue
                out_ids.append(tid)
                continue

            # otherwise process normal text: split by whitespace but keep leading spaces
            words = part.split()
            # reconstruct tokens with leading spaces where appropriate
            cursor = 0
            for wi, w in enumerate(words):
                # determine if this word had a leading space in original part
                if wi > 0 or (cursor < len(part) and part[cursor] in " \t\n\r"):
                    word_bytes = b" " + w.encode("utf-8")
                else:
                    word_bytes = w.encode("utf-8")

                cursor = part.find(w, cursor) + len(w)

                # initial tokens: split into single-byte tokens
                tokens = [bytes([b]) for b in word_bytes]
                # apply merges
                merged = self._apply_bpe_to_tokens(tokens)
                # map to ids
                for tk in merged:
                    tid = self.token_to_id.get(tk)
                    if tid is None:
                        # fallback: break into single bytes
                        for b in tk:
                            out_ids.append(self.token_to_id[bytes([b])])
                    else:
                        out_ids.append(tid)

        return out_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token ids back to a string.

        This concatenates the token bytes and decodes as UTF-8 with replacement for errors.
        """
        parts: list[bytes] = []
        for i in ids:
            b = self.vocab.get(int(i))
            if b is None:
                continue
            parts.append(b)
        joined = b"".join(parts)
        try:
            return joined.decode("utf-8")
        except Exception:
            return joined.decode("utf-8", errors="replace")

        for i in range(256):
            self.vocab ={bytes([i]):i}
            self.vocab_v = {i:bytes([i])}

        

