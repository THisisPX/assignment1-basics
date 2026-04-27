import json
from collections.abc import Iterable, Iterator
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


# GPT-2 pretokenization pattern
PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class BPE_Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize tokenizer with `vocab` (id->bytes), `merges` (ordered list of pairs of bytes), and optional special tokens."""
        self.vocab = dict(vocab)
        # reverse mapping: bytes -> id
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges = list(merges)
        # map pair -> rank (lower is earlier/higher priority)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.special_tokens = list(special_tokens or [])
        # Pre-compile special token pattern, sorted by length descending for greedy matching
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
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

    def _encode_single_piece(self, text: str) -> list[int]:
        """Encode a single text piece (without special tokens) into token ids."""
        out_ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            word_bytes = match.group(0).encode("utf-8")
            tokens = [bytes([b]) for b in word_bytes]
            merged = self._apply_bpe_to_tokens(tokens)
            for tk in merged:
                tid = self.token_to_id.get(tk)
                if tid is None:
                    # fallback: break into single bytes
                    for b in tk:
                        out_ids.append(self.token_to_id[bytes([b])])
                else:
                    out_ids.append(tid)
        return out_ids

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids using byte-level BPE and provided merges/vocab.

        Special tokens in `self.special_tokens` are preserved as single tokens when present.
        """
        if self._special_pattern is None:
            return self._encode_single_piece(text)

        parts = self._special_pattern.split(text)
        out_ids: list[int] = []

        for idx, part in enumerate(parts):
            if idx % 2 == 1 and part in self.special_tokens:
                # This is a special token
                b = part.encode("utf-8")
                tid = self.token_to_id.get(b)
                if tid is not None:
                    out_ids.append(tid)
            else:
                out_ids.extend(self._encode_single_piece(part))

        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of text chunks (e.g., lines from a file) into token ids lazily.

        This yields token ids one at a time without materializing the full input or output.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

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
