"""
Lightweight Tokenizer for AURA
==============================
Supports both character-level and subword/word-level tokenization
with zero external dependencies.
"""

import json
import re
from typing import List, Dict, Union
import torch


class AURATokenizer:
    def __init__(self, mode: str = "char"):
        self.mode = mode
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train_from_text(self, text: str, max_vocab_size: int = 2000):
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(special_tokens)}

        if self.mode == "char":
            chars = sorted(set(text))
            for c in chars:
                if c not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[c] = idx
                    self.id_to_token[idx] = c
        else:
            # Word / Subword level based on whitespace & punctuation
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            from collections import Counter
            counts = Counter(tokens)
            most_common = [tok for tok, _ in counts.most_common(max_vocab_size - len(special_tokens))]
            for tok in most_common:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok

    def encode(self, text: str) -> torch.Tensor:
        unk_id = self.token_to_id[self.unk_token]
        if self.mode == "char":
            ids = [self.token_to_id.get(c, unk_id) for c in text]
        else:
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            ids = [self.token_to_id.get(tok, unk_id) for tok in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(ids, torch.Tensor):
            if ids.dim() > 1:
                ids = ids.squeeze(0)
            ids = ids.tolist()

        tokens = [self.id_to_token.get(i, "") for i in ids]
        if self.mode == "char":
            return "".join(tokens)
        else:
            # Space joining with punctuation attachment
            res = ""
            for tok in tokens:
                if tok in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                    continue
                if re.match(r"[^\w\s]", tok):
                    res += tok
                else:
                    res += (" " if res else "") + tok
            return res

    def save(self, filepath: str):
        data = {
            "mode": self.mode,
            "token_to_id": self.token_to_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AURATokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(mode=data["mode"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        return tok
