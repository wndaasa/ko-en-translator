"""병렬 코퍼스 로딩과 배치 구성.

teacher forcing 형식:
  src           = encode(소스) + <eos>
  tgt_full      = <bos> + encode(타깃) + <eos>
  decoder_input = tgt_full[:-1]
  labels        = tgt_full[1:]
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import BOS_ID, EOS_ID, PAD_ID, Tokenizer, encode, load_tokenizer, tag_id


def read_pairs(tsv_path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


class TranslationDataset(Dataset):
    """source_lang -> target_lang 방향의 번역 예시.

    direction="ko2en": TSV 1열(한국어)→2열(영어). "en2ko"는 반대.
    """

    def __init__(self, tsv_path: str | Path, tokenizer: Tokenizer, direction: str = "ko2en"):
        self.tok = tokenizer
        self.pairs = read_pairs(tsv_path)
        self.direction = direction

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ko, en = self.pairs[idx]
        src_text, tgt_text = (ko, en) if self.direction == "ko2en" else (en, ko)
        src = encode(self.tok, src_text, add_eos=True)
        tgt_full = encode(self.tok, tgt_text, add_bos=True, add_eos=True)
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "dec_in": torch.tensor(tgt_full[:-1], dtype=torch.long),
            "labels": torch.tensor(tgt_full[1:], dtype=torch.long),
        }


def make_example(
    tokenizer: Tokenizer,
    src_text: str,
    tgt_text: str,
    target_lang: str,
    max_len: int,
) -> dict[str, torch.Tensor]:
    """방향 태그가 붙은 한 개의 학습 예시 생성.

    src      = <2{target}> + encode(src) + <eos>
    tgt_full = <bos> + encode(tgt) + <eos>
    길이는 토큰 기준 max_len 으로 잘라 과도한 시퀀스를 방지한다.
    """
    src_ids = encode(tokenizer, src_text)[: max_len - 2]
    tgt_ids = encode(tokenizer, tgt_text)[: max_len - 2]
    src = [tag_id(target_lang)] + src_ids + [EOS_ID]
    tgt_full = [BOS_ID] + tgt_ids + [EOS_ID]
    return {
        "src": torch.tensor(src, dtype=torch.long),
        "dec_in": torch.tensor(tgt_full[:-1], dtype=torch.long),
        "labels": torch.tensor(tgt_full[1:], dtype=torch.long),
    }


class MTDataset(Dataset):
    """양방향 번역 데이터셋. 각 (ko, en) 쌍에서 지정한 방향들의 예시를 생성한다.

    bidirectional=True 이면 한 쌍이 ko→en, en→ko 두 예시가 된다(데이터 2배).
    """

    def __init__(
        self,
        tsv_path: str | Path,
        tokenizer: Tokenizer,
        max_len: int = 128,
        bidirectional: bool = True,
        limit: int = 0,
    ):
        self.tok = tokenizer
        self.pairs = read_pairs(tsv_path)
        if limit:
            self.pairs = self.pairs[:limit]
        self.max_len = max_len
        # 각 예시: (pair_idx, target_lang)
        self.index: list[tuple[int, str]] = []
        for i in range(len(self.pairs)):
            self.index.append((i, "en"))  # ko -> en
            if bidirectional:
                self.index.append((i, "ko"))  # en -> ko

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_idx, target_lang = self.index[idx]
        ko, en = self.pairs[pair_idx]
        src_text, tgt_text = (ko, en) if target_lang == "en" else (en, ko)
        return make_example(self.tok, src_text, tgt_text, target_lang, self.max_len)


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """배치 내 최대 길이에 맞춰 PAD_ID로 패딩."""
    def pad(key: str) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        maxlen = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    return {"src": pad("src"), "dec_in": pad("dec_in"), "labels": pad("labels")}


__all__ = ["TranslationDataset", "collate", "read_pairs", "load_tokenizer",
           "BOS_ID", "EOS_ID", "PAD_ID"]
