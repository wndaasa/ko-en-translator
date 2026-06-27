"""소스측 문자(음절) 토크나이저 — MCE(형태소 합성 인코더)용.

단어를 문자 시퀀스로 인코딩한다. 한글은 음절(가/나/다…) 단위, 영어는 알파벳, 숫자·기호는
각 문자가 하나의 단위가 된다(파이썬 문자열 반복이 곧 음절/문자 단위라 별도 분해 불필요).

방향 태그(<2en>/<2ko>)는 모델의 '단어' 레벨에서 학습형 벡터로 처리하므로 여기서는 다루지 않는다.

특수 토큰:
  <pad_char> 0  단어 길이 패딩
  <unk_char> 1  vocab에 없는 문자
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .tokenizer import read_tsv_texts

PAD_CHAR, UNK_CHAR = "<pad_char>", "<unk_char>"
PAD_CHAR_ID, UNK_CHAR_ID = 0, 1


def build_char_vocab(data_paths: list[str], out_path: str | Path, min_freq: int = 2) -> dict[str, int]:
    """병렬 코퍼스 양쪽 컬럼의 문자 빈도를 세어 vocab 구축, JSON 저장."""
    counter: Counter[str] = Counter()
    for p in data_paths:
        for text in read_tsv_texts(p):
            for word in text.split():
                counter.update(word)  # 문자열 반복 = 음절/문자 단위

    vocab: dict[str, int] = {PAD_CHAR: PAD_CHAR_ID, UNK_CHAR: UNK_CHAR_ID}
    for ch, freq in counter.most_common():
        if freq >= min_freq:
            vocab[ch] = len(vocab)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab


class CharTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_word(self, word: str) -> list[int]:
        return [self.vocab.get(ch, UNK_CHAR_ID) for ch in word]

    def encode_sentence(self, text: str) -> list[list[int]]:
        """문장 → 단어 목록, 각 단어는 문자 id 리스트."""
        return [self.encode_word(w) for w in text.split() if w]


def _main() -> None:
    ap = argparse.ArgumentParser(description="MCE용 문자(음절) vocab 구축")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="runs/mce/char_vocab.json")
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    vocab = build_char_vocab(args.data, args.out, args.min_freq)
    print(f"saved: {args.out}")
    print(f"char vocab size: {len(vocab)}")
    tok = CharTokenizer(vocab)
    sample = "안녕하세요 Hello"
    print(f"sample: {sample!r}")
    print(f"  encoded: {tok.encode_sentence(sample)}")


if __name__ == "__main__":
    _main()
