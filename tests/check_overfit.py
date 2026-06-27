"""토이 overfit 검증: 학습 문장을 번역해 정답과 정확히 일치하는지 측정.

  python -m tests.check_overfit

overfit이 성공했다면 학습 데이터에 대한 exact-match 정확도가 100%에 가까워야 한다.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.data import read_pairs
from src.tokenizer import load_tokenizer
from src.translate import greedy_translate, load_model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path("artifacts")
    tokenizer = load_tokenizer(art / "tokenizer.json")
    model, direction = load_model(art / "model.pt", device)

    pairs = read_pairs("data/toy_parallel.tsv")
    correct = 0
    shown = 0
    for ko, en in pairs:
        src_text, gold = (ko, en) if direction == "ko2en" else (en, ko)
        out = greedy_translate(model, tokenizer, src_text, device).strip()
        ok = out == gold.strip()
        correct += ok
        if shown < 8:
            mark = "OK " if ok else "XX "
            print(f"{mark}{src_text!r} -> {out!r}  (정답: {gold!r})")
            shown += 1

    acc = correct / len(pairs) * 100
    print(f"\nexact-match: {correct}/{len(pairs)} = {acc:.1f}%  (direction={direction}, device={device})")


if __name__ == "__main__":
    main()
