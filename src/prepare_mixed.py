"""보유 코퍼스로 도메인 혼합 데이터 생성 (다운로드 불필요).

  casual = OPUS-100 (일상·대화)        : data/processed/{train,val}.tsv
  formal = AI Hub 기술과학 (학술·격식)  : data/processed/aihub_tech/{train,val}.tsv

두 도메인을 균형 맞춰 섞고 도메인 라벨을 붙인다. 출력 형식: domain<TAB>ko<TAB>en
(도메인 태그 학습으로 '한 모델이 여러 레지스터를 망각 없이' 가설을 검증하기 위함.)

  python -m src.prepare_mixed --out-dir data/processed/mixed --per-domain-train 700000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import read_pairs

DOMAINS = [
    ("casual", "data/processed/train.tsv", "data/processed/val.tsv"),
    ("formal", "data/processed/aihub_tech/train.tsv", "data/processed/aihub_tech/val.tsv"),
]


def take(pairs, cap, seed):
    if cap and len(pairs) > cap:
        rng = random.Random(seed)
        idx = rng.sample(range(len(pairs)), cap)
        return [pairs[i] for i in idx]
    return pairs


def build(split: str, cap: int, seed: int) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for domain, train_p, val_p in DOMAINS:
        path = train_p if split == "train" else val_p
        pairs = read_pairs(path)
        pairs = take(pairs, cap, seed)
        rows.extend((domain, ko, en) for ko, en in pairs)
    random.Random(seed + 1).shuffle(rows)
    return rows


def write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for domain, ko, en in rows:
            f.write(f"{domain}\t{ko}\t{en}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="도메인 혼합 코퍼스 생성")
    ap.add_argument("--out-dir", default="data/processed/mixed")
    ap.add_argument("--per-domain-train", type=int, default=700000, help="도메인별 train 상한(균형)")
    ap.add_argument("--per-domain-val", type=int, default=1000, help="도메인별 val 상한")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    train = build("train", args.per_domain_train, args.seed)
    val = build("val", args.per_domain_val, args.seed)
    write_tsv(out / "train.tsv", train)
    write_tsv(out / "val.tsv", val)

    def counts(rows):
        c: dict[str, int] = {}
        for d, _, _ in rows:
            c[d] = c.get(d, 0) + 1
        return c

    print(f"train: {len(train)} {counts(train)}")
    print(f"val:   {len(val)} {counts(val)}")
    print(f"saved: {out}/train.tsv, {out}/val.tsv")


if __name__ == "__main__":
    main()
