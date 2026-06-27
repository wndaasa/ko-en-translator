"""AI Hub '한국어-영어 번역 말뭉치(기술과학)' CSV → 정제 TSV 변환.

학술논문(출처: 한국학술정보, 문어체) 한-영 쌍. 사람 번역인 `en` 컬럼을 사용하고
기계번역 `mt` 컬럼은 버린다. OPUS 정제와 동일한 규칙 필터를 적용한다.

  python -m src.prepare_aihub \
    --root "/mnt/c/Users/dladu/Desktop/study/데이터셋/한국어-영어 번역 말뭉치(기술과학)" \
    --out-dir data/processed/aihub_tech
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .prepare_data import is_clean_pair, write_tsv

# 학술 문장은 길어서 OPUS보다 상한을 키운다(토큰 절단은 학습 단계에서 별도 처리).
MAX_CHARS = 400


def find_csv(split_dir: Path) -> Path:
    cands = sorted(split_dir.rglob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"CSV를 찾을 수 없음: {split_dir}")
    return cands[0]


def convert_csv(csv_path: Path, min_chars: int) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    seen_src: set[str] = set()
    out: list[tuple[str, str]] = []
    total = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            ko = (row.get("ko") or "").replace("\t", " ").replace("\n", " ").strip()
            en = (row.get("en") or "").replace("\t", " ").replace("\n", " ").strip()
            if not is_clean_pair(ko, en, min_chars, MAX_CHARS):
                continue
            key = (ko, en)
            if key in seen or ko in seen_src:
                continue
            seen.add(key)
            seen_src.add(ko)
            out.append((ko, en))
    return out, total


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    ap = argparse.ArgumentParser(description="AI Hub 기술과학 한영 CSV 정제")
    ap.add_argument("--root", required=True, help="데이터셋 루트(Training/Validation 포함)")
    ap.add_argument("--out-dir", default="data/processed/aihub_tech")
    ap.add_argument("--min-chars", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    train_csv = find_csv(root / "Training")
    val_csv = find_csv(root / "Validation")
    print(f"train csv: {train_csv.name}")
    print(f"val   csv: {val_csv.name}")

    train, n_train = convert_csv(train_csv, args.min_chars)
    val, n_val = convert_csv(val_csv, args.min_chars)

    write_tsv(out_dir / "train.tsv", train)
    write_tsv(out_dir / "val.tsv", val)
    print(f"train: {n_train} -> {len(train)} ({len(train)/max(n_train,1)*100:.1f}% 유지)")
    print(f"val:   {n_val} -> {len(val)}")
    print(f"saved: {out_dir}/train.tsv, {out_dir}/val.tsv")


if __name__ == "__main__":
    main()
