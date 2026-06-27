"""OPUS-100 (en-ko) 다운로드 + 정제 → train/val TSV 산출.

OPUS-100은 자막/웹 기반이라 노이즈가 많다(미번역 쌍, 괄호 효과음, 정렬 오류 등).
보수적인 규칙 필터로 학습에 해로운 쌍을 걸러낸다.

  python -m src.prepare_data --out-dir data/processed
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
# 괄호 안 효과음/지문 위주 라인 (예: "( tires squealing )")
BRACKET_ONLY = re.compile(r"^[\s\(\[\)\]\-—:]*$")


def is_clean_pair(ko: str, en: str, min_chars: int, max_chars: int) -> bool:
    ko, en = ko.strip(), en.strip()
    if not ko or not en:
        return False
    if ko == en:  # 미번역(양쪽 동일) — 효과음/고유명사 라인
        return False
    if not HANGUL.search(ko):  # 한국어측에 한글이 없음
        return False
    if not LATIN.search(en):  # 영어측에 라틴 문자가 없음
        return False
    if len(ko) < min_chars or len(en) < min_chars:
        return False
    if len(ko) > max_chars or len(en) > max_chars:
        return False
    # 길이비 필터: 정렬 오류로 한쪽만 비정상적으로 긴 쌍 제거
    ratio = len(en) / max(len(ko), 1)
    if ratio < 0.5 or ratio > 5.0:
        return False
    # 괄호/구두점만 있는 라인 제거
    hangul_strip = BRACKET_ONLY.sub("", ko)
    if not HANGUL.search(hangul_strip):
        return False
    return True


def clean_split(rows, min_chars: int, max_chars: int) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    seen_src: set[str] = set()
    out: list[tuple[str, str]] = []
    for ex in rows:
        t = ex["translation"]
        ko = t["ko"].replace("\t", " ").replace("\n", " ").strip()
        en = t["en"].replace("\t", " ").replace("\n", " ").strip()
        if not is_clean_pair(ko, en, min_chars, max_chars):
            continue
        key = (ko, en)
        if key in seen or ko in seen_src:  # 완전중복 + 동일 소스 중복 제거
            continue
        seen.add(key)
        seen_src.add(ko)
        out.append((ko, en))
    return out


def write_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ko, en in pairs:
            f.write(f"{ko}\t{en}\n")


def main() -> None:
    import datasets

    ap = argparse.ArgumentParser(description="OPUS-100 en-ko 다운로드/정제")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print("loading OPUS-100 en-ko ...")
    dd = datasets.load_dataset("Helsinki-NLP/opus-100", "en-ko")

    train = clean_split(dd["train"], args.min_chars, args.max_chars)
    val = clean_split(dd["validation"], args.min_chars, args.max_chars)
    if args.max_train and len(train) > args.max_train:
        train = train[: args.max_train]

    write_tsv(out_dir / "train.tsv", train)
    write_tsv(out_dir / "val.tsv", val)

    raw_train = dd["train"].num_rows
    print(f"train: {raw_train} -> {len(train)} ({len(train)/raw_train*100:.1f}% 유지)")
    print(f"val:   {dd['validation'].num_rows} -> {len(val)}")
    print(f"saved: {out_dir}/train.tsv, {out_dir}/val.tsv")


if __name__ == "__main__":
    main()
