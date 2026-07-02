"""Phase 2: 문서/문맥 단위 번역 데이터 생성 (context-concatenation).

OPUS moses 파일은 빈 줄로 문서(강연·자막 등) 경계를 표시한다. 이 경계를 살려
문서 내 연속 문장을 슬라이딩 윈도우로 이어붙여 (ko_context, en_context) 쌍을
만든다. 앞뒤 문맥이 한 시퀀스에 들어가므로 용어·대명사·문체 일관성을 학습할 수
있고, 이것이 minRNN의 긴 컨텍스트(O(L)) 이점을 쓰는 지점이다.

정제는 prepare_data.is_clean_pair 를 쌍 단위로 재사용한다(한쪽만 지워 정렬이
깨지지 않도록 ko/en 을 함께 버린다). 윈도우 결합 후 전역 중복을 제거한다.

  python -m src.prepare_context \
      --src-en data/raw_opus/TED2020.en-ko.en \
      --src-ko data/raw_opus/TED2020.en-ko.ko \
      --out-dir data/processed/ctx_ted --window 5 --stride 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import CONTEXT_SEP
from .prepare_data import is_clean_pair, write_tsv


def read_documents_csv(csv_path: str) -> list[list[tuple[str, str]]]:
    """AI Hub 기술과학 CSV → [문서][ (ko,en) ]. file_name=논문(문서), sn=문장순서.

    CSV는 논문별로 섞여 있으므로 file_name 으로 묶고 sn 으로 정렬해 원문 흐름을 복원한다.
    """
    import csv as _csv
    from collections import defaultdict
    _csv.field_size_limit(10 ** 7)
    docs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            docs[row["file_name"]].append((row["sn"], row["ko"].strip(), row["en"].strip()))
    out: list[list[tuple[str, str]]] = []
    for sents in docs.values():
        sents.sort(key=lambda x: x[0])  # sn 정렬 = 논문 내 문장 순서
        out.append([(ko, en) for _, ko, en in sents])
    return out


def read_documents(path_en: str, path_ko: str) -> list[list[tuple[str, str]]]:
    """빈 줄을 문서 경계로 보고 [문서][ (ko,en) ] 를 만든다.

    moses 는 1:1 정렬이라 en/ko 라인 수와 빈 줄 위치가 같다. 문서별로 zip 한다.
    """
    with open(path_en, encoding="utf-8") as f:
        en_lines = [ln.rstrip("\n") for ln in f]
    with open(path_ko, encoding="utf-8") as f:
        ko_lines = [ln.rstrip("\n") for ln in f]
    if len(en_lines) != len(ko_lines):
        raise ValueError(f"en/ko 라인 수 불일치: {len(en_lines)} vs {len(ko_lines)}")

    docs: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for en, ko in zip(en_lines, ko_lines):
        if not en.strip() and not ko.strip():  # 문서 경계
            if cur:
                docs.append(cur)
                cur = []
            continue
        cur.append((ko.strip(), en.strip()))
    if cur:
        docs.append(cur)
    return docs


def clean_document(doc: list[tuple[str, str]], min_chars: int, max_chars: int
                   ) -> list[tuple[str, str]]:
    """문서 내 문장쌍을 쌍 단위로 정제. 순서(문맥)는 보존한다."""
    return [(ko, en) for ko, en in doc if is_clean_pair(ko, en, min_chars, max_chars)]


def make_windows(doc: list[tuple[str, str]], window: int, stride: int,
                 sep: str, max_chunk_chars: int) -> list[tuple[str, str]]:
    """문서를 슬라이딩 윈도우로 묶어 (ko_chunk, en_chunk) 리스트 생성.

    window=결합할 문장 수, stride=이동 폭. k-to-k(입력 k문장→출력 k문장).
    결합 후 어느 한쪽이라도 max_chunk_chars 를 넘으면 학습 max_len 을 초과할
    위험이 크므로 그 윈도우는 버린다(잘라내면 ko/en 문장 경계가 어긋난다).
    """
    out: list[tuple[str, str]] = []
    n = len(doc)
    if n == 0:
        return out
    for start in range(0, n, stride):
        chunk = doc[start:start + window]
        if not chunk:
            break
        ko_chunk = sep.join(ko for ko, _ in chunk)
        en_chunk = sep.join(en for _, en in chunk)
        if len(ko_chunk) > max_chunk_chars or len(en_chunk) > max_chunk_chars:
            continue
        out.append((ko_chunk, en_chunk))
        if start + window >= n:  # 문서 끝 도달(마지막 부분 윈도우 포함 후 종료)
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="문서/문맥 단위 번역 데이터 생성")
    ap.add_argument("--src-en", help="moses .en (빈 줄=문서경계)")
    ap.add_argument("--src-ko", help="moses .ko (빈 줄=문서경계)")
    ap.add_argument("--csv", help="AI Hub CSV (file_name=문서, sn=문장순서). moses 대신 사용.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=5, help="결합할 연속 문장 수 k")
    ap.add_argument("--stride", type=int, default=3, help="윈도우 이동 폭")
    ap.add_argument("--sep", default=CONTEXT_SEP,
                    help="문장 결합 구분자(기본: 문서모드 마커. 학습 시 <eos>로 치환됨)")
    ap.add_argument("--min-chars", type=int, default=2, help="문장별 최소 길이")
    ap.add_argument("--max-chars", type=int, default=200, help="문장별 최대 길이")
    ap.add_argument("--max-chunk-chars", type=int, default=1500,
                    help="윈도우 결합 후 한쪽 최대 문자 수(초과 시 버림)")
    ap.add_argument("--val-docs", type=int, default=50,
                    help="검증용으로 뒤에서 뗄 문서 수(문서 단위 held-out)")
    args = ap.parse_args()

    if args.csv:
        docs = read_documents_csv(args.csv)
    else:
        docs = read_documents(args.src_en, args.src_ko)
    docs = [clean_document(d, args.min_chars, args.max_chars) for d in docs]
    docs = [d for d in docs if d]  # 정제 후 빈 문서 제거
    print(f"문서 수(정제 후): {len(docs)} | 문장쌍 총: {sum(len(d) for d in docs):,}")

    # 문서 단위로 train/val 분리(문맥 누수 방지: 같은 문서가 양쪽에 걸치지 않게)
    val_docs = docs[-args.val_docs:] if args.val_docs else []
    train_docs = docs[:-args.val_docs] if args.val_docs else docs

    def build(dset: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        rows: list[tuple[str, str]] = []
        for d in dset:
            for ko, en in make_windows(d, args.window, args.stride, args.sep,
                                        args.max_chunk_chars):
                if (ko, en) in seen:
                    continue
                seen.add((ko, en))
                rows.append((ko, en))
        return rows

    train = build(train_docs)
    val = build(val_docs)

    out_dir = Path(args.out_dir)
    write_tsv(out_dir / "train.tsv", train)
    write_tsv(out_dir / "val.tsv", val)
    print(f"window={args.window} stride={args.stride} | "
          f"train 윈도우 {len(train):,} | val 윈도우 {len(val):,}")
    print(f"saved: {out_dir}/train.tsv, {out_dir}/val.tsv")


if __name__ == "__main__":
    main()
