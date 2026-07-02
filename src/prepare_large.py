"""대용량 ko-en 코퍼스 준비 — OPUS moses 직접 다운로드 + 정제 + 전역 중복제거.

설치된 datasets(5.x)는 스크립트 기반 데이터셋(NLLB·CCMatrix·OpenSubtitles)을 더 이상
로드하지 못하므로, OPUS의 moses 포맷(zip 안에 .en/.ko 평행 텍스트)을 URL로 직접 받아
prepare_data.py의 정제 규칙을 그대로 적용한다. 여러 코퍼스를 합치며 소스 기준 전역 중복을
제거하고, corpus당 cap으로 총량을 조절한다(기본 목표 ~8M 깨끗한 쌍).

  python -m src.prepare_large --corpora NLLB CCMatrix OpenSubtitles ParaCrawl \
      --cap NLLB=3000000 CCMatrix=3000000 OpenSubtitles=2500000 ParaCrawl=2000000 \
      --out-dir data/processed/large

코퍼스/가중치는 대용량·라이선스 이슈로 저장소에 커밋하지 않는다(재생성).
"""
from __future__ import annotations

import argparse
import random
import subprocess
import zipfile
from pathlib import Path

from .prepare_data import is_clean_pair, write_tsv

# OPUS 코퍼스 → (corpus 이름, 버전). en-ko moses zip URL은 아래 패턴으로 생성한다.
#   https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/en-ko.txt.zip
# 버전은 2026-06 기준 OPUS에 존재하는 것으로 확인된 값.
OPUS_CORPORA: dict[str, tuple[str, str]] = {
    "NLLB": ("NLLB", "v1"),
    "CCMatrix": ("CCMatrix", "v1"),
    "OpenSubtitles": ("OpenSubtitles", "v2024"),
    "ParaCrawl": ("ParaCrawl", "v9"),
    "WikiMatrix": ("WikiMatrix", "v1"),
    "TED2020": ("TED2020", "v1"),
}

BASE = "https://object.pouta.csc.fi/OPUS-{corpus}/{ver}/moses/en-ko.txt.zip"


def opus_url(name: str) -> str:
    corpus, ver = OPUS_CORPORA[name]
    return BASE.format(corpus=corpus, ver=ver)


def download(name: str, raw_dir: Path) -> Path:
    """moses zip을 raw_dir에 받는다(이미 있으면 건너뜀). curl -C 로 이어받기 지원."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{name}.en-ko.txt.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"  [{name}] zip 이미 존재 → 다운로드 생략 ({zip_path.stat().st_size/1e6:.0f} MB)")
        return zip_path
    url = opus_url(name)
    print(f"  [{name}] 다운로드 중: {url}")
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3",
                    "-C", "-", "-o", str(zip_path), url], check=True)
    return zip_path


def extract_pair_files(name: str, zip_path: Path, raw_dir: Path) -> tuple[Path, Path]:
    """zip에서 .en/.ko 평행 텍스트만 추출. (en_path, ko_path) 반환."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        en_name = next(n for n in names if n.endswith(".en"))
        ko_name = next(n for n in names if n.endswith(".ko"))
        en_path = raw_dir / Path(en_name).name
        ko_path = raw_dir / Path(ko_name).name
        for member, dst in ((en_name, en_path), (ko_name, ko_path)):
            if dst.exists() and dst.stat().st_size > 0:
                continue
            with zf.open(member) as src, open(dst, "wb") as out:
                # 청크 복사(대용량 메모리 안전)
                while chunk := src.read(1 << 20):
                    out.write(chunk)
    return en_path, ko_path


def iter_clean_pairs(en_path: Path, ko_path: Path, min_chars: int, max_chars: int):
    """en/ko 평행 파일을 줄 단위로 함께 읽으며 정제 통과한 (ko, en)만 산출."""
    with open(en_path, encoding="utf-8") as fe, open(ko_path, encoding="utf-8") as fk:
        for en, ko in zip(fe, fk):
            en = en.replace("\t", " ").strip()
            ko = ko.replace("\t", " ").strip()
            if is_clean_pair(ko, en, min_chars, max_chars):
                yield ko, en


def main() -> None:
    ap = argparse.ArgumentParser(description="OPUS 대용량 ko-en 다운로드/정제/병합")
    ap.add_argument("--corpora", nargs="+", default=["NLLB", "CCMatrix", "OpenSubtitles", "ParaCrawl"],
                    choices=list(OPUS_CORPORA), help="병합할 OPUS 코퍼스(우선순위=순서).")
    ap.add_argument("--cap", nargs="*", default=[],
                    help="corpus당 최대 채택 쌍 수. 예: NLLB=3000000 (지정 안 하면 --cap-default).")
    ap.add_argument("--cap-default", type=int, default=3000000, help="--cap에 없는 코퍼스의 기본 cap.")
    ap.add_argument("--out-dir", default="data/processed/large")
    ap.add_argument("--raw-dir", default="data/raw_opus", help="zip/추출본 캐시(저장소 밖, /workspace).")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=5000, help="검증셋 홀드아웃 쌍 수.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    caps: dict[str, int] = {}
    for item in args.cap:
        k, v = item.split("=")
        caps[k] = int(v)

    raw_dir = Path(args.raw_dir)
    rng = random.Random(args.seed)

    seen_src: set[int] = set()   # 소스(ko) 해시 — 코퍼스 간 전역 중복 제거
    seen_pair: set[int] = set()  # (ko,en) 해시 — 완전중복 제거
    pairs: list[tuple[str, str]] = []
    stats: list[tuple[str, int]] = []

    for name in args.corpora:
        cap = caps.get(name, args.cap_default)
        zip_path = download(name, raw_dir)
        en_path, ko_path = extract_pair_files(name, zip_path, raw_dir)
        kept = 0
        for ko, en in iter_clean_pairs(en_path, ko_path, args.min_chars, args.max_chars):
            hs, hp = hash(ko), hash((ko, en))
            if hs in seen_src or hp in seen_pair:
                continue
            seen_src.add(hs)
            seen_pair.add(hp)
            pairs.append((ko, en))
            kept += 1
            if kept >= cap:
                break
        stats.append((name, kept))
        print(f"  [{name}] 채택 {kept:,} 쌍 (cap {cap:,}) | 누적 {len(pairs):,}")

    rng.shuffle(pairs)
    val = pairs[: args.val_size]
    train = pairs[args.val_size:]

    out_dir = Path(args.out_dir)
    write_tsv(out_dir / "train.tsv", train)
    write_tsv(out_dir / "val.tsv", val)

    print("\n=== 요약 ===")
    for name, kept in stats:
        print(f"  {name:<14} {kept:>12,}")
    print(f"  {'전역중복제거후':<14} {len(pairs):>12,}")
    print(f"  train {len(train):,} | val {len(val):,}")
    print(f"  saved: {out_dir}/train.tsv, {out_dir}/val.tsv")


if __name__ == "__main__":
    main()
