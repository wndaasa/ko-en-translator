# 프로젝트 가이드 (CLAUDE.md)

한국어↔영어 번역용 Transformer를 PyTorch로 바닥부터 구현하는 프로젝트. 이 문서는 저장소에서
작업할 때의 환경·명령·구조·규약을 정리한다. (개요·결과는 [README.md](README.md), 단계별 상세는
[docs/](docs/) 참고.)

## 실행 환경

- 개발 GPU: **AMD Radeon RX 7800 XT (RDNA3, gfx1101)** — CUDA 불가라 **WSL2(Ubuntu 24.04) + ROCm 7.2.1 + ROCDXG**로 구성.
- Python 3.12 (conda env `tf`), PyTorch 2.9.1(ROCm 휠). 구축 절차·트러블슈팅: [docs/environment-setup.md](docs/environment-setup.md).
- **GPU 코드 실행 시 `HSA_ENABLE_DXG_DETECTION=1` 필수** (ROCm 7.13 미만 WSL 요구):

```bash
HSA_ENABLE_DXG_DETECTION=1 python -m src.train ...
```

> NVIDIA/CPU 환경이면 표준 PyTorch만 설치하면 되고, 위 환경변수는 불필요하다.

## 저장소 구조

```
src/         tokenizer.py model.py data.py prepare_data.py prepare_aihub.py train.py translate.py
tests/       test_model.py(단위) check_overfit.py(토이 overfit)
scripts/     00~02 환경 셋업(쉘)
docs/        environment-setup / stage1 / stage2 / roadmap-stage3-mce
data/        toy_parallel.tsv(소스). processed/는 생성물 → .gitignore
runs/        학습 산출물(가중치) → .gitignore (HF로 별도 배포 예정)
```

## 데이터 준비 (저장소에 미포함, 재생성)

대용량·라이선스 이슈로 코퍼스/가중치는 커밋하지 않는다.

```bash
# OPUS-100 (일반/대화) 다운로드·정제
python -m src.prepare_data --out-dir data/processed

# AI Hub '기술과학' 한영(논문) 정제 — 데이터는 aihub.or.kr 에서 직접 받아야 함(재배포 금지)
python -m src.prepare_aihub --root "<AI Hub 데이터 루트>" --out-dir data/processed/aihub_tech
```

## 주요 명령

```bash
# 토크나이저 학습(공용 BPE)
python -m src.tokenizer --data data/processed/train.tsv --out runs/base/tokenizer.json --vocab-size 32000

# 기반 학습(OPUS 양방향) / 중단·이어하기(--resume) / 파인튜닝(--init-from)
HSA_ENABLE_DXG_DETECTION=1 python -m src.train --epochs 5 --batch-size 96 --artifacts runs/base
HSA_ENABLE_DXG_DETECTION=1 python -m src.train --init-from runs/base/best.pt \
    --train-data data/processed/aihub_tech/train.tsv --val-data data/processed/aihub_tech/val.tsv \
    --artifacts runs/finetune --max-val 1000

# 번역(방향 태그) / 테스트
HSA_ENABLE_DXG_DETECTION=1 python -m src.translate --artifacts runs/finetune --to en --text "..."
python -m tests.test_model
```

## 작업 규약

- **최소 변경**: 요청에 직접 연결되는 코드만. 베이스라인 경로는 보존(새 기능은 플래그로 추가).
- **검증 우선**: 변경 후 단위 테스트 → 토이 overfit → 본 학습 순으로 작게 확인.
- **문서 유지**: 단계가 끝나면 `docs/`에 무엇을·왜·정량 결과를 기록. README는 한국어, 현재 상태만 기술.
- 주석은 "왜"를 설명(자명한 코드 반복 금지).

## 진행 상태 / 로드맵

- ✅ Stage 0 환경 셋업 · ✅ Stage 1 트랜스포머 직접 구현 · ✅ Stage 2 실코퍼스 학습+논문 파인튜닝
- ▶ Stage 3 (계획) **MCE: 형태소 합성 인코더** — 저빈도 전문어 오역 개선 가설 검증. 설계: [docs/roadmap-stage3-mce.md](docs/roadmap-stage3-mce.md).

## 알아둘 함정

- WSL+ROCm은 `librocdxg` 별도 설치 + `HSA_ENABLE_DXG_DETECTION=1` 필요(미설치 시 `hipErrorNoDevice`).
- 학습 배치는 16GB 기준 **96** 권장(128은 OOM 이력). ROCm은 `expandable_segments` 미지원.
- `python ... | tee` 사용 시 `set -o pipefail`로 종료코드 가림 방지.
