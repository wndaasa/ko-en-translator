# 프로젝트 가이드 (CLAUDE.md)

한국어↔영어 번역용 Transformer를 PyTorch로 바닥부터 구현하는 프로젝트. 이 문서는 저장소에서
작업할 때의 환경·명령·구조·규약을 정리한다. (개요·결과는 [README.md](README.md), 단계별 상세는
[docs/](docs/) 참고.)

## 실행 환경

- 개발 GPU: **NVIDIA RTX 4090 (24GB)** — RunPod 클라우드 Linux 컨테이너, CUDA 12.8.
- Python 3.12, PyTorch 2.8.0(cu128). 시스템 python 그대로 사용(conda 없음), 환경변수 불필요.
- 의존성: `pip install torch tokenizers sacrebleu`.

> Stage 0~5의 구 로컬 환경(AMD RX 7800 XT + WSL2 + ROCm, `HSA_ENABLE_DXG_DETECTION=1` 필요)은
> [docs/environment-setup.md](docs/environment-setup.md)에 기록으로만 남김. 현 환경에선 해당 없음.

## 저장소 구조

```
src/         tokenizer.py model.py minrnn.py data.py train.py translate.py
             prepare_data.py prepare_aihub.py prepare_mixed.py prepare_large.py prepare_context.py
             char_tokenizer.py char_encoder.py (Stage 3 MCE)
tests/       test_model.py test_mce.py test_minrnn.py(단위) check_overfit.py(토이 overfit) benchmark_efficiency.py
scripts/     00~02 구 로컬 환경(WSL+ROCm) 셋업(쉘) — 현 환경에선 불필요
docs/        environment-setup / stage1 / stage2 / roadmap-stage3-mce / stage4-minrnn / stage5-domain-mix / stage6-context
data/        toy_parallel.tsv(소스). processed/·raw_opus/는 생성물 → .gitignore
runs/        학습 산출물(가중치) → .gitignore (HF로 별도 배포)
```

## 데이터 준비 (저장소에 미포함, 재생성)

대용량·라이선스 이슈로 코퍼스/가중치는 커밋하지 않는다.

```bash
# OPUS-100 (일반/대화) 다운로드·정제
python -m src.prepare_data --out-dir data/processed

# AI Hub '기술과학' 한영(논문) 정제 — 데이터는 aihub.or.kr 에서 직접 받아야 함(재배포 금지)
python -m src.prepare_aihub --root "<AI Hub 데이터 루트>" --out-dir data/processed/aihub_tech

# (Stage 6) OPUS moses 대용량 코퍼스(17M쌍) 다운로드·정제·중복제거
python -m src.prepare_large --out-dir data/processed/large

# (Stage 6) 문서 경계 보존 문맥 윈도우 생성 (moses 빈 줄 경계 / AI Hub CSV --csv)
python -m src.prepare_context --help
```

## 주요 명령

```bash
# 토크나이저 학습(공용 BPE)
python -m src.tokenizer --data data/processed/train.tsv --out runs/base/tokenizer.json --vocab-size 32000

# 기반 학습(양방향) / 중단·이어하기(--resume) / 파인튜닝(--init-from) / 아키텍처(--arch bpe|mce|minrnn)
python -m src.train --arch minrnn --epochs 5 --batch-size 64 --artifacts runs/base
python -m src.train --init-from runs/base/best.pt \
    --train-data data/processed/aihub_tech/train.tsv --val-data data/processed/aihub_tech/val.tsv \
    --artifacts runs/finetune --max-val 1000

# (Stage 6) 문맥 파인튜닝: 문서모드 + 단방향 + compile
python -m src.train --arch minrnn --init-from runs/big/best.pt --context --no-bidi --compile \
    --max-len 640 --batch-size 12 --artifacts runs/ctx_paper

# 번역(방향 태그) / 테스트
python -m src.translate --artifacts runs/finetune --to en --text "..."
python -m tests.test_model
```

## 작업 규약

- **최소 변경**: 요청에 직접 연결되는 코드만. 베이스라인 경로는 보존(새 기능은 플래그로 추가).
- **검증 우선**: 변경 후 단위 테스트 → 토이 overfit → 본 학습 순으로 작게 확인.
- **문서 유지**: 단계가 끝나면 `docs/`에 무엇을·왜·정량 결과를 기록. README는 한국어, 현재 상태만 기술.
- 주석은 "왜"를 설명(자명한 코드 반복 금지).

## 진행 상태 / 로드맵

- ✅ Stage 0 환경 셋업 · ✅ Stage 1 트랜스포머 직접 구현 · ✅ Stage 2 실코퍼스 학습+논문 파인튜닝
- ✅ Stage 3 MCE(음성 결과) · ✅ Stage 4 minRNN(양성, 이후 주력 아키텍처) · ✅ Stage 5 도메인 혼합
- ✅ Stage 6 문서/문맥 단위 번역 — 논문 도메인 ko→en 문서모드 BLEU 35.46. 상세: [docs/stage6-context.md](docs/stage6-context.md)
- ▶ 다음: 가중치 HF 배포, 이후 발전 모델은 회사 GPU 자원으로 진행 예정(서비스 탑재 목표).

## 알아둘 함정

- 24GB(4090) 기준 배치: 문장 학습(max_len 128) **64**, 문맥 학습은 max_len 1024 → **4**
  (+`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), max_len 640 → **12**. 크로스어텐션이
  O(L·S)라 max_len을 키우면 배치를 크게 줄여야 한다.
- 조기종료·베스트 판정은 노이즈 큰 BLEU가 아니라 **val_loss** 기준으로. inverse-sqrt 스케줄을
  `--resume`으로 이어붙일 땐 **동일 warmup**을 명시해야 lr 궤적이 유지된다.
- 문서모드 번역은 `greedy_translate`가 아니라 `greedy_translate_context`(k-to-k, 입력 문장 수만큼
  `<eos>` 생성 후 정지)를 써야 한다 — 전자는 첫 `<eos>`에서 멈춰 첫 문장만 나온다.
- `python ... | tee` 사용 시 `set -o pipefail`로 종료코드 가림 방지.
- (구 로컬 환경 한정) WSL+ROCm은 `librocdxg` + `HSA_ENABLE_DXG_DETECTION=1` 필요, 배치 96/16GB,
  `expandable_segments` 미지원 — [docs/environment-setup.md](docs/environment-setup.md) 참고.
