# 한 <-> 영 번역 Transformer

PyTorch로 Transformer를 **직접 구현**하는 프로젝트입니다.
주 목적은 **한국어 <-> 영어 번역 전용 소형 모델**이며, 부차적으로 간단한 대화도 가능하도록 합니다.

> 이 프로젝트의 모델은 "작고 빠른 번역기"에 초점을 둡니다.

## 목표와 범위

- **1순위: 한↔영 번역.** 작은 크기에서 번역 품질을 우선.
- **2순위: 간단한 대화.** 짧은 멀티턴 수준.
- 비목표: 대용량 컨텍스트(롱컨텍스트), RAG, 범용 챗봇.

긴 문서(예: 50페이지 논문) 번역은 모델이 통째로 읽는 방식이 아니라,
**문장/문단 단위로 분할 → 각 조각 번역 → 재조립**하는 청킹 파이프라인으로 처리합니다
(어텐션의 $O(n^2)$ 비용 때문에 소형 모델은 짧은 컨텍스트를 유지).

## 아키텍처 결정: 인코더-디코더 (원조 Transformer)

번역이 1순위이므로 *Attention is All You Need*(2017)의 **인코더-디코더(seq2seq)** 구조를 채택합니다.

| 구분 | 인코더-디코더 (채택) | 디코더-온리 (GPT) |
|---|---|---|
| 번역 적합성 | 높음 — 입력 전체를 인코더가 이해 후 디코더가 생성 | 보통 — 조건부 생성(프롬프트)으로 학습 |
| 소형 모델 번역 품질 | 유리 (파라미터 효율적) | 상대적으로 불리 |
| 학습 효과 | 인코더/디코더 셀프어텐션 + **크로스어텐션** 전부 구현 | 단일 스택 |
| 대화 처리 | seq2seq(입력→응답) | 더 자연스러움 |

대화는 seq2seq 형태(입력→응답, T5/mT5 방식)로 처리합니다.

## 개발 환경

| 항목 | 내용 |
|---|---|
| GPU | AMD Radeon RX 7800 XT (RDNA3, gfx1101, 16GB) |
| 백엔드 | WSL2(Ubuntu 24.04) + ROCm 7.2.1 + ROCDXG (CUDA 불가, AMD GPU) |
| 프레임워크 | PyTorch 2.9.1 (ROCm 빌드), Python 3.12 (conda env `tf`) |

> AMD GPU는 Windows에서 CUDA를 쓸 수 없어 WSL2 + ROCm로 구성했습니다.
> 구축 과정과 트러블슈팅은 [docs/environment-setup.md](docs/environment-setup.md)에 정리되어 있습니다.

## 진행 상태

- [x] **Stage 0 — 환경 셋업** (완료): WSL2 + ROCm + ROCDXG, GPU 학습 동작 검증
- [x] **Stage 1 — 트랜스포머 직접 구현** (완료): 한·영 공용 BPE → 인코더-디코더 모델 → 학습 루프 → 토이 overfit 검증 (loss 6.52→0.00, 학습셋 exact-match 100%). 상세: [docs/stage1-implementation.md](docs/stage1-implementation.md)
- [x] **Stage 2 — 실코퍼스 번역 학습** (완료): OPUS-100 양방향 기반 학습 + AI Hub 기술과학 논문 파인튜닝. ko→en BLEU 7.36(일반) → **44.10(학술)**. 긴 문단은 문장 청킹으로 번역. 상세: [docs/stage2-translation.md](docs/stage2-translation.md)
  - 산출물: `runs/base/best.pt`(일반·대화), `runs/finetune/best.pt`(논문 특화). 학술 파인튜닝은 catastrophic forgetting으로 대화 능력을 잃어 **2-모델 구성**으로 사용.
- [x] **Stage 3 — MCE (형태소 합성 인코더)** (완료, **음성 결과**): 플랫 BPE 대신 문자(음절)→단어 CharCNN 합성 인코더를 직접 구현해 베이스라인과 동일 조건으로 비교. 동일 step에서 MCE가 val_loss·BLEU 모두 일관되게 뒤지고 학습은 ~5배 느림 → 가설 미지지(BPE가 강한 베이스라인). "가설→공정한 실험→분석"의 정직한 음성 결과. 상세: [docs/roadmap-stage3-mce.md](docs/roadmap-stage3-mce.md)
- [x] **Stage 4 — minRNN (선형 순환 seq2seq)** (완료, **양성 결과**): 어텐션을 minGRU 선형 순환으로 대체(학습 병렬·추론 O(1)). 동일 파라미터(60.5M)에서 **품질은 동일 step 기준 트랜스포머를 앞서고**, 디코딩 길이 L=4096에서 **메모리 ~2.5배 적고 속도 ~2.2배 빠름**(O(L) vs O(L²)). 상세: [docs/stage4-minrnn.md](docs/stage4-minrnn.md)
- [x] **Stage 5 — 도메인 혼합 학습** (완료, **양성 결과**): 일상(OPUS)+격식(AI Hub) 균형 혼합 + 도메인 태그(`<casual>`/`<formal>`)로 minRNN 학습, 도메인별 BLEU 측정. 격식 추가 후에도 일상체가 단일도메인보다 오히려 높음(positive transfer) → **작은 모델이 두 레지스터를 망각 없이 공존**. 상세: [docs/stage5-domain-mix.md](docs/stage5-domain-mix.md)

> **데이터·가중치는 저장소에 포함하지 않습니다.** 코퍼스는 대용량·라이선스(AI Hub 재배포 금지) 이슈로 `prepare_data.py`/`prepare_aihub.py`로 재생성하며, 학습 가중치(`runs/`)는 추후 Hugging Face로 별도 배포 예정입니다. 데이터 준비 명령은 [CLAUDE.md](CLAUDE.md) 참고.

## 환경 재현

WSL2(Ubuntu 24.04)와 Windows용 AMD Adrenalin 드라이버가 설치된 상태에서:

```bash
# 1) ROCm 설치 (sudo 필요)
bash scripts/00_install_rocm.sh
# 2) librocdxg 설치 — WSL에서 GPU를 잡기 위한 핵심 (sudo 필요)
bash scripts/01_install_rocdxg.sh
# 3) GPU 동작 검증
bash scripts/02_verify_gpu.sh
```

자세한 단계와 발생 가능한 문제는 [docs/environment-setup.md](docs/environment-setup.md) 참고.

## 사용법 (Stage 1, 토이 데이터)

```bash
# WSL(Ubuntu)에서. GPU 사용 시 HSA_ENABLE_DXG_DETECTION=1 필수.
PY=~/miniconda3/envs/tf/bin/python

# 학습 (토이 병렬셋 overfit)
HSA_ENABLE_DXG_DETECTION=1 $PY -m src.train --dropout 0.0 --epochs 400

# 번역
HSA_ENABLE_DXG_DETECTION=1 $PY -m src.translate --text "안녕하세요"

# 단위 테스트 / overfit 검증
$PY -m tests.test_model
HSA_ENABLE_DXG_DETECTION=1 $PY -m tests.check_overfit
```

## 프로젝트 구조

```
Transformer/
├─ README.md
├─ docs/
│  ├─ environment-setup.md      # 환경 구축 + 트러블슈팅 기록
│  ├─ stage1-implementation.md  # Stage 1 구현/결정/결과
│  └─ stage2-translation.md     # Stage 2 실코퍼스 학습/파인튜닝/결과
├─ scripts/                     # 환경 셋업 스크립트 (00~02)
├─ src/
│  ├─ tokenizer.py              # 한·영 공용 ByteLevel BPE (방향 태그 포함)
│  ├─ model.py                  # 인코더-디코더 트랜스포머 (from scratch)
│  ├─ data.py                   # 병렬 코퍼스 로딩/배치 (양방향 MTDataset, MCEDataset)
│  ├─ prepare_data.py           # OPUS-100 다운로드/정제
│  ├─ prepare_aihub.py          # AI Hub 기술과학 CSV 정제
│  ├─ prepare_mixed.py          # (Stage 5) 도메인 혼합 코퍼스 생성
│  ├─ char_tokenizer.py         # (Stage 3) 문자(음절) 토크나이저
│  ├─ char_encoder.py           # (Stage 3) CharCNN 합성기 + MCE 트랜스포머
│  ├─ minrnn.py                 # (Stage 4) minGRU 선형순환 seq2seq
│  ├─ train.py                  # 학습 루프 (resume·init-from·--arch bpe/mce/minrnn)
│  └─ translate.py              # greedy 디코딩 번역 (bpe/mce/minrnn)
├─ tests/
│  ├─ test_model.py             # shape/causality 단위 테스트
│  ├─ test_mce.py               # MCE 구조 단위 테스트
│  ├─ test_minrnn.py            # minRNN 구조 단위 테스트
│  ├─ benchmark_efficiency.py   # 디코딩 메모리/속도 벤치 (Transformer vs minRNN)
│  └─ check_overfit.py          # 토이 overfit exact-match 검증
├─ data/
│  ├─ toy_parallel.tsv          # 한-영 토이 병렬 코퍼스 (Stage 1)
│  └─ processed/                # OPUS·AI Hub 정제 TSV (Stage 2, 생성물)
├─ artifacts/                   # Stage 1 토이 산출물
└─ runs/                        # Stage 2 산출물 (base/, finetune/)
```
